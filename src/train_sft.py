"""SFT training script for emoji-llm using Tinker.

Fine-tunes Qwen3.5-4B with LoRA to respond in emoji-only.
Uses the pre-generated SFT dataset (data/sft_train.jsonl, data/sft_test.jsonl).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import tinker
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import PreTrainedTokenizer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3.5-4B"

# Shorter system prompt used in training data
SYSTEM_PROMPT = (
    "You communicate exclusively using emoji. No text, numbers, or punctuation ever. "
    "Use 2-8 emoji per response that capture the core meaning, emotion, and key concepts "
    "of the user's message."
)

# Sample prompts to test the model during training
EVAL_PROMPTS = [
    "I just got promoted at work!",
    "My dog passed away today",
    "What's the weather like outside?",
    "Can you recommend a good Italian restaurant?",
    "I'm moving to a new city and I'm nervous",
    "Happy birthday!",
    "The sunset over the ocean was beautiful",
    "I'm so hungry I could eat a horse",
]


def load_conversations(path: str) -> list[list[dict]]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            conversations.append(row["messages"])
    return conversations


def conversation_to_datum(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> tinker.Datum | None:
    """Convert a conversation to a Tinker Datum for supervised training.

    Tokenizes using the chat template. Sets loss weights to 1 only on the
    assistant response tokens (the emoji output we want the model to learn).
    """
    # Build the full conversation token sequence using the chat template
    # We need to figure out which tokens are the assistant's response
    # Strategy: tokenize with and without the assistant turn, diff to find boundary

    # Full conversation (all messages)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    # Everything except the last assistant message
    context_messages = messages[:-1]
    context_text = tokenizer.apply_chat_template(context_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    context_tokens = tokenizer.encode(context_text, add_special_tokens=False)

    if len(full_tokens) > max_length:
        return None

    n_total = len(full_tokens)
    n_context = len(context_tokens)

    # Input tokens: all but last (teacher forcing)
    # Target tokens: all but first (shifted by 1)
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    # Weights: 1 for assistant response tokens, 0 for context
    # The assistant response starts at position n_context in the full sequence
    # In the shifted scheme, target position i corresponds to predicting token i+1
    # So we want weight=1 for target positions [n_context-1, n_total-2]
    weights = np.zeros(len(target_tokens), dtype=np.float32)
    resp_start = max(0, n_context - 1)
    weights[resp_start:] = 1.0

    if weights.sum() == 0:
        return None

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "weights": tinker.TensorData.from_numpy(weights),
            "target_tokens": tinker.TensorData.from_numpy(
                np.array(target_tokens, dtype=np.int64)
            ),
        },
    )


def compute_mean_nll(
    loss_fn_outputs: list[dict],
    datums: list[tinker.Datum],
) -> float:
    """Compute weighted mean negative log-likelihood from forward_backward outputs."""
    total_nll = 0.0
    total_weight = 0.0
    for output, datum in zip(loss_fn_outputs, datums):
        logprobs = output["logprobs"].to_numpy()
        weights = datum.loss_fn_inputs["weights"].to_numpy()
        total_nll += -float(np.dot(logprobs, weights))
        total_weight += float(weights.sum())
    if total_weight == 0:
        return 0.0
    return total_nll / total_weight


def sample_from_model(
    sampling_client: tinker.SamplingClient,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
) -> list[str]:
    """Generate responses from the current model for qualitative eval."""
    results = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        model_input = tinker.ModelInput.from_ints(tokens)

        response = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=30,
                temperature=0.7,
            ),
        ).result()

        output_tokens = response.sequences[0].tokens
        decoded = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        results.append(decoded)
    return results


def main():
    parser = argparse.ArgumentParser(description="SFT training for emoji-llm on Tinker")
    parser.add_argument("--train-data", type=str, default="data/sft_train.jsonl")
    parser.add_argument("--test-data", type=str, default="data/sft_test.jsonl")
    parser.add_argument("--log-dir", type=str, default="runs/sft")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-schedule", type=str, default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20, help="Save checkpoint every N steps")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate on test set every N steps")
    parser.add_argument("--sample-every", type=int, default=50, help="Sample from model every N steps")
    parser.add_argument("--eval-batches", type=int, default=5, help="Number of test batches for eval")
    parser.add_argument("--resume", type=str, default=None, help="Tinker state path to resume from")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(log_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Load data ---
    logger.info("Loading training data from %s", args.train_data)
    train_conversations = load_conversations(args.train_data)
    logger.info("Loading test data from %s", args.test_data)
    test_conversations = load_conversations(args.test_data)

    # --- Setup Tinker ---
    service_client = tinker.ServiceClient()

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        training_client = service_client.create_training_client_from_state_with_optimizer(args.resume)
    else:
        logger.info("Creating new LoRA training client for %s (rank=%d)", MODEL_NAME, args.lora_rank)
        training_client = service_client.create_lora_training_client(
            base_model=MODEL_NAME,
            rank=args.lora_rank,
        )

    tokenizer = training_client.get_tokenizer()

    # --- Tokenize datasets ---
    logger.info("Tokenizing training data...")
    train_datums = []
    skipped = 0
    for conv in tqdm(train_conversations, desc="Tokenizing train"):
        datum = conversation_to_datum(conv, tokenizer, max_length=args.max_length)
        if datum is not None:
            train_datums.append(datum)
        else:
            skipped += 1
    logger.info("Train: %d datums (%d skipped)", len(train_datums), skipped)

    logger.info("Tokenizing test data...")
    test_datums = []
    for conv in test_conversations:
        datum = conversation_to_datum(conv, tokenizer, max_length=args.max_length)
        if datum is not None:
            test_datums.append(datum)
    logger.info("Test: %d datums", len(test_datums))

    # --- Training loop ---
    n_train = len(train_datums)
    n_batches = n_train // args.batch_size
    total_steps = n_batches * args.num_epochs

    logger.info("Training config:")
    logger.info("  Batches per epoch: %d", n_batches)
    logger.info("  Total steps: %d", total_steps)
    logger.info("  Batch size: %d (sequences)", args.batch_size)
    logger.info("  Learning rate: %s (%s schedule)", args.learning_rate, args.lr_schedule)
    logger.info("  LoRA rank: %d", args.lora_rank)

    metrics_log = open(log_dir / "metrics.jsonl", "a")
    step = 0

    for epoch in range(args.num_epochs):
        # Shuffle training data each epoch
        rng = np.random.default_rng(seed=epoch)
        indices = rng.permutation(n_train)

        for batch_idx in range(n_batches):
            start_time = time.time()
            batch_start = batch_idx * args.batch_size
            batch_end = batch_start + args.batch_size
            batch_indices = indices[batch_start:batch_end]
            batch = [train_datums[i] for i in batch_indices]

            # Learning rate schedule
            progress = step / total_steps
            if args.lr_schedule == "linear":
                lr_mult = max(0.0, 1.0 - progress)
            elif args.lr_schedule == "cosine":
                lr_mult = 0.5 * (1.0 + np.cos(np.pi * progress))
            else:
                lr_mult = 1.0
            current_lr = args.learning_rate * lr_mult

            # Forward-backward + optimizer step (pipelined)
            fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_future = training_client.optim_step(
                tinker.AdamParams(
                    learning_rate=current_lr,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )
            )

            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_future.result()

            # Compute training loss
            train_nll = compute_mean_nll(fwd_bwd_result.loss_fn_outputs, batch)
            elapsed = time.time() - start_time
            n_tokens = sum(d.model_input.length for d in batch)

            metrics = {
                "step": step,
                "epoch": epoch,
                "train_nll": train_nll,
                "learning_rate": current_lr,
                "progress": progress,
                "n_tokens": n_tokens,
                "time_s": round(elapsed, 2),
                "tokens_per_s": round(n_tokens / elapsed, 1),
            }

            if optim_result.metrics:
                metrics["optim_metrics"] = optim_result.metrics

            # --- Eval on test set ---
            if args.eval_every > 0 and step % args.eval_every == 0:
                eval_datums = test_datums[: args.eval_batches * args.batch_size]
                if eval_datums:
                    eval_future = training_client.forward(eval_datums, loss_fn="cross_entropy")
                    eval_result = eval_future.result()
                    test_nll = compute_mean_nll(eval_result.loss_fn_outputs, eval_datums)
                    metrics["test_nll"] = test_nll
                    logger.info(
                        "Step %d/%d | train_nll=%.4f | test_nll=%.4f | lr=%.2e | %.1fs",
                        step, total_steps, train_nll, test_nll, current_lr, elapsed,
                    )
                else:
                    logger.info(
                        "Step %d/%d | train_nll=%.4f | lr=%.2e | %.1fs",
                        step, total_steps, train_nll, current_lr, elapsed,
                    )
            else:
                logger.info(
                    "Step %d/%d | train_nll=%.4f | lr=%.2e | %.1fs",
                    step, total_steps, train_nll, current_lr, elapsed,
                )

            # --- Save checkpoint ---
            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                ckpt_name = f"step-{step:06d}"
                logger.info("Saving checkpoint: %s", ckpt_name)
                save_future = training_client.save_state(ckpt_name, ttl_seconds=604800)
                save_result = save_future.result()
                metrics["checkpoint"] = ckpt_name
                logger.info("Checkpoint saved: %s", save_result)

            # --- Sample from model ---
            if args.sample_every > 0 and step % args.sample_every == 0:
                logger.info("Sampling from model...")
                try:
                    sampling_client = training_client.save_weights_and_get_sampling_client(
                        name=f"sample-{step:06d}"
                    )
                    responses = sample_from_model(sampling_client, tokenizer, EVAL_PROMPTS)
                    sample_log = []
                    for prompt, response in zip(EVAL_PROMPTS, responses):
                        logger.info("  Q: %s", prompt)
                        logger.info("  A: %s", response)
                        sample_log.append({"prompt": prompt, "response": response})
                    metrics["samples"] = sample_log
                except Exception as e:
                    logger.warning("Sampling failed: %s", e)

            metrics_log.write(json.dumps(metrics) + "\n")
            metrics_log.flush()
            step += 1

    # --- Final checkpoint ---
    logger.info("Saving final checkpoint...")
    final_save = training_client.save_state("final", ttl_seconds=None)
    final_result = final_save.result()
    logger.info("Final checkpoint saved: %s", final_result)

    # Final sampling
    logger.info("Final model samples:")
    try:
        sampling_client = training_client.save_weights_and_get_sampling_client(name="final-sampler")
        responses = sample_from_model(sampling_client, tokenizer, EVAL_PROMPTS)
        for prompt, response in zip(EVAL_PROMPTS, responses):
            logger.info("  Q: %s", prompt)
            logger.info("  A: %s", response)
    except Exception as e:
        logger.warning("Final sampling failed: %s", e)

    metrics_log.close()
    logger.info("Training complete! Logs saved to %s", log_dir)


if __name__ == "__main__":
    main()
