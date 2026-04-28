"""
Benchmark HF generate() vs vLLM for emoji rollout generation.

Simulates one training step:
- 8 rollouts (group_size=8)
- Up to 5 turns per rollout
- ~20 max tokens per turn
- Emoji logit masking applied

Reports:
- Time per generation call
- Time per full step (8 rollouts × avg 3 turns)
- Estimated total time for 500 steps
- Recommendation: HF or vLLM

HF and vLLM run in separate Modal functions so each starts with a clean GPU —
PyTorch's CUDA allocator doesn't release memory to the OS between del/empty_cache,
which causes OOM when vLLM tries to pre-allocate 90% of the card.
"""

import time

import modal

from src.rl.custom.generate import MODEL_NAME, build_emoji_mask, format_prompt

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "transformers>=4.45.0",
        "torch>=2.4.0",
        "peft>=0.14.0",
        "sentence-transformers",
        "regex",
    )
    .add_local_python_source("src")
)

_volume = modal.Volume.from_name("emoji-model-weights", create_if_missing=True)
_MODEL_CACHE_DIR = "/model-cache"

app = modal.App("emoji-benchmark")

N_TRIALS = 5
GROUP_SIZE = 8
MAX_TURNS = 5
MAX_TOKENS = 20
PHRASE = "job interview"
GPU_COST_PER_HOUR = 3.67  # A100 40GB on Modal (approx)

_secrets = [
    modal.Secret.from_name("anthropic-api-key"),
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("hf-secret"),
    modal.Secret.from_name("wandb-secret"),
]

_fn_kwargs = dict(
    gpu="A100",
    timeout=1800,
    image=_image,
    volumes={_MODEL_CACHE_DIR: _volume},
    secrets=_secrets,
)


@app.function(**_fn_kwargs)
def benchmark_hf() -> dict:
    """Benchmark HF generate on a clean GPU."""
    import os

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = _MODEL_CACHE_DIR

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=_MODEL_CACHE_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    emoji_mask = build_emoji_mask(tokenizer)

    print(f"Loading policy model (LoRA) for {MODEL_NAME}...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=_MODEL_CACHE_DIR,
    )
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
        bias="none",
    )
    policy_model = get_peft_model(base, lora_cfg)

    from src.rl.custom.train import generate_rollouts

    print("Warming up HF generate...")
    _ = generate_rollouts(policy_model, tokenizer, emoji_mask, PHRASE, group_size=2)

    print(f"Running HF benchmark ({N_TRIALS} trials)...")
    times = []
    for trial in range(N_TRIALS):
        start = time.time()

        _ = generate_rollouts(
            policy_model, tokenizer, emoji_mask, PHRASE,
            group_size=GROUP_SIZE, temperature=1.0,
        )
        for turn in range(2, MAX_TURNS + 1):
            n_continuing = max(1, int(GROUP_SIZE * (0.6 ** (turn - 1))))
            for _ in range(n_continuing):
                _ = generate_rollouts(
                    policy_model, tokenizer, emoji_mask, PHRASE,
                    group_size=1, temperature=1.0,
                )

        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Trial {trial + 1}: {elapsed:.2f}s")

    per_step = sum(times) / len(times)
    return {"per_step": per_step}


@app.function(**_fn_kwargs)
def benchmark_vllm() -> dict:
    """Benchmark vLLM on a clean GPU, including Option A reload timing."""
    import os
    import tempfile

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams  # ty: ignore[unresolved-import]

    os.environ["HF_HOME"] = _MODEL_CACHE_DIR

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=_MODEL_CACHE_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    emoji_mask = build_emoji_mask(tokenizer)
    emoji_token_ids = emoji_mask.nonzero(as_tuple=False).squeeze(1).tolist()
    params = SamplingParams(
        temperature=1.0,
        max_tokens=MAX_TOKENS,
        allowed_token_ids=emoji_token_ids,
    )

    print(f"Loading vLLM for {MODEL_NAME}...")
    llm = LLM(model=MODEL_NAME, download_dir=_MODEL_CACHE_DIR, dtype="bfloat16")

    print("Warming up vLLM...")
    _ = llm.generate([format_prompt(PHRASE, tokenizer)], params)

    print(f"Running vLLM benchmark ({N_TRIALS} trials)...")
    times = []
    for trial in range(N_TRIALS):
        start = time.time()

        prompts = [format_prompt(PHRASE, tokenizer)] * GROUP_SIZE
        _ = llm.generate(prompts, params)
        for turn in range(2, MAX_TURNS + 1):
            n_continuing = max(1, int(GROUP_SIZE * (0.6 ** (turn - 1))))
            _ = llm.generate([format_prompt(PHRASE, tokenizer)] * n_continuing, params)

        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Trial {trial + 1}: {elapsed:.2f}s")

    vllm_per_step = sum(times) / len(times)

    # Option A: time vLLM reload (reinit LLM with updated LoRA weights each step)
    print("\n--- Benchmarking vLLM Option A: reload time ---")
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM", lora_dropout=0.0, bias="none",
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cpu", cache_dir=_MODEL_CACHE_DIR,
    )
    policy = get_peft_model(base, lora_cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        policy.save_pretrained(tmpdir)
        del policy, base

        reload_times = []
        for trial in range(3):
            start = time.time()
            del llm
            torch.cuda.empty_cache()
            llm = LLM(model=MODEL_NAME, download_dir=_MODEL_CACHE_DIR, dtype="bfloat16")
            elapsed = time.time() - start
            reload_times.append(elapsed)
            print(f"  Reload trial {trial + 1}: {elapsed:.2f}s")

    reload_mean = sum(reload_times) / len(reload_times)
    return {"per_step": vllm_per_step, "reload_mean": reload_mean}


@app.local_entrypoint()
def main():
    """Run HF and vLLM benchmarks in parallel, then print the summary table."""
    print("Launching HF and vLLM benchmarks in parallel...")
    hf_future = benchmark_hf.spawn()
    vllm_future = benchmark_vllm.spawn()

    hf = hf_future.get()
    vllm = vllm_future.get()

    hf_per_step = hf["per_step"]
    vllm_per_step = vllm["per_step"]
    reload_mean = vllm["reload_mean"]

    hf_total_500 = hf_per_step * 500 / 3600
    vllm_total_500 = vllm_per_step * 500 / 3600
    hf_cost = hf_total_500 * GPU_COST_PER_HOUR
    vllm_cost = vllm_total_500 * GPU_COST_PER_HOUR
    speedup = hf_per_step / vllm_per_step if vllm_per_step > 0 else float("inf")

    print("\nGeneration Benchmark Results")
    print("============================")
    print(f"{'':25s} {'HF generate':>14s}    {'vLLM':>10s}")
    print(f"{'Per-step (full):':25s} {hf_per_step:>12.2f}s    {vllm_per_step:>8.2f}s")
    print(f"{'500 steps:':25s} {hf_total_500:>12.2f}h    {vllm_total_500:>8.2f}h")
    print(f"{'Est. GPU cost:':25s} ${hf_cost:>12.2f}    ${vllm_cost:>8.2f}")
    print()
    print(f"Speedup: {speedup:.1f}x")
    print(f"vLLM reload time per step (Option A): {reload_mean:.2f}s")
    print(f"  → vLLM net per step w/ reload: {vllm_per_step + reload_mean:.2f}s")
    print(f"  → 500 steps w/ reload: {(vllm_per_step + reload_mean) * 500 / 3600:.2f}h")
    print()

    if hf_total_500 < 8:
        rec = "HF generate"
        reason = f"HF total ({hf_total_500:.1f}h) < 8h — simpler, no weight sync complexity"
    elif vllm_total_500 < 4:
        rec = "vLLM"
        reason = f"vLLM total ({vllm_total_500:.1f}h) < 4h — speedup justifies complexity"
    else:
        rec = "Reduce group_size or max_turns"
        reason = "Both options are slow — consider group_size=4 or max_turns=3"

    print(f"Recommendation: {rec}")
    print(f"Reason: {reason}")
