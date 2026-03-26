"""Interactive chat with the SFT emoji model on Tinker."""

import argparse

import tinker
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You communicate exclusively using emoji. No text, numbers, or punctuation ever. "
    "Use 2-8 emoji per response that capture the core meaning, emotion, and key concepts "
    "of the user's message."
)


def main():
    parser = argparse.ArgumentParser(description="Chat with the emoji model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tinker://51acb88b-35a0-5f3b-a102-a4b5d5643714:train:0/weights/final",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_training_client_from_state(args.checkpoint)
    sampling_client = training_client.save_weights_and_get_sampling_client(name="chat")
    tokenizer = sampling_client.get_tokenizer()
    print("Ready!\n")

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye! 👋")
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        model_input = tinker.ModelInput.from_ints(tokens)

        response = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        decoded = tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True).strip()
        print(f"Bot: {decoded}\n")


if __name__ == "__main__":
    main()
