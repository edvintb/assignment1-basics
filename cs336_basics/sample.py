import argparse
import sys
import time

import torch as th
from jaxtyping import Int
from cs336_basics.tokenizer import Tokenizer

from cs336_basics.io_functions import load_model_from_checkpoint
from cs336_basics.model import TransformerLM


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a transformer language model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--max-tokens-per-reponse", type=int, default=100, help="Maximum number of new tokens to generate"
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation (if not provided, will use interactive mode)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (higher = more random, lower = more deterministic)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Only sample from the top probability tokens.")
    args = parser.parse_args()
    return args


def generate(
    model: TransformerLM,
    token_ids: Int[th.Tensor, "batch sequence"],
    token_positions: Int[th.Tensor, "batch sequence"] | None = None,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    for _ in range(max_new_tokens):
        logits = model(token_ids, token_positions)
        last_token_logits = logits[:, -1, :]

        if temperature == 0:
            # Create a one-hot distribution with 1 for the highest prob and 0 elsewhere
            max_indices = th.argmax(last_token_logits, dim=-1, keepdim=True)
            probs = th.zeros_like(last_token_logits)
            probs.scatter_(dim=-1, index=max_indices, value=1.0)
        else:
            # Sample from the distribution instead of always picking the max
            probs = th.softmax(last_token_logits / temperature, dim=-1)

        if top_p < 1.0:
            # top-p sampling
            sorted_probs, sorted_indices = th.sort(probs, descending=True)
            cumulative_probs = th.cumsum(sorted_probs, dim=-1)
            mask = cumulative_probs <= top_p
            # Zero out probabilities outside top-p
            probs_filtered = th.zeros_like(probs)
            probs_filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_probs * mask)
            probs = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)

        next_token_ids = th.multinomial(probs, num_samples=1)

        token_ids = th.cat([token_ids, next_token_ids], dim=-1)
        if token_positions is not None:
            token_positions = th.cat([token_positions, token_positions[:, -1:] + 1], dim=-1)
    return token_ids


def main(args) -> None:
    # instantiate model from checkpoint
    print(f"Loading model from checkpoint {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint)

    # load tokenizer from vocab and merges files
    print("Loading tokenizer from vocab and merges files...")

    # parametrize the tokenizer...
    tokenizer = Tokenizer.from_file(
        vocab_file="data/ts-vocab.json",
        merges_file="data/ts-merges.json",
        special_tokens=["<|endoftext|>"],
    )

    # move model to device
    device = th.device("cuda" if th.cuda.is_available() else "mps")
    model = model.to(device)
    model.eval()

    # prompt the user for text input
    if args.prompt is not None:
        # Non-interactive mode: use provided prompt
        prompt = args.prompt
        token_ids = tokenizer.encode(prompt)
        token_ids = th.tensor(token_ids, dtype=th.int32).unsqueeze(0).to(device)
        token_positions = th.arange(token_ids.shape[1], device=device).unsqueeze(0)
        start_time = time.time()
        generated_token_ids = generate(model, token_ids, token_positions, args.max_tokens_per_reponse, args.temperature)
        generated_text = tokenizer.decode(generated_token_ids[0].tolist())
        end_time = time.time()
        print(f"Generated text in {end_time - start_time:.2f}s:\n {generated_text}")
    else:
        # Interactive mode: prompt repeatedly
        while True:
            prompt = input("Prompt: ")
            # Create fresh token_ids and token_positions for each new prompt (clears context)
            token_ids = tokenizer.encode(prompt)
            token_ids = th.tensor(token_ids, dtype=th.int32).unsqueeze(0).to(device)
            token_positions = th.arange(token_ids.shape[1], device=device).unsqueeze(0)
            original_length = token_ids.shape[1]

            start_time = time.time()
            generated_token_ids = generate(model, token_ids, token_positions, args.max_tokens_per_reponse, args.temperature)

            # Decode only the newly generated tokens (excluding the original prompt)
            new_tokens_only = generated_token_ids[0, original_length:].tolist()
            generated_text = tokenizer.decode(new_tokens_only)
            end_time = time.time()
            print(f"Generated text in {end_time - start_time:.2f}s:\n{generated_text}")


if __name__ == "__main__":
    main(get_args())
