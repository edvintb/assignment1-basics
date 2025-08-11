import argparse
import sys
import time

import torch as th
from jaxtyping import Int
from loguru import logger

from cs336_basics.io_functions import load_checkpoint_for_inference
from cs336_basics.model import TransformerLM

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a transformer language model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--max-tokens-per-reponse", type=int, default=100,
                       help="Maximum number of new tokens to generate")
    args = parser.parse_args()
    return args



def generate(
    model: TransformerLM,
    token_ids: Int[th.Tensor, 'batch sequence'],
    token_positions: Int[th.Tensor, 'batch sequence'] | None = None,
    max_new_tokens: int = 100,
):
    for _ in range(max_new_tokens):
        logits = model(token_ids, token_positions)
        next_token_ids = th.argmax(logits, dim=-1, keepdim=True)
        token_ids = th.cat([token_ids, next_token_ids], dim=-1)
        if token_positions is not None:
            token_positions = th.cat([token_positions, token_positions[:, -1:] + 1], dim=-1)
    return token_ids

def main(args) -> None:
    # instantiate model from checkpoint
    logger.info(f"Loading model from checkpoint {args.checkpoint}")
    model, config, iteration = load_model_from_checkpoint(args.checkpoint)

    # log the config, checkpoint and iteration
    logger.info(f"Iteration: {iteration}")
    logger.info(f"Config: {config}")

    # load tokenizer from vocab and merges files
    logger.info(f"Loading tokenizer from vocab and merges files...")

    # parametrize the tokenizer...
    tokenizer = Tokenizer.from_file(
        vocab_file="data/ts-vocab.json",
        merges_file="data/ts-merges.json",
        special_tokens=["<|endoftext|>"],
    )

    # move model to device
    device = th.device("cuda" if th.cuda.is_available() else "mps")
    model = model.to(device)

    # prompt the user for text input
    while True:
        prompt = input("Prompt: ")
        token_ids = tokenizer.encode(prompt)
        token_ids = th.tensor(token_ids, dtype=th.int32).unsqueeze(0).to(device)
        token_positions = th.arange(token_ids.shape[1], device=device).unsqueeze(0)
        start_time = time.time()
        generated_token_ids = generate(model, token_ids, token_positions, args.max_tokens_per_reponse)
        generated_text = tokenizer.decode(generated_token_ids[0].tolist())
        end_time = time.time()
        logger.info(f"Generated text in {end_time - start_time:.2f}s:\n {generated_text}")

    

if __name__ == "__main__":
    main(get_args())
