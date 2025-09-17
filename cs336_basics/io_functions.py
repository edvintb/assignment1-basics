import os
from typing import IO, BinaryIO
import random
import numpy as np
import numpy.typing as npt
import torch as th


def get_batch(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
):
    # sample some random indicies
    indices = random.sample(range(len(x) - context_length), batch_size)

    # use these indicies to pick sequences
    x_batch = np.array([x[i:i+context_length] for i in indices])
    y_batch = np.array([x[i+1:i+context_length+1] for i in indices])

    # convert to torch tensors
    x_batch = th.tensor(x_batch, dtype=th.int, device=device)
    y_batch = th.tensor(y_batch, dtype=th.int, device=device)

    return x_batch, y_batch

def save_checkpoint(
    model: th.nn.Module,
    optimizer: th.optim.Optimizer,
    iteration: int,
    model_config: dict,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    th.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "model_config": model_config,
    }, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: th.nn.Module,
    optimizer: th.optim.Optimizer,
):
    checkpoint = th.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

def load_model_from_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
) -> tuple[th.nn.Module, dict, int]:
    """
    Load a model directly from checkpoint without needing a pre-existing model instance.

    Args:
        src: Path to checkpoint file

    Returns:
        tuple of (model, model_config, iteration)
    """
    checkpoint = th.load(src)
    model_config = checkpoint["model_config"]
    iteration = checkpoint["iteration"]

    # Create model from saved config
    from cs336_basics.model import TransformerLM
    model = TransformerLM(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
    )

    # Load the saved weights
    model.load_state_dict(checkpoint["model"])

    return model, model_config, iteration