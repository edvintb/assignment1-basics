import os
from typing import IO, BinaryIO
import random
import numpy as np
import numpy.typing as npt
import torch as th

from cs336_basics.model import TransformerLM


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
    config: dict,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    th.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "config": config,
    }, out)


def save_checkpoint_from_state_dicts(
    model_state_dict: dict,
    optimizer_state_dict: dict,
    iteration: int,
    config: dict,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """Save checkpoint from pre-extracted state dictionaries (async-safe)."""
    th.save({
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
        "iteration": iteration,
        "config": config,
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
) -> TransformerLM:
    checkpoint = th.load(src)
    config = checkpoint["config"]
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    )
    model.load_state_dict(checkpoint["model"])
    return model