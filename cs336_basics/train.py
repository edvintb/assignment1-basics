import argparse
import time
import numpy as np
import numpy.typing as npt
import torch

from cs336_basics.functions import cross_entropy, gradient_clipping
from cs336_basics.io_functions import get_batch, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.optimizers import AdamW, cosine_lr_schedule


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                       help="Override config values (e.g., --override training.batch_size=32)")

    args = parser.parse_args()

    # Load YAML config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides
    for override in args.override:
        key, value = override.split('=', 1)
        keys = key.split('.')

        # Navigate to the right nested dict
        current = config
        for k in keys[:-1]:
            current = current[k]

        # Convert value to appropriate type
        try:
            current[keys[-1]] = int(value)
        except ValueError:
            try:
                current[keys[-1]] = float(value)
            except ValueError:
                current[keys[-1]] = value

    # Convert dict to namespace for easy access
    def dict_to_namespace(d):
        namespace = argparse.Namespace()
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(namespace, key, dict_to_namespace(value))
            else:
                setattr(namespace, key, value)
        return namespace

    return dict_to_namespace(config)


def main() -> None:
    config = get_args()

    # load the data (mmap to support large dataset)
    # Load NPZ file with memory mapping for large datasets
    npz_data = np.load(config.data.dataset_path, mmap_mode='r')
    data: npt.NDArray[np.int32] = npz_data['tokens']

    # instantiate model
    model = TransformerLM(
        vocab_size=config.data.vocab_size,
        context_length=config.data.context_length,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        rope_theta=config.model.rope_theta,
    )

    # Move model to device
    device = torch.device(config.training.device)
    model = model.to(device)

    # instantiate optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.max_learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # train model
    for step in range(config.training.num_steps):
        step_start_time = time.time()

        # Get the learning rate for this step
        current_lr = cosine_lr_schedule(
            it=step,
            max_learning_rate=config.training.max_learning_rate,
            min_learning_rate=config.training.min_learning_rate,
            warmup_iters=config.training.warmup_iters,
            cosine_cycle_iters=config.training.cosine_cycle_iters,
        )

        # Apply the learning rate to all parameter groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Sample a batch of data
        x_batch, y_batch = get_batch(
            x=data,
            batch_size=config.data.batch_size,
            context_length=config.data.context_length,
            device=device,
        )

        # zero the gradients in the optimizer
        optimizer.zero_grad()

        # compute the loss
        logits = model(x_batch)
        loss = cross_entropy(logits, y_batch)

        # backprop to compute parameter gradients wrt loss
        loss.backward()

        # clip the gradients if needed
        gradient_clipping(model.parameters(), config.training.max_l2_norm)

        # optimizer step to update the parameters
        optimizer.step()

        step_end_time = time.time()

        print(f"Step: {step:03d}, Loss: {loss:.4f}, Time: {step_end_time - step_start_time:.2f}s")

        if step % config.training.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, step, config.training.checkpoint_path)


if __name__ == "__main__":
    main()
