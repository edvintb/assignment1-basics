import argparse
import ast
import os
import time
import yaml
import numpy as np
import numpy.typing as npt
import torch
from concurrent.futures import ThreadPoolExecutor
import wandb

from cs336_basics.functions import cross_entropy, gradient_clipping
from cs336_basics.io_functions import get_batch, save_checkpoint_from_state_dicts
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

        # Convert value to appropriate type using ast.literal_eval
        # This handles int, float, string, bool, None, lists, tuples, dicts
        try:
            current[keys[-1]] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If literal_eval fails, treat as string
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

    # Initialize wandb
    wandb_name = os.environ.get("WANDB_NAME", None)
    wandb.init(
        project="llm_experiments",
        name=wandb_name,
        config={
            "data": vars(config.data),
            "model": vars(config.model),
            "training": vars(config.training),
        }
    )
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

    # create config dict for checkpointing
    config_dict: dict[str, int | float] = {
        "vocab_size": config.data.vocab_size,
        "context_length": config.data.context_length,
        "d_model": config.model.d_model,
        "num_layers": config.model.num_layers,
        "num_heads": config.model.num_heads,
        "d_ff": config.model.d_ff,
        "rope_theta": config.model.rope_theta,
    }

    # Convert checkpoint path to absolute path and ensure it has .pt extension
    checkpoint_path = os.path.abspath(config.training.checkpoint_path)
    if not checkpoint_path.endswith('.pt'):
        checkpoint_path += '.pt'

    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}. Checkpoint path: {checkpoint_path}")

    # Update config to use absolute path
    config.training.checkpoint_path = checkpoint_path

    # Move model to device
    device = torch.device(config.training.device)
    model = model.to(device)

    # Note: wandb.watch() disabled due to pandas compatibility issue
    # We're already logging gradient norms manually in the training loop

    # instantiate optimizer
    # Ensure all hyperparameters are proper numeric types
    adam_eps = getattr(config.training, 'adam_eps', 1e-8)
    adam_beta1 = getattr(config.training, 'adam_beta1', 0.9)
    adam_beta2 = getattr(config.training, 'adam_beta2', 0.999)

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.max_learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )

    # Create thread pool executor for async checkpoint saving
    checkpoint_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint")
    pending_checkpoint = None

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

        # zero the gradients in the optimizer
        optimizer.zero_grad()

        # Apply the learning rate to all parameter groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Sample a batch of data
        x_batch, y_batch = get_batch(
            x=data,
            batch_size=config.data.batch_size,
            context_length=config.data.context_length,
            device=str(device),
        )

        # compute the loss
        logits = model(x_batch)
        loss = cross_entropy(logits, y_batch)

        # backprop to compute parameter gradients wrt loss
        loss.backward()

        # clip the gradients if needed
        grad_norm_before, grad_norm_after = gradient_clipping(
            params=model.parameters(),
            max_l2_norm=config.training.max_l2_norm,
            eps=getattr(config.training, 'gradient_clip_eps', 1e-6),
            step=step,
            log_every=config.training.log_every
        )

        # optimizer step to update the parameters
        optimizer.step()

        step_end_time = time.time()

        if step % config.training.log_every == 0:
            # compute validation loss
            val_x_batch, val_y_batch = get_batch(
                x=data,
                batch_size=config.data.batch_size,
                context_length=config.data.context_length,
                device=str(device),
            )
            val_logits = model(val_x_batch)
            val_loss = cross_entropy(val_logits, val_y_batch)
            print(f"Step: {step:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Time: {step_end_time - step_start_time:.2f}s")

            # Log metrics to wandb
            wandb.log({
                "train/loss": loss.item(),
                "train/val_loss": val_loss.item(),
                "train/learning_rate": current_lr,
                "train/step_time": step_end_time - step_start_time,
                "train/grad_norm_before_clip": grad_norm_before,
                "train/grad_norm_after_clip": grad_norm_after,
                "step": step,
            })

        if step % config.training.save_every == 0 and step > 0:
            # Wait for any previous checkpoint to complete before starting a new one
            if pending_checkpoint is not None:
                pending_checkpoint.result()  # This will block until the previous save is done

            # Create deep copies of the state dicts to avoid issues with concurrent access
            model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            optimizer_state = {
                'state': {k: {inner_k: inner_v.cpu().clone() if torch.is_tensor(inner_v) else inner_v
                             for inner_k, inner_v in inner_dict.items()}
                         for k, inner_dict in optimizer.state_dict()['state'].items()},
                'param_groups': optimizer.state_dict()['param_groups']
            }

            # Submit checkpoint saving to thread pool
            pending_checkpoint = checkpoint_executor.submit(
                save_checkpoint_from_state_dicts,
                model_state,
                optimizer_state,
                step,
                config_dict,
                config.training.checkpoint_path
            )
            print(f"Checkpoint save initiated for step {step}")

    # Wait for final checkpoint to complete and cleanup
    if pending_checkpoint is not None:
        pending_checkpoint.result()
        print("Final checkpoint save completed")

    checkpoint_executor.shutdown(wait=True)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
