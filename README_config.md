# Training Configuration Guide

## Usage

Instead of passing many command line arguments, use a YAML configuration file:

```bash
# Basic usage
python cs336_basics/train.py --config config/train_small.yaml

# Override specific values
python cs336_basics/train.py --config config/train_small.yaml --override training.batch_size=64 training.device=cuda

# Multiple overrides
python cs336_basics/train.py --config config/train_large.yaml --override training.device=cpu model.d_model=768 training.num_steps=50000
```

## Configuration Files

### Available Configs

- `config/debug.yaml` - Tiny model for quick testing and debugging
- `config/train_small.yaml` - Small model for experimentation  
- `config/train_large.yaml` - Large model for serious training

### Configuration Structure

```yaml
# Data Configuration
data:
  dataset_path: "path/to/your/dataset.npy"
  vocab_size: 10000
  context_length: 512
  batch_size: 32

# Model Architecture
model:
  d_model: 512
  d_ff: 2048
  num_heads: 8
  num_layers: 6
  rope_theta: 10000.0

# Training Configuration
training:
  device: "cpu"  # or "cuda"
  num_steps: 10000
  
  # Learning Rate Schedule
  max_learning_rate: 3e-4
  min_learning_rate: 3e-5
  warmup_iters: 1000
  cosine_cycle_iters: 8000
  
  # Regularization
  max_l2_norm: 1.0
  weight_decay: 0.01
  
  # Checkpointing & Logging
  checkpoint_path: "checkpoints/my_model"
  save_every: 1000
  log_every: 100
```

## Benefits

- **Clean**: No more 20+ command line arguments
- **Organized**: Related parameters grouped together
- **Reusable**: Save and share configurations
- **Flexible**: Override specific values without editing files
- **Version Control**: Track configuration changes in git
