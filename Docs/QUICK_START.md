# SchemaBank - Quick Start Guide

Get up and running in 5 minutes.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended)
- ~2GB disk space for model downloads

## Installation

```bash
# Clone the repository
git clone https://github.com/dean-brown1/sb_poc_V4
cd sb_poc_V4

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
# Test PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Test Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test module imports
python -c "from src import data, training, model, evaluation, utils; print('All modules loaded successfully')"
```

## Run Training

### Option 1: Baseline (Quick Test)

```bash
python train.py --config configs/baseline_10epochs.yaml
```

**Expected results:**
- Training: 18,680 steps (~10-15 minutes depending on hardware)
- Final accuracy: ~3% (2-5% range)
- Output: `results/baseline_10epochs/`

### Option 2: SchemaBank (Recommended)

```bash
python train.py --config configs/schemabank_10epochs.yaml
```

**Expected results:**
- Training: 18,680 steps in 3 stages
- Final accuracy: ~10% (10-11% range)
- Output: `results/schemabank_10epochs/`

**Training stages:**
```
Stage 1: Router + Tags (25% of steps)
Stage 2: Schemas (50% of steps)
Stage 3: Joint fine-tuning (25% of steps)
```

## Evaluate Results

```bash
# Evaluate trained checkpoint
python evaluate.py --checkpoint results/schemabank_10epochs/checkpoint --num_samples 500
```

**Evaluation outputs:**
- GSM8K Accuracy
- Perplexity measurements
- Sample predictions
- Results saved to `checkpoint/evaluation.json`

## Check Results

```bash
# View summary
cat results/schemabank_10epochs/results.json

# View training progress
tail -20 results/schemabank_10epochs/training_log.jsonl

# Check evaluation
cat results/schemabank_10epochs/checkpoint/evaluation.json
```

## Output Structure

After training:

```
results/schemabank_10epochs/
├── config.yaml              # Configuration used
├── results.json             # Training summary
├── training_log.jsonl       # Per-step metrics
└── checkpoint/
    ├── adapter_config.json
    ├── adapter_model.safetensors    # LoRA weights
    ├── schemabank_adapters.pt       # SchemaBank weights
    ├── schemabank_config.json
    └── evaluation.json              # After running evaluate.py
```

## Run Multiple Seeds

For statistical validation:

```bash
#!/bin/bash
for seed in 42 123 456; do
    # Create seed-specific config
    sed "s/seed: 42/seed: $seed/" configs/schemabank_10epochs.yaml > configs/temp_seed_$seed.yaml
    sed -i "s/output_dir: .*/output_dir: \"\.\/results\/schemabank_10epochs_seed$seed\"/" configs/temp_seed_$seed.yaml

    # Train
    python train.py --config configs/temp_seed_$seed.yaml

    # Evaluate
    python evaluate.py --checkpoint results/schemabank_10epochs_seed$seed/checkpoint --num_samples 500

    # Cleanup
    rm configs/temp_seed_$seed.yaml
done
```

## Configuration Options

### Adjust Training Duration

Edit the config file:

```yaml
training:
  total_steps: 18680  # 10 epochs
  # For 5 epochs: 9340
  # For 20 epochs: 37360

  stages:
    stage1_router_pretrain:
      steps: 4670  # 25% of total
    stage2_schema_train:
      steps: 9340  # 50% of total
    stage3_joint_finetune:
      steps: 4670  # 25% of total
```

**Note:** Keep stage ratios at 1:2:1 (25%:50%:25%)

### Adjust SchemaBank Architecture

```yaml
schemabank:
  num_schemas: 32    # Options: 16, 32, 64
  rank: 16           # Options: 8, 16, 24
  topk: 2            # Options: 1, 2, 3
```

### Adjust Learning Rate

```yaml
training:
  learning_rate: 1.0e-4  # Default
  # Lower: 5e-5
  # Higher: 2e-4
```

## Troubleshooting

### Out of Memory

Reduce sequence length in config:

```yaml
training:
  seq_len: 2048  # Reduce from 4096
```

Or enable gradient checkpointing in `train.py`:

```python
model.gradient_checkpointing_enable()
```

### Import Errors

```bash
# Verify virtual environment is activated
which python  # Should show path to venv

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Low Accuracy

1. Verify seed is set in config (`seed: 42`)
2. Verify training completed all steps (check logs)
3. Confirm GSM8K dataset loaded correctly (7,473 training examples)

## Expected Performance

| Method | Mean Accuracy | Std Dev |
|--------|---------------|---------|
| Baseline (LoRA only) | 2-5% | ±1.3% |
| SchemaBank | 10-11% | ±0.55% |

## Next Steps

1. Run baseline to verify setup
2. Run SchemaBank to compare results
3. Analyze training logs
4. Experiment with hyperparameters
5. Run multiple seeds for statistical validation

## Validation Checklist

- [ ] Baseline achieves 2-5% accuracy
- [ ] SchemaBank achieves 9-12% accuracy
- [ ] SchemaBank shows lower variance than baseline
- [ ] Training completes without errors
- [ ] Checkpoint files are created correctly

---

**Ready to start?**
```bash
python train.py --config configs/schemabank_10epochs.yaml
```
