# SchemaBank - Quick Start Guide

Get up and running in 5 minutes.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended)
- ~2GB disk space for model downloads

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/schemabank
cd schemabank

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
python -c "from src import data, training, model, evaluation, utils; print('✓ All modules loaded')"
```

Expected output:
```
PyTorch: 2.8.0
CUDA: True
Transformers: 4.56.1
✓ All modules loaded
```

## Run Your First Training

### Option 1: Baseline (Quick Test - ~12 min)

```bash
python train.py --config configs/baseline_10epochs.yaml
```

**What to expect:**
- Training: 18,680 steps (~12 minutes)
- Final accuracy: ~3% (2-5% range)
- High variance: This is normal for baseline
- Output: `results/baseline_10epochs/`

### Option 2: SchemaBank (Recommended - ~12 min)

```bash
python train.py --config configs/schemabank_10epochs.yaml
```

**What to expect:**
- Training: 18,680 steps in 3 stages (~12 minutes)
- Final accuracy: ~10% (10-11% range)
- Low variance: More stable than baseline
- Output: `results/schemabank_10epochs/`

**Training Progress:**
```
Stage 1: Router + Tags: 100%|████████| 4670/4670 [01:29<00:00]
Stage 2: Schemas:       100%|████████| 9340/9340 [07:03<00:00]
Stage 3: Joint:         100%|████████| 4670/4670 [03:33<00:00]
```

## Evaluate Results

```bash
# Evaluate trained checkpoint
python evaluate.py --checkpoint results/schemabank_10epochs/checkpoint --num_samples 500
```

**Evaluation outputs:**
- GSM8K Accuracy
- Perplexity (512 and 4096 tokens)
- Sample predictions
- Saves to `checkpoint/evaluation.json`

## Check Results

```bash
# View summary
cat results/schemabank_10epochs/results.json

# View training progress
tail -20 results/schemabank_10epochs/training_log.jsonl

# Check evaluation
cat results/schemabank_10epochs/checkpoint/evaluation.json
```

## Understanding Output Structure

After training, you'll have:

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

For research / statistical validation:

```bash
#!/bin/bash
# Run 3 seeds for SchemaBank

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

## Customize Experiments

### Change Number of Epochs

Edit `configs/schemabank_10epochs.yaml`:

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
  num_schemas: 32    # Try: 16, 32, 64
  rank: 16           # Try: 8, 16, 24
  topk: 2            # Try: 1, 2, 3
```

### Change Learning Rate

```yaml
training:
  learning_rate: 1.0e-4  # Default
  # Try: 5e-5 (lower), 2e-4 (higher)
```

## Monitor Training

### Watch Live Progress

```bash
tail -f results/schemabank_10epochs/training_log.jsonl
```

### Quick Analysis

```python
import json

# Load training log
with open('results/schemabank_10epochs/training_log.jsonl') as f:
    logs = [json.loads(line) for line in f]

# Training summary
print(f"Total steps: {len(logs)}")
print(f"Initial loss: {logs[0]['loss']:.2f}")
print(f"Final loss: {logs[-1]['loss']:.2f}")

# By stage
for stage in ['stage1_router_pretrain', 'stage2_schema_train', 'stage3_joint_finetune']:
    stage_logs = [l for l in logs if l.get('stage') == stage]
    if stage_logs:
        print(f"\n{stage}:")
        print(f"  Steps: {len(stage_logs)}")
        print(f"  Final loss: {stage_logs[-1]['loss']:.2f}")
```

## Troubleshooting

### Out of Memory

**Symptoms:** CUDA out of memory error

**Solutions:**
```yaml
training:
  batch_size: 1              # Already minimum
  gradient_accumulation_steps: 4  # Keep this
  seq_len: 2048              # Reduce from 4096
```

Or use gradient checkpointing (slower but less memory):
```python
# Add to model preparation in train.py
model.gradient_checkpointing_enable()
```

### Slow Training

**Check GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

Should see:
- GPU Util: 95-100%
- Memory: 20-25GB used
- Power: Near max (300W for RTX 5000 Ada)

**If low utilization:**
- Increase `batch_size` if memory allows
- Check CUDA is being used: `torch.cuda.is_available()`

### Wrong Results

**Accuracy much lower than expected?**

1. **Check seed:** Make sure config has `seed: 42` for reproducibility
2. **Check epochs:** 10 epochs = 18,680 steps (verify in training log)
3. **Check data:** GSM8K should show 7,473 training examples loaded
4. **Compare configs:** Diff against proven configs in repo

```bash
# Compare your config to reference
diff configs/schemabank_10epochs.yaml configs/schemabank_10epochs.yaml.reference
```

### Import Errors

```bash
# Verify you're in project directory
pwd  # Should show: .../schemabank

# Verify venv is activated
which python  # Should show: .../schemabank/venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Expected Performance

### Baseline (LoRA only)
- **Accuracy**: 2-5% (mean ~3%)
- **Variance**: High (±1.3%)
- **Training**: Faster convergence but lower plateau
- **Use case**: Comparison baseline

### SchemaBank
- **Accuracy**: 10-11% (mean ~10.35%)
- **Variance**: Low (±0.55%)
- **Training**: Structured 3-stage curriculum
- **Use case**: Target method

### Training Time (RTX 5000 Ada)
- **10 epochs**: ~12 minutes per run
- **Per seed**: Baseline and SchemaBank take similar time
- **4 seeds**: ~48 minutes total

### Disk Space
- **Model downloads**: ~1GB (Qwen2-0.5B)
- **Per checkpoint**: ~20MB (LoRA + SchemaBank adapters)
- **Logs**: ~2MB per run

## Next Steps

1. ✅ **Run baseline** - Verify setup works (~3% expected)
2. ✅ **Run SchemaBank** - Compare to baseline (~10% expected)
3. 📊 **Analyze results** - Check training logs and evaluation
4. 🔬 **Experiment** - Try different hyperparameters
5. 📈 **Scale up** - Test multiple seeds for statistical validation

## Getting Help

- **Documentation**: See `README.md` for comprehensive details
- **Issues**: Check GitHub issues for known problems
- **Configs**: All proven configurations in `configs/` directory
- **Code**: Fully documented source code in `src/` directory

## Validation Checklist

Before considering your setup validated:

- [ ] Baseline achieves 2-5% accuracy
- [ ] SchemaBank achieves 9-12% accuracy
- [ ] SchemaBank shows lower variance than baseline
- [ ] Training completes in ~12 minutes
- [ ] Evaluation runs without errors
- [ ] Checkpoint files are created correctly

If all boxes checked: ✅ **Your setup is validated!**

---

**Ready to run?** Start with:
```bash
python train.py --config configs/schemabank_10epochs.yaml
```