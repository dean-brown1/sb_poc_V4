# SchemaBank V4 - Quick Start Guide

## Installation

```bash
# Clone the repository
cd ~/your-repo-location

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from src import data, training, model, evaluation, utils; print('✓ All modules loaded')"
```

## Run Training

### Option 1: Baseline (5.4% expected)

```bash
python train.py --config configs/baseline.yaml
```

**Expected output:**
```
Results: results/baseline_seed42/
├── config.yaml
├── results.json           # Accuracy: ~5.4%
├── training_log.jsonl
└── checkpoint/
```

**Training time:** ~30-45 minutes on RTX 5000 Ada

### Option 2: SchemaBank (11.8% expected)

```bash
python train.py --config configs/schemabank.yaml
```

**Expected output:**
```
Results: results/schemabank_seed42/
├── config.yaml
├── results.json           # Accuracy: ~11.8%
├── training_log.jsonl
└── checkpoint/
    ├── schemabank_adapters.pt
    └── schemabank_config.json
```

**Training time:** ~45-60 minutes on RTX 5000 Ada

## Evaluate Saved Model

```bash
python evaluate.py --checkpoint results/schemabank_seed42/checkpoint
```

## Customize Configuration

Edit `configs/schemabank.yaml` to modify:

```yaml
schemabank:
  num_schemas: 32        # Number of schemas (try: 16, 32, 64)
  rank: 16               # Low-rank dimension (try: 8, 16, 24)
  topk: 2                # Schemas per token (try: 1, 2, 3)
  
training:
  total_steps: 1000      # Total training steps
  learning_rate: 1.0e-4  # Learning rate
  
  stages:
    stage1_router_pretrain:
      steps: 250           # Router training steps
    stage2_schema_train:
      steps: 500           # Schema training steps
    stage3_joint_finetune:
      steps: 250           # Joint training steps
```

## Smoke Test (Quick Validation)

Test the pipeline without full training (runs in ~2-3 minutes):

```bash
# Create minimal test config
cat > configs/smoke_test.yaml << 'YAML'
experiment:
  name: "smoke_test"
  seed: 42
  mode: "schemabank"
  output_dir: "./results/smoke_test"

model:
  base_model: "Qwen/Qwen2-0.5B"
  torch_dtype: "bfloat16"
  device_map: "auto"

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  bias: "none"
  task_type: "CAUSAL_LM"

schemabank:
  num_schemas: 32
  rank: 16
  topk: 2
  attr_dim: 32
  layers: "last_2"
  regularization:
    ortho_weight: 0.01

training:
  total_steps: 10
  batch_size: 1
  gradient_accumulation_steps: 4
  seq_len: 2048
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.05
  max_grad_norm: 1.0
  stages:
    stage1_router_pretrain:
      steps: 5
      frozen: ["base_model", "lora_adapters", "schema_transforms"]
      trainable: ["router_weights"]
      tag_curriculum:
        dropout_schedule: [0.0, 0.75, 0.50, 0.25]
    stage2_schema_train:
      steps: 3
      frozen: ["base_model", "router_weights"]
      trainable: ["schema_transforms", "lora_adapters"]
      use_tags: false
    stage3_joint_finetune:
      steps: 2
      frozen: ["base_model"]
      trainable: ["router_weights", "schema_transforms", "lora_adapters"]
      use_tags: false

dataset:
  name: "gsm8k"
  split_train: "train"
  split_test: "test"
  num_train_samples: null
  tagging_method: "hash"

evaluation:
  num_test_samples: 10
  max_new_tokens: 200
  do_sample: false

logging:
  log_every_n_steps: 1
  save_final_only: true
YAML

# Run smoke test
python train.py --config configs/smoke_test.yaml
```

## Monitor Training

Watch training progress:
```bash
tail -f results/schemabank_seed42/training_log.jsonl
```

Parse logs for quick stats:
```python
import json

with open('results/schemabank_seed42/training_log.jsonl') as f:
    logs = [json.loads(line) for line in f]
    
print(f"Total steps: {len(logs)}")
print(f"Final loss: {logs[-1]['loss']:.4f}")
print(f"Stages: {set(log['stage'] for log in logs)}")
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config (try 1)
- Use gradient accumulation instead
- Try smaller base model

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Verify CUDA is being used: check training output
- Consider increasing `batch_size` if memory allows

### Import Errors
```bash
# Verify you're in the right directory
pwd  # Should show .../sb_poc_v4

# Verify virtual environment is activated
which python  # Should show venv path

# Reinstall if needed
pip install -r requirements.txt --force-reinstall
```

### Wrong Results
- Check seed: `experiment.seed` in config
- Verify data is loading: check first few training steps
- Compare config to proven configs in `configs/`

## Next Steps

1. **Run baseline first** to verify setup
2. **Run SchemaBank** to reproduce results  
3. **Experiment** with different hyperparameters
4. **Analyze results** in generated JSON files

## Getting Help

Check these documents:
- `README.md` - Full documentation
- `PHASE1_COMPLETE.md` - Critical bug fixes
- `PHASE2_COMPLETE.md` - Code structure
- `PHASE2_VERIFICATION.md` - Testing guide
