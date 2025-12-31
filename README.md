# SchemaBank: Sparse Routing as Training Curriculum

**Parameter-efficient fine-tuning through structured routing-based training**

📄 **[Read the full paper](Docs/SchemaBank_Paper.pdf)**

## Overview

SchemaBank demonstrates that sparse routing mechanisms, when used as a **training curriculum** rather than an inference tool, dramatically improve LoRA adapter learning. Training with routing then removing it at inference achieves **3x better accuracy** than baseline LoRA on GSM8K mathematical reasoning.

### Key Results (Epoch 6 Peak, 4 Seeds)

| Method | Mean Accuracy | Std Dev | Improvement |
|--------|---------------|---------|-------------|
| **Baseline** (LoRA only) | 3.75% | ±1.50% | - |
| **SchemaBank** (trained with routing) | **11.8%** | ±1.30% | **3.1x** |

**Critical Finding:** SchemaBank provides both higher performance AND lower variance across seeds.

## Architecture

SchemaBank attaches sparse routing modules to the last 2 transformer layers during training:

```
Input → Transformer Layers → [Last-2 layers with SchemaBank] → Output
                                    ↓
                              Router (H→S logits)
                                    ↓
                              Top-k selection (k=2)
                                    ↓
                          Low-rank transforms (U, V)
                                    ↓
                              Gated mixing → Output
```

**Key Properties:**
- **S=32 schemas**: Specialized transformation pathways
- **Rank r=16**: Low-rank bottleneck per schema
- **Top-k=2**: Sparse activation (2 schemas per token)
- **Removed at inference**: Only LoRA adapters used for predictions

## Three-Stage Training Curriculum

### Stage 1: Router Pre-training (25% of steps)
- **Train**: Router weights only
- **Frozen**: Base model + LoRA + Schema transforms
- **Tag Curriculum**: Progressive dropout (0% → 25% → 50% → 75%)
- **Goal**: Learn task-appropriate routing patterns

### Stage 2: Schema Training (50% of steps)
- **Train**: Schema transforms (U, V) + LoRA adapters
- **Frozen**: Base model + Router
- **No tags**: Pure outcome-based learning
- **Goal**: Schemas learn specialized transformations

### Stage 3: Joint Fine-tuning (25% of steps)
- **Train**: Router + Schemas + LoRA
- **Frozen**: Base model only
- **No tags**: End-to-end optimization
- **Goal**: Refine coordination between components

**At inference:** Router and schemas are removed. Only LoRA adapters remain.

## Installation

```bash
# Clone repository
git clone https://github.com/dean-brown1/sb_poc_V4
cd sb_poc_V4

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- PyTorch 2.1+
- Transformers 4.40+
- CUDA-capable GPU (16GB+ VRAM recommended)

## Quick Start

### Train Baseline (LoRA only)

```bash
python train.py --config configs/baseline_10epochs.yaml
```

**Expected:** ~3% accuracy, ~12 minutes training

### Train SchemaBank

```bash
python train.py --config configs/schemabank_10epochs.yaml
```

**Expected:** ~10% accuracy, ~12 minutes training

### Evaluate Checkpoint

```bash
python evaluate.py --checkpoint results/schemabank_10epochs/checkpoint --num_samples 500
```

## Configuration

### Baseline Configuration

```yaml
experiment:
  name: "baseline_10epochs"
  seed: 42
  mode: "baseline"

model:
  base_model: "Qwen/Qwen2-0.5B"
  torch_dtype: "bfloat16"

lora:
  r: 8
  lora_alpha: 16
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training:
  total_steps: 18680  # 10 epochs
  batch_size: 1
  learning_rate: 1.0e-4
```

### SchemaBank Configuration

```yaml
schemabank:
  num_schemas: 32
  rank: 16
  topk: 2
  layers: "last_2"

training:
  total_steps: 18680
  
  stages:
    stage1_router_pretrain:
      steps: 4670  # 25%
      tag_curriculum:
        dropout_schedule: [0.0, 0.25, 0.5, 0.75]
      
    stage2_schema_train:
      steps: 9340  # 50%
      
    stage3_joint_finetune:
      steps: 4670  # 25%
```

## Project Structure

```
schemabank/
├── configs/
│   ├── baseline_10epochs.yaml
│   └── schemabank_10epochs.yaml
├── src/
│   ├── data.py           # Dataset loading, schema tagging
│   ├── training.py       # Three-stage training logic
│   ├── model.py          # SchemaBank architecture
│   ├── evaluation.py     # GSM8K accuracy, perplexity
│   └── utils.py          # Config, logging, telemetry
├── train.py              # Main training entry point
├── evaluate.py           # Standalone evaluation
└── requirements.txt
```

## Detailed Results

### Accuracy by Seed (10 Epochs)

**SchemaBank:**
- Seed 42: 10.0%
- Seed 123: 10.0%
- Seed 456: 10.2%
- Seed 789: 11.2%
- **Mean: 10.35% ± 0.55%**

**Baseline:**
- Seed 42: 2.4%
- Seed 123: 5.0%
- Seed 456: 2.4%
- Seed 789: 2.4%
- **Mean: 3.05% ± 1.30%**

### Key Findings

1. **3.4x Performance Improvement**: SchemaBank achieves 10.35% vs baseline's 3.05%
2. **2.4x Better Stability**: SchemaBank has ±0.55% variance vs baseline's ±1.30%
3. **Curriculum Matters**: Training with routing provides structured learning signal
4. **Inference Simplicity**: Removing routing at inference avoids overhead while keeping benefits

### Training Dynamics

**Final Training Losses (10 Epochs):**
- SchemaBank: ~1.52 (well converged)
- Baseline: ~2.4-2.5 (less converged)

**Evaluation Timing:**
- SchemaBank: Consistent 2.1-2.8s per question
- Baseline: Erratic 1-3s per question (less stable outputs)

## Why SchemaBank Works

### Hypothesis: Training Curriculum Effect

The routing mechanism during training provides:

1. **Structured Exploration**: Router forces model to consider multiple specialized pathways
2. **Ensemble Learning**: Schemas create implicit ensemble during training
3. **Better LoRA Initialization**: LoRA adapters learn from structured signal, not chaos
4. **Compositional Reasoning**: Different schemas handle different reasoning patterns

### Evidence

- **Tag alignment during training**: Router learns task structure (captured in routing telemetry)
- **Schema specialization**: Different schemas activate for different problem types
- **Transfer to LoRA**: Benefits persist even after removing routing
- **Stability improvement**: Reduced variance suggests better optimization landscape

## Reproducibility

### Verified Configurations
- **Model**: Qwen/Qwen2-0.5B (base model, not instruction-tuned)
- **Training**: 10 epochs = 18,680 steps (batch_size=1, grad_accum=4)
- **Seeds**: 42, 123, 456, 789 (all tested)
- **Hardware**: CUDA-capable GPU (16GB+ VRAM recommended)

### Random Seed Control
```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

### Results Package
Each training run produces:
```
results/schemabank_10epochs_seed42/
├── config.yaml              # Exact configuration used
├── training_log.jsonl       # Per-step metrics
├── results.json             # Final evaluation results
└── checkpoint/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── schemabank_adapters.pt      # SchemaBank weights
    └── schemabank_config.json
```

## Citation

```bibtex
@software{schemabank2025,
  title={SchemaBank: Sparse Routing as Training Curriculum for Parameter-Efficient Fine-Tuning},
  author={Dean Brown},
  year={2025},
  url={https://github.com/dean-brown1/sb_poc_V4}
}
```

## Future Work

- [ ] Scale to 1B-2B models
- [ ] Test on additional reasoning tasks (MATH, ARC)
- [ ] Ablation studies (stage ratios, curriculum variations)
- [ ] Mechanistic analysis (what do schemas learn?)
- [ ] Compare to other curriculum methods

## Acknowledgments

Built on:
- **Qwen2** (Alibaba): Base model
- **LoRA** (Hu et al.): Parameter-efficient fine-tuning
- **GSM8K** (Cobbe et al.): Evaluation dataset

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Version**: 4.0 (10-Epoch Validated)  
**Date**: November 2025  
**Status**: Research code - validated results with 4 seeds