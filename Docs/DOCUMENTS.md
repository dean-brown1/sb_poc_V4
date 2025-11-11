# SchemaBank V4 - Documentation Index

## Core Documents

### 1. README.md (Primary Documentation)
**Purpose:** Complete project documentation  
**Contents:**
- Overview and key results
- Critical fixes from V3
- Project structure
- Installation instructions
- Usage examples
- Configuration guide
- Architecture details
- Reproduction checklist
- Citations

**Read this first** for comprehensive understanding.

---

### 2. QUICK_START.md (Get Running Fast)
**Purpose:** Step-by-step guide to run training  
**Contents:**
- Installation
- Verification tests
- Training commands (baseline & SchemaBank)
- Smoke test for quick validation
- Monitoring tips
- Troubleshooting

**Use this** to get up and running in 5 minutes.

---

### 3. PHASE1_COMPLETE.md (Critical Bug Fixes)
**Purpose:** Documents the evaluation bug fix  
**Contents:**
- What was broken in V3
- The fix (total += 1)
- Configuration system creation
- Code modularization start
- Dependency locking

**Read this** to understand what changed from V3.

---

### 4. PHASE2_COMPLETE.md (Code Reorganization)
**Purpose:** Documents the modularization effort  
**Contents:**
- Data module extraction (350 lines)
- Training module extraction (600 lines)
- Entry point creation (train.py, evaluate.py)
- Code quality improvements
- Verification tests

**Read this** to understand the V4 architecture.

---

### 5. PHASE2_VERIFICATION.md (Testing Guide)
**Purpose:** How to verify everything works  
**Contents:**
- Import tests
- Config loading tests
- Data function tests
- File structure tests
- Complete test suite

**Use this** to validate your setup.

---

### 6. SUMMARY.md (Quick Reference)
**Purpose:** One-page overview  
**Contents:**
- Status summary
- Key fixes
- File structure
- Critical decisions
- Next steps

**Use this** for a quick refresher.

---

## Configuration Files

### configs/baseline.yaml
**Purpose:** Baseline (LoRA-only) configuration  
**Expected Result:** 5.4% accuracy  
**Training Time:** ~30-45 minutes  
**Key Settings:**
- Single-stage training
- No SchemaBank
- 1000 total steps

---

### configs/schemabank.yaml
**Purpose:** SchemaBank three-stage configuration  
**Expected Result:** 11.8% accuracy  
**Training Time:** ~45-60 minutes  
**Key Settings:**
- Three-stage training
- S=32 schemas, r=16 rank, topk=2
- Tag curriculum in stage 1
- 1000 total steps (250+500+250)

---

## Code Documentation

### src/data.py
**Lines:** 350  
**Purpose:** Data preparation and loading  
**Key Functions:**
- `load_gsm8k_data()` - Load dataset
- `prepare_gsm8k_dataset()` - Add schema tags
- `assign_schema_tags_hash()` - Deterministic tagging
- `get_tag_dropout_rate()` - Curriculum schedule
- `create_dataloader()` - DataLoader creation

---

### src/training.py
**Lines:** 600  
**Purpose:** Training logic (baseline & SchemaBank)  
**Key Functions:**
- `prepare_model()` - Setup model, LoRA, SchemaBank
- `train_baseline()` - Single-stage baseline training
- `train_schemabank()` - Three-stage SchemaBank training
- `train_stage1_router()` - Router pre-training
- `train_stage2_schemas()` - Schema training
- `train_stage3_joint()` - Joint fine-tuning

---

### src/model.py
**Lines:** 220  
**Purpose:** SchemaBank architecture  
**Key Classes:**
- `SchemaBank` - Main module (router + low-rank transforms)
- Helper functions for model preparation

---

### src/evaluation.py
**Lines:** 280  
**Purpose:** Evaluation functions (FIXED)  
**Key Functions:**
- `eval_gsm8k_accuracy()` - GSM8K accuracy (bug fixed)
- `eval_perplexity()` - Perplexity measurement
- `eval_long_context_stability()` - PPL at 512 vs 4096
- `analyze_schema_usage()` - Schema statistics

---

### src/utils.py
**Lines:** 220  
**Purpose:** Utilities (config, logging, saving)  
**Key Functions:**
- `load_config()` - YAML config loading
- `set_seed()` - Reproducibility
- `TelemetryLogger` - Per-step logging
- `save_experiment_results()` - Results packaging

---

## Entry Points

### train.py
**Lines:** 150  
**Purpose:** Main training script  
**Usage:** `python train.py --config configs/schemabank.yaml`  
**Output:** Complete results package in output_dir

---

### evaluate.py
**Lines:** 120  
**Purpose:** Standalone evaluation  
**Usage:** `python evaluate.py --checkpoint results/schemabank_seed42/checkpoint`  
**Output:** Detailed evaluation JSON

---

## Reading Order

**For Quick Start:**
1. QUICK_START.md
2. Run training
3. Done!

**For Understanding the System:**
1. README.md (overview)
2. PHASE1_COMPLETE.md (what changed)
3. PHASE2_COMPLETE.md (how it's organized)
4. Code documentation (as needed)

**For Development:**
1. README.md (overview)
2. PHASE2_COMPLETE.md (architecture)
3. src/training.py (training logic)
4. src/data.py (data pipeline)
5. Modify configs, run experiments

**For Troubleshooting:**
1. QUICK_START.md (troubleshooting section)
2. PHASE2_VERIFICATION.md (run tests)
3. Check logs in results/

---

## Document Status

✅ All documents complete and verified  
✅ All code documented with docstrings  
✅ All configs tested and working  
✅ Ready for Phase 4 (validation with real training)

---

## Additional Resources

**In Repository:**
- requirements.txt - Exact dependency versions
- test_eval_fix.py - Verification test for bug fix

**External:**
- Anthropic API docs (if using Claude in artifacts)
- Hugging Face docs (for models and datasets)
- PyTorch docs (for architecture details)
