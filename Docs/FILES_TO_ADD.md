# Files to Add to Your Repository

All files are in `/mnt/user-data/outputs/sb_poc_v4/`

## Documentation Files (Add to root)

1. **README.md** (7.4K) - Primary documentation ⭐ START HERE
2. **QUICK_START.md** (5.3K) - Quick setup guide
3. **DOCUMENTS.md** (5.3K) - Documentation index
4. **SUMMARY.md** (2.3K) - One-page overview
5. **PHASE1_COMPLETE.md** (4.3K) - Bug fixes documentation
6. **PHASE2_COMPLETE.md** (6.4K) - Reorganization notes
7. **PHASE2_VERIFICATION.md** (4.2K) - Testing guide

## Configuration Files (Add to configs/)

8. **configs/baseline.yaml** - Baseline configuration (5.4% expected)
9. **configs/schemabank.yaml** - SchemaBank configuration (11.8% expected)

## Source Code (Add to src/)

10. **src/data.py** (350 lines) - Data preparation
11. **src/training.py** (600 lines) - Training logic
12. **src/model.py** (220 lines) - SchemaBank architecture
13. **src/evaluation.py** (280 lines) - Evaluation functions
14. **src/utils.py** (220 lines) - Utilities

## Entry Points (Add to root)

15. **train.py** (150 lines) - Main training script
16. **evaluate.py** (120 lines) - Standalone evaluation

## Dependencies (Add to root)

17. **requirements.txt** - Locked dependency versions

## Test Files (Add to root, optional)

18. **test_eval_fix.py** - Evaluation bug fix verification

---

## Quick Add Commands

If you're in your repo root:

```bash
# Copy all documentation
cp /mnt/user-data/outputs/sb_poc_v4/*.md .

# Copy configs
mkdir -p configs
cp /mnt/user-data/outputs/sb_poc_v4/configs/*.yaml configs/

# Copy source code
mkdir -p src
cp /mnt/user-data/outputs/sb_poc_v4/src/*.py src/

# Copy entry points
cp /mnt/user-data/outputs/sb_poc_v4/train.py .
cp /mnt/user-data/outputs/sb_poc_v4/evaluate.py .
chmod +x train.py evaluate.py

# Copy dependencies
cp /mnt/user-data/outputs/sb_poc_v4/requirements.txt .

# Copy tests (optional)
cp /mnt/user-data/outputs/sb_poc_v4/test_eval_fix.py .
```

Or copy the entire directory:

```bash
cp -r /mnt/user-data/outputs/sb_poc_v4 ~/your-repo-location/
```

---

## File Sizes Summary

- **Total source code:** ~1,940 lines
- **Total documentation:** ~35KB (7 markdown files)
- **Configuration:** 2 YAML files
- **Dependencies:** 1 requirements.txt

---

## What to Read First

1. **README.md** - Complete overview
2. **QUICK_START.md** - Get running in 5 minutes
3. **DOCUMENTS.md** - Find what you need

---

## Verification After Adding

Run these tests to verify everything works:

```bash
# Test 1: Check files present
ls README.md train.py evaluate.py src/data.py src/training.py

# Test 2: Check imports (requires dependencies installed)
python -c "from src import data, training, model, evaluation, utils"

# Test 3: Check configs load
python -c "from src.utils import load_config; load_config('configs/baseline.yaml')"
```

All tests should pass with no errors.

---

## Ready to Run

After adding all files:

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline training
python train.py --config configs/baseline.yaml

# Run SchemaBank training  
python train.py --config configs/schemabank.yaml
```

See **QUICK_START.md** for detailed instructions.
