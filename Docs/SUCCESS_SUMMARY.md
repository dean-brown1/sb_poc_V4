# V4 Training Success - 6.0% Baseline Achieved

**Date:** 2025-11-12  
**Result:** Successfully trained SchemaBank to 6.0% accuracy on GSM8K

## Problems Found and Fixed

### 1. Prompt Format Issue
**Problem:** Qwen2-0.5B generates Python code for `"Question: ... Answer:"` format  
**Solution:** Changed to `"{question}\nThe answer is:"` format  
**Files:** `src/data.py`, `src/evaluation.py`

### 2. Severe Undertraining
**Problem:** Only 0.535 epochs (1000 steps with batch=4)  
**Solution:** Trained for 2 epochs (3736 steps)  
**Impact:** Model never saw full dataset before

### 3. Number Extraction Bug
**Problem:** Evaluation used `numbers[-1]` (last number in generation)  
**Solution:** Changed to `numbers[0]` (first number = the answer)  
**File:** `src/evaluation.py`

## Final Results

**Checkpoint:** `results/schemabank_2epochs_run004/`

- **Accuracy:** 6.0% (6/100 correct)
- **Loss:** 15.27 → 2.47
- **Perplexity:** 351 → 96
- **Training:** 3736 steps (2 epochs)
- **Match V3 baseline:** ✓ (V3 was 5.4%)

## Next Steps

1. Apply these fixes to full SchemaBank training
2. Expected SchemaBank accuracy: 10-15% (vs 6% baseline)
3. Compare to V3's SchemaBank: 11.8%

## Lessons Learned

1. **Always test base model behavior** - Qwen2 has quirks
2. **Calculate epochs properly** - Steps ≠ epochs
3. **Test extraction logic** - First vs last number matters
4. **Verify training actually ran** - Check step counts
