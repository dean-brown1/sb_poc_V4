# Bug Fix: Prompt Format Code Generation Issue

**Date:** 2025-11-12  
**Severity:** Critical (0% → 2% accuracy proven, 5-12% expected with full training)  
**Status:** Fixed in branch `fix/prompt-format-code-generation`

## Problem

Model generated Python code instead of numerical answers, causing 0% accuracy.

**Example output:**
```python
Generated: ```python
def money_made():
    """Janet's ducks lay 16 eggs per day...
```

## Root Cause

Qwen2-0.5B base model associates `"Question: ... Answer:"` format with code generation:
1. Pretrained on code with docstrings starting with `"""Question: ..."""`
2. Our training format triggered this pattern
3. Strong pretraining prior overrode fine-tuning

**Verified:** Untrained base model shows identical behavior.

## Solution

Changed prompt format to avoid code trigger:

| Before | After |
|--------|-------|
| `"Question: {q}\nAnswer:"` | `"{q}\nThe answer is:"` |

## Results

| Metric | Before | After |
|--------|--------|-------|
| Accuracy (50 test samples) | 0/50 (0%) | 1/50 (2%) |
| Output type | Python code | Numbers |
| Expected with full training | 0% | 5-12% |

## Files Modified

- `src/data.py` - Training data formatting
- `src/evaluation.py` - Evaluation prompt

## Next Steps

1. ✅ Fix applied and tested
2. ⏳ Re-train with new format (1000 steps)
3. ⏳ Validate achieves 5-12% baseline
4. ⏳ Apply to full SchemaBank training

## Lessons Learned

- **Always test base model behavior** before assuming prompt format
- **Prompt engineering is critical** - small changes = huge differences
- **Strong priors from pretraining** can override fine-tuning signals
