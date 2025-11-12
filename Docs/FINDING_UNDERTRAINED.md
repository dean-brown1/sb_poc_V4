# Critical Finding: Severe Undertraining

**Date:** 2025-11-12  
**Issue:** 0% accuracy despite correct prompt format and decreasing loss

## Root Cause

Model was trained for only **0.535 epochs** - never saw the full dataset even once!

### Calculations
```
Batch size: 1
Gradient accumulation: 4
Effective batch size: 4
Total steps: 1000
Examples seen: 4,000
Dataset size: 7,473
Epochs: 0.535
```

**Required for 1 epoch:** 1,868 steps  
**V4 used:** 1,000 steps (53.5% of one epoch)

## Impact

- Loss decreased: 15.57 → 4.52 ✓
- Perplexity improved: 351 → 238 ✓
- But accuracy: 0% ✗

Model learned the format but not the math reasoning.

## Solution

Train for minimum **2 epochs** (3,736 steps):
- Stage 1: 934 steps (0.5 epoch)
- Stage 2: 1,868 steps (1.0 epoch)
- Stage 3: 934 steps (0.5 epoch)

Expected result: 5-12% accuracy (baseline performance)

## Config

See `configs/schemabank_2epochs.yaml`
