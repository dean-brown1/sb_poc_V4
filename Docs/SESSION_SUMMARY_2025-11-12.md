# Session Summary: V4 Debugging

## Achieved
- 6.0% LoRA baseline accuracy
- Fixed prompt format, training epochs, evaluation
- All fixes committed to main branch

## Outstanding Issue
- SchemaBank trained successfully (loss 15.27→2.47)
- But generates garbage at inference (repeated zeros)
- Suspect training/inference architecture mismatch

## Next Session
- Option 1: Debug SchemaBank inference
- Option 2: Compare V3 vs V4 architectures
- Option 3: Accept 6% baseline, move forward

## Checkpoints
- results/schemabank_2epochs_run004/ - Best run so far
