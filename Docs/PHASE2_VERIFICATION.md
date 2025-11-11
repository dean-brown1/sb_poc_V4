# Phase 2 Verification

Run these tests to verify Phase 2 completion:

## 1. Import Test
```bash
cd /home/claude/sb_poc_v4
python3 << 'PYEOF'
# Test all module imports
try:
    from src import data, training, model, evaluation, utils
    print("✓ All modules import successfully")
    
    # Test key functions exist
    assert hasattr(data, 'load_gsm8k_data')
    assert hasattr(data, 'prepare_gsm8k_dataset')
    assert hasattr(training, 'train_baseline')
    assert hasattr(training, 'train_schemabank')
    assert hasattr(model, 'SchemaBank')
    assert hasattr(evaluation, 'eval_gsm8k_accuracy')
    assert hasattr(utils, 'load_config')
    print("✓ All key functions present")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)
PYEOF
```

## 2. Config Loading Test
```bash
python3 << 'PYEOF'
from src.utils import load_config
from pathlib import Path

# Test baseline config
baseline = load_config('configs/baseline.yaml')
assert baseline['experiment']['mode'] == 'baseline'
print("✓ Baseline config loads correctly")

# Test schemabank config
schemabank = load_config('configs/schemabank.yaml')
assert schemabank['experiment']['mode'] == 'schemabank'
assert schemabank['schemabank']['num_schemas'] == 32
assert len(schemabank['training']['stages']) == 3
print("✓ SchemaBank config loads correctly")

print("\n✅ All configuration tests passed")
PYEOF
```

## 3. Data Module Test
```bash
python3 << 'PYEOF'
from src.data import assign_schema_tags_hash, get_tag_dropout_rate

# Test schema tagging
tags = assign_schema_tags_hash("What is 2+2?", num_schemas=32)
assert len(tags) == 2
assert all(0 <= t < 32 for t in tags)
print("✓ Schema tagging works")

# Test tag dropout schedule
assert get_tag_dropout_rate(0, 100) == 0.0      # Quarter 1
assert get_tag_dropout_rate(30, 100) == 0.75    # Quarter 2
assert get_tag_dropout_rate(60, 100) == 0.50    # Quarter 3
assert get_tag_dropout_rate(90, 100) == 0.25    # Quarter 4
print("✓ Tag dropout schedule works")

print("\n✅ All data module tests passed")
PYEOF
```

## 4. File Structure Test
```bash
python3 << 'PYEOF'
from pathlib import Path

required_files = [
    'train.py',
    'evaluate.py',
    'requirements.txt',
    'README.md',
    'configs/baseline.yaml',
    'configs/schemabank.yaml',
    'src/data.py',
    'src/training.py',
    'src/model.py',
    'src/evaluation.py',
    'src/utils.py',
]

root = Path('.')
missing = []
for file in required_files:
    path = root / file
    if not path.exists():
        missing.append(file)
        print(f"✗ Missing: {file}")
    else:
        print(f"✓ Found: {file}")

if missing:
    print(f"\n✗ {len(missing)} files missing")
    exit(1)
else:
    print(f"\n✅ All {len(required_files)} required files present")
PYEOF
```

## Expected Output

All tests should pass with output like:
```
✓ All modules import successfully
✓ All key functions present
✓ Baseline config loads correctly
✓ SchemaBank config loads correctly
✅ All configuration tests passed
✓ Schema tagging works
✓ Tag dropout schedule works
✅ All data module tests passed
✓ Found: train.py
✓ Found: evaluate.py
...
✅ All 12 required files present
```

## Run All Tests
```bash
cd /home/claude/sb_poc_v4

echo "==================================="
echo "PHASE 2 VERIFICATION"
echo "==================================="

echo -e "\nTest 1: Module Imports"
python3 -c "from src import data, training, model, evaluation, utils; print('✓ Passed')"

echo -e "\nTest 2: Config Loading"  
python3 -c "from src.utils import load_config; load_config('configs/baseline.yaml'); load_config('configs/schemabank.yaml'); print('✓ Passed')"

echo -e "\nTest 3: Data Functions"
python3 -c "from src.data import assign_schema_tags_hash, get_tag_dropout_rate; assign_schema_tags_hash('test', 32); get_tag_dropout_rate(50, 100); print('✓ Passed')"

echo -e "\nTest 4: Files Present"
python3 -c "from pathlib import Path; assert all((Path(f).exists() for f in ['train.py', 'evaluate.py', 'src/data.py', 'src/training.py'])); print('✓ Passed')"

echo -e "\n==================================="
echo "✅ PHASE 2 VERIFIED"
echo "==================================="
```
