#!/usr/bin/env python3
"""
Convert SchemaBank training logs from JSONL to CSV

Usage:
    python convert_logs_to_csv.py results/schemabank_seed42/training_log.jsonl
    python convert_logs_to_csv.py results/schemabank_seed42/training_log.jsonl -o custom_output.csv
"""

import json
import csv
import sys
import argparse
from pathlib import Path


def jsonl_to_csv(jsonl_path, csv_path=None):
    """
    Convert JSONL training log to CSV format
    
    Args:
        jsonl_path: Path to training_log.jsonl file
        csv_path: Optional output CSV path (default: same name with .csv extension)
    """
    jsonl_path = Path(jsonl_path)
    
    if not jsonl_path.exists():
        print(f"✗ Error: File not found: {jsonl_path}")
        return False
    
    # Determine output path
    if csv_path is None:
        csv_path = jsonl_path.with_suffix('.csv')
    else:
        csv_path = Path(csv_path)
    
    # Read all log entries
    print(f"Reading {jsonl_path}...")
    logs = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠ Warning: Skipping invalid JSON at line {line_num}: {e}")
    
    if not logs:
        print("✗ Error: No valid log entries found!")
        return False
    
    # Get all unique keys across all logs (handles varying columns across stages)
    all_keys = set()
    for log in logs:
        all_keys.update(log.keys())
    
    # Sort keys for consistent column order (step first, then alphabetical)
    fieldnames = ['step'] + sorted([k for k in all_keys if k != 'step'])
    
    # Write CSV
    print(f"Writing {csv_path}...")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(logs)
    
    print(f"\n✓ Success!")
    print(f"  Rows: {len(logs)}")
    print(f"  Columns: {len(fieldnames)}")
    print(f"  Output: {csv_path}")
    print(f"\nColumns: {', '.join(fieldnames)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert SchemaBank training logs from JSONL to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_logs_to_csv.py results/schemabank_seed42/training_log.jsonl
  python convert_logs_to_csv.py results/baseline_seed42/training_log.jsonl -o baseline.csv
        """
    )
    
    parser.add_argument(
        'jsonl_file',
        help='Path to training_log.jsonl file'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output CSV file path (default: same name as input with .csv extension)',
        default=None
    )
    
    args = parser.parse_args()
    
    success = jsonl_to_csv(args.jsonl_file, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
