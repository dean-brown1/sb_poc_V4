#!/usr/bin/env python3
"""
SchemaBank Training Script

Main entry point for training baseline or SchemaBank models.

Usage:
    python train.py --config configs/baseline.yaml
    python train.py --config configs/schemabank.yaml
"""

import argparse
import os
from pathlib import Path

from src.utils import (
    load_config,
    set_seed,
    TelemetryLogger,
    print_config_summary,
)
from src.data import load_gsm8k_data, prepare_gsm8k_dataset
from src.training import prepare_model, train_baseline, train_schemabank



def main():
    parser = argparse.ArgumentParser(description="Train SchemaBank or baseline model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/baseline.yaml)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    exp_config = config['experiment']
    
    # Auto-increment run number if output directory exists
    base_output = Path(exp_config['output_dir'])
    if base_output.exists():
        parent = base_output.parent
        basename = base_output.name
        
        run_num = 1
        while True:
            new_output = parent / f"{basename}_run{run_num:03d}"
            if not new_output.exists():
                exp_config['output_dir'] = str(new_output)
                print(f"⚠️  Output directory exists, using: {exp_config['output_dir']}")
                break
            run_num += 1
    
    # Print configuration summary
    print_config_summary(config)
    
    # Set random seed
    set_seed(exp_config['seed'])
    
    # Create output directory
    output_dir = Path(exp_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize telemetry logger
    log_path = output_dir / "training_log.jsonl"
    telemetry_logger = TelemetryLogger(log_path)
    
    print("\n" + "="*70)
    print("PHASE 1: Model Preparation")
    print("="*70)
    
    # Prepare model
    model, tokenizer, hidden_size = prepare_model(config, mode=exp_config['mode'])
    print(f"✓ Model prepared: {config['model']['base_model']}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Mode: {exp_config['mode']}")
    
    print("\n" + "="*70)
    print("PHASE 2: Data Preparation")
    print("="*70)
    
# Check which dataset to use
    dataset_name = config['dataset'].get('name', 'gsm8k')  # Default to gsm8k for backward compatibility

    if dataset_name == 'humaneval':
        from src.data_humaneval import load_humaneval_data, prepare_humaneval_dataset
        
        print("Loading HumanEval dataset...")
        raw_data = load_humaneval_data("test")
        print(f"✓ Loaded {len(raw_data)} problems")
        
        if exp_config['mode'] == 'schemabank':
            tagging_method = config['dataset'].get('tagging_method', 'hash')
            num_schemas = config['schemabank']['num_schemas']
            print(f"Preparing data with schema tags (method: {tagging_method})...")
            tagged_data = prepare_humaneval_dataset(
                raw_data,
                num_schemas=num_schemas,
                tagging_method=tagging_method
            )
        else:
            print("Preparing data for baseline training...")
            tagged_data = prepare_humaneval_dataset(raw_data, num_schemas=32, tagging_method='hash')
        
        print(f"✓ Prepared {len(tagged_data)} training examples")

    else:  # gsm8k (existing code)
        print("Loading GSM8K dataset...")
        gsm8k_train = load_gsm8k_data("train")
        gsm8k_test = load_gsm8k_data("test")
        print(f"✓ Loaded {len(gsm8k_train)} training examples")
        print(f"✓ Loaded {len(gsm8k_test)} test examples")
        
        # Prepare tagged data (for SchemaBank) or regular data (for baseline)
        if exp_config['mode'] == 'schemabank':
            tagging_method = config['dataset'].get('tagging_method', 'hash')
            num_schemas = config['schemabank']['num_schemas']
            print(f"Preparing data with schema tags (method: {tagging_method})...")
            tagged_data = prepare_gsm8k_dataset(
                gsm8k_train,
                num_schemas=num_schemas,
                tagging_method=tagging_method
            )
        else:
            # For baseline, still use the same structure but tags won't be used
            print("Preparing data for baseline training...")
            tagged_data = prepare_gsm8k_dataset(gsm8k_train, num_schemas=32, tagging_method='hash')
        
        print(f"✓ Prepared {len(tagged_data)} training examples")
    
    print("\n" + "="*70)
    print("PHASE 3: Training")
    print("="*70)
    
        
    # Train model
    if exp_config['mode'] == 'baseline':
        model = train_baseline(model, tagged_data, tokenizer, config, telemetry_logger)
    else:  # schemabank
        model = train_schemabank(model, tagged_data, tokenizer, config, telemetry_logger)
    
        # DEBUG: Check if SchemaBank is still attached
    print("\n=== DEBUG: Checking model state after training ===")
    print(f"Has schemabank_adapters: {hasattr(model, 'schemabank_adapters')}")
    if hasattr(model, 'schemabank_adapters'):
        print(f"Number of adapters: {len(model.schemabank_adapters)}")
        # Check if hooks are still registered
        from src.model import get_block_list
        blocks = get_block_list(model)
        for i, idx in enumerate([-2, -1]):
            block = blocks[idx]
            print(f"Block {idx}: hooks={len(block._forward_hooks)}, has_adapter_module={hasattr(block, 'schemabank_adapter')}")
    else:
        print("⚠️ WARNING: SchemaBank adapters are MISSING after training!")
    
    
    print("\n" + "="*70)
    print("PHASE 5: Saving Results")
    print("="*70)
    
    # Save model checkpoint
    checkpoint_dir = output_dir / "checkpoint"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"✓ Model saved to {checkpoint_dir}")
    
    # Save SchemaBank adapters separately if present
    if hasattr(model, 'schemabank_adapters'):
        import torch
        sb_state = {}
        for i, adapter in enumerate(model.schemabank_adapters):
            sb_state[f'adapter_{i}'] = adapter.state_dict()
        
        sb_path = checkpoint_dir / 'schemabank_adapters.pt'
        torch.save(sb_state, sb_path)
        
        # Save SchemaBank config
        import json
        sb_config = {
            'num_schemas': config['schemabank']['num_schemas'],
            'rank': config['schemabank']['rank'],
            'topk': config['schemabank']['topk'],
            'hidden_size': hidden_size
        }
        with open(checkpoint_dir / 'schemabank_config.json', 'w') as f:
            json.dump(sb_config, f, indent=2)
        
        print(f"✓ SchemaBank adapters saved to {sb_path}")
        
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - config.yaml")
    print(f"  - results.json")
    print(f"  - training_log.jsonl")
    print(f"  - checkpoint/")
    print("\n")


if __name__ == "__main__":
    main()
