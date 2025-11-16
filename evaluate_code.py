# evaluate_code.py

#!/usr/bin/env python3
"""
Code Evaluation Script

Standalone evaluation for trained code generation models.

Usage:
    python evaluate_code.py --checkpoint results/baseline_mbpp_2epochs_seed42/checkpoint --dataset mbpp
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation_code import eval_code_passk
from src.data_code import load_code_data


def load_checkpoint(checkpoint_path):
    """Load model and tokenizer"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("✓ Model loaded\n")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", default="mbpp", choices=["mbpp"], help="Dataset to evaluate on")
    parser.add_argument("--split", default="test", help="Dataset split (test, validation, etc.)")
    parser.add_argument("--output", default=None, help="Output file (default: checkpoint_dir/evaluation.json)")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    
    print("="*70)
    print(f"CODE EVALUATION: {args.dataset.upper()}")
    print("="*70)
    
    # Load model
    model, tokenizer = load_checkpoint(checkpoint_path)
    
    # Load test data
    print(f"\nLoading {args.dataset} {args.split} set...")
    test_data = load_code_data(args.dataset, args.split)
    print(f"✓ Loaded {len(test_data)} problems\n")
    
    # Evaluate
    results = eval_code_passk(
        model,
        tokenizer,
        test_data,
        dataset_name=args.dataset,
        device="cuda",
        k=1,
        temperature=0.0,
        max_new_tokens=512
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path / "evaluation.json"
    
    with open(output_path, 'w') as f:
        json.dump({args.dataset: results}, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()