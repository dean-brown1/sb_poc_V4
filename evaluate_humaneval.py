# evaluate_humaneval.py

#!/usr/bin/env python3
"""
HumanEval Evaluation Script

Standalone evaluation of trained models on HumanEval.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation_humaneval import eval_humaneval_passk
from src.data_humaneval import load_humaneval_data


def load_model_from_checkpoint(checkpoint_path):
    """
    Load model from checkpoint directory
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        (model, tokenizer)
    """
    checkpoint_path = Path(checkpoint_path)
    
    print(f"Loading model from {checkpoint_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("✓ Model loaded")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on HumanEval")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (default: checkpoint_dir/evaluation.json)"
    )
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    
    print("="*70)
    print("HUMANEVAL EVALUATION")
    print("="*70)
    
    # Load model
    model, tokenizer = load_model_from_checkpoint(checkpoint_path)
    
    # Load test data
    print("\nLoading HumanEval test set...")
    humaneval_test = load_humaneval_data("test")
    print(f"✓ Loaded {len(humaneval_test)} problems")
    
    print("\n" + "="*70)
    print("Evaluating pass@1...")
    print("="*70)
    
    # Evaluate
    results = eval_humaneval_passk(
        model,
        tokenizer,
        humaneval_test,
        device="cuda",
        k=1,
        num_samples_per_task=1,
        temperature=0.0,
        max_new_tokens=512
    )
    
    print(f"\n✓ pass@1: {results['pass_at_k']:.1%}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path / "evaluation.json"
    
    with open(output_path, 'w') as f:
        json.dump({'humaneval': results}, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()