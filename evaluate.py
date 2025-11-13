#!/usr/bin/env python3
"""
SchemaBank Evaluation Script

Standalone evaluation of trained models on GSM8K and other metrics.

Usage:
    python evaluate.py --checkpoint results/schemabank_seed42/checkpoint
    python evaluate.py --checkpoint results/baseline_seed42/checkpoint --output eval_results.json
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation import (
    eval_gsm8k_accuracy,
    eval_long_context_stability,
    analyze_schema_usage
)
from src.data import load_gsm8k_data


def load_model_from_checkpoint(checkpoint_path):
    """
    Load model from checkpoint directory
    
    Handles both baseline and SchemaBank models.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        (model, tokenizer, has_schemabank)
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
    
    # CRITICAL: Set to evaluation mode
    model.eval()
    
    # Explicitly disable dropout
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    
    # Check if SchemaBank adapters exist
    sb_path = checkpoint_path / 'schemabank_adapters.pt'
    config_path = checkpoint_path / 'schemabank_config.json'
    has_schemabank = False
    
    if sb_path.exists() and config_path.exists():
        print("Loading SchemaBank adapters...")
        
        from src.model import attach_schemabank_last2
        
        # Load config
        with open(config_path, 'r') as f:
            sb_config = json.load(f)
        
        # Attach SchemaBank - modifies model in-place
        attach_schemabank_last2(
            model,
            H=sb_config['hidden_size'],
            S=sb_config['num_schemas'],
            r=sb_config['rank'],
            topk=sb_config['topk'],
            ad=32
        )
        
        # Load weights directly
        sb_state = torch.load(sb_path, map_location='cpu')
        model.load_state_dict(sb_state, strict=False)
        
        print(f"✓ SchemaBank loaded: {sb_config['num_schemas']} schemas, rank {sb_config['rank']}")
        has_schemabank = True
    else:
        print("✓ Baseline model loaded (no SchemaBank)")

    return model, tokenizer, has_schemabank

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
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
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of GSM8K test samples to evaluate (default: 500)"
    )
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    
    print("="*70)
    print("SCHEMABANK EVALUATION")
    print("="*70)
    
    # Load model
    model, tokenizer, has_schemabank = load_model_from_checkpoint(checkpoint_path)
    
    # Load test data
    print("\nLoading GSM8K test set...")
    gsm8k_test = load_gsm8k_data("test")
    print(f"✓ Loaded {len(gsm8k_test)} test examples")
    
    print("\n" + "="*70)
    print("EVALUATION 1: GSM8K Accuracy")
    print("="*70)
    
    # Evaluate GSM8K accuracy
    gsm8k_results = eval_gsm8k_accuracy(
        model,
        tokenizer,
        gsm8k_test,
        device="cuda",
        max_samples=args.num_samples
    )
    
    print(f"\n✓ Results:")
    print(f"  Accuracy: {gsm8k_results['accuracy']:.1%}")
    print(f"  Correct: {gsm8k_results['correct']}/{gsm8k_results['total']}")
    
    print("\n" + "="*70)
    print("EVALUATION 2: Long-Context Stability")
    print("="*70)
    
    # Evaluate perplexity
    lc_results = eval_long_context_stability(
        model,
        tokenizer,
        gsm8k_test,
        device="cuda",
        num_samples=100
    )
    
    print(f"\n✓ Results:")
    print(f"  PPL (512 tokens):  {lc_results['ppl_512']:.2f}")
    print(f"  PPL (4096 tokens): {lc_results['ppl_4096']:.2f}")
    print(f"  Relative gap:      {lc_results['rel_gap']:.4f}")
    
    # Compile results
    results = {
        'checkpoint': str(checkpoint_path),
        'has_schemabank': has_schemabank,
        'gsm8k': gsm8k_results,
        'long_context': lc_results
    }
    
    # Schema analysis (if SchemaBank)
    if has_schemabank:
        print("\n" + "="*70)
        print("EVALUATION 3: Schema Usage Analysis")
        print("="*70)
        
        schema_results = analyze_schema_usage(
            model,
            tokenizer,
            gsm8k_test,
            device="cuda",
            num_samples=args.num_samples
        )
        
        results['schema_analysis'] = schema_results
        
        print(f"\n✓ Schema statistics captured")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path / "evaluation.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: {checkpoint_path.name}")
    print(f"Type: {'SchemaBank' if has_schemabank else 'Baseline'}")
    print(f"GSM8K Accuracy: {gsm8k_results['accuracy']:.1%} ({gsm8k_results['correct']}/{gsm8k_results['total']})")
    print(f"PPL Gap: {lc_results['rel_gap']:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()