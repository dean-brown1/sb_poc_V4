# /src/evaluation_humaneval.py

"""
HumanEval Evaluation

Evaluates code generation using unit tests.
Uses the official HumanEval evaluation harness.
"""

import torch
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
import tempfile
import os


def eval_humaneval_passk(
    model,
    tokenizer,
    dataset,
    device="cuda",
    k=1,
    num_samples_per_task=1,
    temperature=0.0,
    max_new_tokens=512
):
    """
    Evaluate model on HumanEval with pass@k metric
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: HumanEval dataset (tagged or raw)
        device: Device
        k: k for pass@k metric (typically 1, 10, or 100)
        num_samples_per_task: Number of solutions to generate per problem
        temperature: Sampling temperature (0.0 = greedy)
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        dict with pass@k results
    """
    model.eval()
    
    # Prepare samples for evaluation
    samples = []
    
    print(f"Generating solutions for {len(dataset)} problems...")
    
    with torch.no_grad():
        for example in tqdm(dataset, desc="Generating code"):
            # Get prompt (function signature + docstring)
            if isinstance(example, dict) and 'prompt' in example:
                prompt = example['prompt']
                task_id = example.get('task_id', example.get('task_id', 'unknown'))
            else:
                # Raw HumanEval format
                prompt = example['prompt']
                task_id = example['task_id']
            
            # Generate multiple samples per task if requested
            for sample_idx in range(num_samples_per_task):
                # Tokenize prompt
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                ).to(device)
                
                # Generate completion
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else 1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode generated code
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the completion (remove prompt)
                completion = generated[len(prompt):]
                
                # Store sample for evaluation
                samples.append({
                    'task_id': task_id,
                    'completion': completion
                })
    
    print(f"\n✓ Generated {len(samples)} solutions")
    
    # Write samples to temporary file for evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        write_jsonl(f.name, samples)
        samples_file = f.name
    
    try:
        # Run official HumanEval evaluation
        print(f"\nEvaluating with pass@{k}...")
        results = evaluate_functional_correctness(
            sample_file=samples_file,
            k=[k],
            n_workers=4,
            timeout=3.0
        )
        
        pass_at_k = results[f'pass@{k}']
        
        print(f"\n✓ Results:")
        print(f"  pass@{k}: {pass_at_k:.1%}")
        
        return {
            'pass_at_k': pass_at_k,
            'k': k,
            'num_samples': len(samples),
            'num_tasks': len(dataset)
        }
        
    finally:
        # Cleanup temp file
        if os.path.exists(samples_file):
            os.remove(samples_file)


def eval_humaneval_accuracy_simple(
    model,
    tokenizer,
    dataset,
    device="cuda",
    max_samples=None,
    max_new_tokens=512
):
    """
    Simplified HumanEval evaluation (greedy decoding, basic checks)
    
    This is faster than full pass@k but less rigorous.
    Use for quick validation during training.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: HumanEval dataset
        device: Device
        max_samples: Maximum problems to evaluate
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        dict with basic accuracy metrics
    """
    model.eval()
    
    if max_samples:
        dataset = dataset[:max_samples]
    
    correct = 0
    total = 0
    samples = []
    
    print(f"Quick evaluation on {len(dataset)} problems...")
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
            if isinstance(example, dict) and 'prompt' in example:
                prompt = example['prompt']
                task_id = example.get('task_id', f'task_{idx}')
            else:
                prompt = example['prompt']
                task_id = example['task_id']
            
            # Generate code
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated[len(prompt):]
            
            # Basic check: does it contain 'return'?
            # (Very rough proxy for valid code)
            has_return = 'return' in completion.lower()
            
            if has_return:
                correct += 1
            
            total += 1
            
            # Store first few samples for inspection
            if len(samples) < 5:
                samples.append({
                    'task_id': task_id,
                    'prompt': prompt[:100] + "...",
                    'completion': completion[:200] + "...",
                    'has_return': has_return
                })
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'samples': samples
    }


def analyze_humaneval_schemas(
    model,
    tokenizer,
    dataset,
    device="cuda"
):
    """
    Analyze which schemas are used for different code patterns
    
    Similar to GSM8K schema analysis but for code.
    
    Args:
        model: Model with SchemaBank
        tokenizer: Tokenizer
        dataset: HumanEval dataset
        device: Device
        
    Returns:
        dict with schema usage statistics
    """
    if not hasattr(model, 'schemabank_adapters'):
        return {
            "active_schemas": 0,
            "dead_schemas": [],
            "usage_by_task": {}
        }
    
    # Placeholder for now
    # Would track which schemas activate for different code patterns
    # (sorting, recursion, string manipulation, etc.)
    
    return {
        "active_schemas": 0,
        "dead_schemas": [],
        "usage_by_task": {}
    }