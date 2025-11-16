# src/evaluation_code.py

"""
Code Evaluation

Unified evaluation for code generation datasets.
"""

import torch
from tqdm import tqdm


def eval_code_passk(
    model,
    tokenizer,
    dataset,
    dataset_name,
    device="cuda",
    k=1,
    temperature=0.0,
    max_new_tokens=512
):
    """
    Evaluate model on code dataset with pass@k
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: Code dataset (tagged or raw)
        dataset_name: 'mbpp'
        device: Device
        k: k for pass@k
        temperature: Sampling temperature
        max_new_tokens: Max tokens to generate
        
    Returns:
        dict with results
    """
    model.eval()
    
    passed = 0
    total = 0
    
    print(f"Evaluating {len(dataset)} problems...")
    
    with torch.no_grad():
        for example in tqdm(dataset, desc="Generating code"):
            # Get problem text (handles both raw and tagged format)
            if isinstance(example, dict) and 'problem' in example:
                # Tagged format
                problem = example['problem']
                tests = example['tests']
                task_id = example.get('task_id', 'unknown')
            else:
                # Raw format - get correct field names
                from .data_code import get_field_names
                problem_field, solution_field, test_field = get_field_names(dataset_name)
                problem = example[problem_field]
                tests = example[test_field]
                task_id = example.get('task_id', 'unknown')
            
            # Generate code
            inputs = tokenizer(
                problem,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated[len(problem):]
            
            # Test the code
            test_passed = run_code_tests(completion, tests, dataset_name)
            
            if test_passed:
                passed += 1
            total += 1
    
    accuracy = passed / total if total > 0 else 0.0
    
    print(f"\n✓ Results:")
    print(f"  pass@{k}: {accuracy:.1%} ({passed}/{total})")
    
    return {
        'pass_at_k': accuracy,
        'k': k,
        'num_samples': total,
        'num_tasks': total,
        'passed': passed
    }


def run_code_tests(code, tests, dataset_name):
    """
    Run unit tests on generated code
    
    Args:
        code: Generated code string
        tests: Test data (format depends on dataset)
        dataset_name: Which dataset
        
    Returns:
        True if tests pass, False otherwise
    """
    try:
        exec_globals = {}
        
        # Execute the generated code
        exec(code, exec_globals)
        
        # Run tests based on dataset format
        if dataset_name == "mbpp":
            # MBPP: test_list is a list of assertion strings
            for test in tests:
                exec(test, exec_globals)
        
       
        return True
        
    except Exception as e:
        return False


def eval_code_quick(
    model,
    tokenizer,
    dataset,
    dataset_name,
    device="cuda",
    max_samples=None,
    max_new_tokens=512
):
    """
    Quick code evaluation (basic sanity check)
    
    Args:
        model: Model
        tokenizer: Tokenizer
        dataset: Code dataset
        dataset_name: Which dataset
        device: Device
        max_samples: Max problems to test
        max_new_tokens: Max tokens
        
    Returns:
        dict with results
    """
    model.eval()
    
    if max_samples:
        dataset = dataset[:max_samples]
    
    correct = 0
    total = 0
    
    print(f"Quick evaluation on {len(dataset)} problems...")
    
    with torch.no_grad():
        for example in tqdm(dataset, desc="Evaluating"):
            # Get problem
            if isinstance(example, dict) and 'problem' in example:
                problem = example['problem']
            else:
                from .data_code import get_field_names
                problem_field, _, _ = get_field_names(dataset_name)
                problem = example[problem_field]
            
            # Generate
            inputs = tokenizer(
                problem,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = generated[len(problem):]
            
            # Basic check: contains 'def'
            if 'def ' in completion:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }