"""
Evaluation Functions

Includes:
- GSM8K accuracy (FIXED: proper total counting)
- Schema reuse measurement
- Long-context stability (perplexity)
- Schema usage analysis
"""

import re
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def extract_answer_number(text):
    """
    Extract numerical answer from GSM8K format
    
    GSM8K answers are in format: "explanation #### number"
    This extracts the number after ####
    
    Args:
        text: Answer string from GSM8K
        
    Returns:
        Numerical answer as string (with commas removed), or None if not found
    """
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    return None


def extract_generated_number(text):
    """
    Extract predicted number from generated text
    
    Strategy: Find all numbers in the generated text, use the last one
    This handles various output formats (with/without formatting)
    
    Args:
        text: Generated answer text
        
    Returns:
        Last number found in text, or None if no numbers found
    """
    numbers = re.findall(r'\b\d+\b', text)
    return numbers[-1] if numbers else None


def eval_gsm8k_accuracy(model, tokenizer, test_dataset, device="cuda", max_samples=500):
    """
    Evaluate GSM8K accuracy with FIXED total counting
    
    CRITICAL FIX: V3 had bug where total was never incremented,
    causing division by zero and 0.0% accuracy reporting.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        test_dataset: GSM8K test split
        device: Device to run on
        max_samples: Max number of problems to evaluate
        
    Returns:
        dict with:
            - accuracy: Float accuracy [0, 1]
            - correct: Number correct
            - total: Number evaluated
            - samples: List of sample predictions for inspection
    """
    model.eval()
    
    correct = 0
    total = 0
    samples = []
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(
            test_dataset.select(range(min(max_samples, len(test_dataset)))),
            desc="Evaluating GSM8K"
        )):
            question = example["question"]
            true_answer = example["answer"]
            
            # Extract true numerical answer
            true_num = extract_answer_number(true_answer)
            if not true_num:
                continue  # Skip if answer format is invalid
            
            # CRITICAL FIX: Increment total for each valid problem
            total += 1
            
            # Generate answer
            prompt = f"{question}\nThe answer is:"
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(device)
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_answer = generated[len(prompt):]  # Remove prompt
            
            # Extract predicted number
            pred_num = extract_generated_number(generated_answer)
            
            # Check correctness
            is_correct = (pred_num == true_num)
            if is_correct:
                correct += 1
            
            # Store sample for inspection (first 10 only)
            if len(samples) < 10:
                samples.append({
                    "index": idx,
                    "question": question[:100] + "...",
                    "true_answer": true_num,
                    "predicted": pred_num,
                    "correct": is_correct,
                    "generated": generated_answer[:100] + "..."
                })
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "samples": samples
    }


def eval_schema_reuse(model, tokenizer, paraphrase_pairs, device="cuda", topk=2):
    """
    Measure schema routing consistency across paraphrases
    
    For each (question, paraphrase) pair, measure how many of the top-k
    selected schemas overlap. Higher overlap = more consistent routing.
    
    Args:
        model: Model with schemabank_adapters
        tokenizer: Tokenizer
        paraphrase_pairs: List of (original, paraphrase) question pairs
        device: Device
        topk: Number of top schemas to compare
        
    Returns:
        Average overlap score [0, topk]
    """
    if not hasattr(model, 'schemabank_adapters'):
        return 0.0
    
    model.eval()
    overlaps = []
    
    with torch.no_grad():
        for orig, para in tqdm(paraphrase_pairs, desc="Schema reuse"):
            # Get routing for original
            inputs_orig = tokenizer(orig, return_tensors="pt").to(device)
            with torch.no_grad():
                out_orig = model(**inputs_orig, output_hidden_states=True)
            
            # Get routing for paraphrase
            inputs_para = tokenizer(para, return_tensors="pt").to(device)
            with torch.no_grad():
                out_para = model(**inputs_para, output_hidden_states=True)
            
            # Extract hidden states and get routing gates
            # This requires hooking or modifying forward to capture gates
            # For now, placeholder - needs integration with model forward
            # overlaps.append(compute_overlap(...))
            pass
    
    return np.mean(overlaps) if overlaps else 0.0


def eval_perplexity(model, tokenizer, dataset, device="cuda", seq_len=512, num_samples=100):
    """
    Evaluate perplexity on dataset at given sequence length
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for example in tqdm(
            dataset.select(range(min(num_samples, len(dataset)))),
            desc=f"Perplexity (seq_len={seq_len})"
        ):
            text = example.get("text", example.get("question", ""))
            
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=seq_len
            ).to(device)
            
            input_ids = inputs["input_ids"]
            
            # Get logits
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate loss manually with reduction='sum'
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )
            
            # Count valid tokens (excluding padding)
            num_tokens = (shift_labels != tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item()
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

def eval_long_context_stability(model, tokenizer, dataset, device="cuda", num_samples=100):
    """
    Measure perplexity gap between short and long contexts
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer  
        dataset: Dataset
        device: Device
        num_samples: Samples to use
        
    Returns:
        dict with ppl_512, ppl_4096, rel_gap
    """
    ppl_512 = eval_perplexity(model, tokenizer, dataset, device, seq_len=512, num_samples=num_samples)
    ppl_4096 = eval_perplexity(model, tokenizer, dataset, device, seq_len=4096, num_samples=num_samples)
    
    rel_gap = (ppl_4096 - ppl_512) / max(ppl_512, 1e-9)
    
    return {
        "ppl_512": ppl_512,
        "ppl_4096": ppl_4096,
        "rel_gap": rel_gap
    }


def analyze_schema_usage(model, tokenizer, dataset, device="cuda", num_samples=500):
    """
    Analyze which schemas are used and how often
    
    Requires model to return routing gates during forward pass.
    
    Returns:
        dict with usage statistics, dead schemas, entropy, etc.
    """
    if not hasattr(model, 'schemabank_adapters'):
        return {
            "active_schemas": 0,
            "dead_schemas": [],
            "entropy": 0.0,
            "usage_distribution": {}
        }
    
    # Placeholder - needs integration with routing gate extraction
    # This would track schema activation frequencies across dataset
    
    return {
        "active_schemas": 0,
        "dead_schemas": [],
        "entropy": 0.0,
        "usage_distribution": {}
    }
