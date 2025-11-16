"""
Data Preparation Module

Includes:
- GSM8K dataset loading and formatting
- Schema tagging (hash-based and content-based)
- Tag dropout curriculum
- Data collation and packing
"""

import re
import random
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from .data_humaneval import (
    load_humaneval_data,
    prepare_humaneval_dataset,
    create_humaneval_dataloader
)

# ========== Schema Tagging ==========

def assign_schema_tags_hash(question, num_schemas=32):
    """
    Deterministic hash-based schema assignment
    
    Ensures same question → same schemas across runs.
    Simple, no manual labeling needed.
    
    Args:
        question: Question text
        num_schemas: Total number of schemas
        
    Returns:
        List of 2 schema indices (sorted)
    """
    h = hash(question)
    schema_1 = h % num_schemas
    schema_2 = (h // num_schemas) % num_schemas
    return sorted([schema_1, schema_2])


def assign_schema_tags_content(question, num_schemas=32):
    """
    Content-based schema assignment (OPTIONAL - more accurate)
    
    Assigns schemas based on detected operations and complexity.
    Can be enabled by changing tagging_method parameter.
    
    Schema allocation:
    - 0-7:   Addition (simple → complex)
    - 8-15:  Subtraction
    - 16-23: Multiplication
    - 24-31: Division & Multi-step
    
    Args:
        question: Question text
        num_schemas: Total number of schemas
        
    Returns:
        List of 2 schema indices (sorted)
    """
    q_lower = question.lower()
    
    # Operation detection
    operations = []
    if any(w in q_lower for w in ['times', 'multiply', 'each', 'per']):
        operations.append('mult')
    if any(w in q_lower for w in ['divide', 'split', 'shared', 'average']):
        operations.append('div')
    if any(w in q_lower for w in ['plus', 'add', 'total', 'sum', 'altogether']):
        operations.append('add')
    if any(w in q_lower for w in ['minus', 'subtract', 'left', 'remaining', 'less']):
        operations.append('sub')
    
    # Complexity detection
    num_numbers = len(re.findall(r'\d+', question))
    is_complex = num_numbers > 3
    
    # Schema allocation
    op_to_base = {
        'add': 0,
        'sub': 8,
        'mult': 16,
        'div': 24,
    }
    
    schemas = []
    for op in operations[:2]:  # Top 2 operations
        base = op_to_base.get(op, 0)
        offset = 4 if is_complex else 0  # Complex problems use higher schemas
        schemas.append(base + offset)
    
    # Fallback to hash if no operations detected
    if len(schemas) < 2:
        h = hash(question)
        while len(schemas) < 2:
            schemas.append(h % num_schemas)
            h = h // num_schemas
    
    return sorted(schemas[:2])


def get_tag_dropout_rate(step, total_steps):
    """
    Progressive tag dropout schedule for curriculum learning
    
    Gradually reduces tag supervision to force router generalization.
    
    Quarter 1 (0-25%):   0% dropout → 100% tags present (learn mapping)
    Quarter 2 (25-50%):  25% dropout → 25% tags present (start generalizing)
    Quarter 3 (50-75%):  50% dropout → 50% tags present (more independent)
    Quarter 4 (75-100%): 75% dropout → 75% tags present (mostly autonomous)
    
    Args:
        step: Current training step
        total_steps: Total steps in stage
        
    Returns:
        Dropout rate (0.0 = keep all tags, 1.0 = drop all tags)
    """
    progress = step / max(total_steps, 1)
    
    if progress < 0.25:
        return 0.0    # Keep all tags
    elif progress < 0.50:
        return 0.25   # Drop 25% of tags
    elif progress < 0.75:
        return 0.50   # Drop 50% of tags
    else:
        return 0.75   # Drop 75% of tags


# ========== Dataset Loading ==========

def load_gsm8k_data(split="train"):
    """
    Load GSM8K dataset from Hugging Face
    
    Args:
        split: 'train' or 'test'
        
    Returns:
        Dataset object
    """
    ds = load_dataset("gsm8k", "main", split=split)
    return ds


def format_gsm8k_example(example, schema_tags=None):
    """
    Format GSM8K example with optional schema tags
    
    Format with tags:    "[Schema: 5,12] Question: ... Answer: ..."
    Format without tags: "Question: ... Answer: ..."
    
    Args:
        example: Dict with 'question' and 'answer'
        schema_tags: Optional list of schema indices [s1, s2]
        
    Returns:
        Formatted text string
    """
    question = example["question"]
    answer = example["answer"]
    
    # Extract final numerical answer (after ####)
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', answer)
    if match:
        final_answer = match.group(1).replace(',', '')
    else:
        final_answer = "unknown"
    
    # Prepend schema tags if provided
    if schema_tags is not None:
        tag_str = ",".join(map(str, schema_tags))
        text = f"[Schema: {tag_str}] {question}\nThe answer is: {final_answer}"
    else:
        text = f"{question}\nThe answer is: {final_answer}"
    
    return text


def prepare_gsm8k_dataset(dataset, num_schemas=32, tagging_method='hash'):
    """
    Prepare GSM8K dataset with schema tags
    
    Each example gets assigned 2 schemas using either hash-based
    (deterministic) or content-based (operation-aware) assignment.
    
    Args:
        dataset: GSM8K dataset from load_gsm8k_data
        num_schemas: Number of schemas (default 32)
        tagging_method: 'hash' (deterministic) or 'content' (operation-based)
    
    Returns:
        List of dicts with 'example' and 'schema_tags'
    """
    tagged_data = []
    
    tag_fn = assign_schema_tags_hash if tagging_method == 'hash' else assign_schema_tags_content
    
    for ex in dataset:
        schema_tags = tag_fn(ex["question"], num_schemas)
        tagged_data.append({
            'example': ex,
            'schema_tags': schema_tags
        })
    
    return tagged_data


def pack_gsm8k_with_tags(tagged_data, tokenizer, max_len, tag_dropout_rate=0.0):
    """
    Pack GSM8K examples into token IDs with tag dropout
    
    Tag dropout is used for curriculum learning: start with all tags,
    gradually drop more to force router generalization.
    
    Args:
        tagged_data: List of dicts with 'example' and 'schema_tags'
        tokenizer: Tokenizer
        max_len: Max sequence length
        tag_dropout_rate: Probability of dropping tags (0.0 = keep all, 1.0 = drop all)
    
    Returns:
        List of token ID lists
    """
    texts = []
    for item in tagged_data:
        # Apply dropout: randomly decide whether to include tags
        if random.random() < tag_dropout_rate:
            # Drop tags for this example
            text = format_gsm8k_example(item['example'], schema_tags=None)
        else:
            # Keep tags
            text = format_gsm8k_example(item['example'], schema_tags=item['schema_tags'])
        texts.append(text)
    
    return [
        tokenizer(t, truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]
        for t in texts
    ]


def collate(batch, pad_token_id):
    """
    Collate batch of token IDs with padding
    
    Args:
        batch: List of token ID lists
        pad_token_id: ID to use for padding
        
    Returns:
        Dict with input_ids, attention_mask, labels tensors
    """
    L = max(len(x) for x in batch)
    ids = [x + [pad_token_id] * (L - len(x)) for x in batch]
    att = [[1] * len(x) + [0] * (L - len(x)) for x in batch]
    
    return {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.tensor(att),
        "labels": torch.tensor(ids)
    }


def create_dataloader(tagged_data, tokenizer, max_len, batch_size, 
                      tag_dropout_rate=0.0, shuffle=True):
    """
    Create DataLoader for training
    
    Automatically detects dataset type (GSM8K or HumanEval) and uses
    appropriate data packing function.
    """
    # Debug: check what we got
    if not tagged_data:
        raise ValueError("tagged_data is empty")
    
    first_item = tagged_data[0]
    print(f"DEBUG: First item keys = {first_item.keys()}")
    
    # Detect dataset type
    if 'prompt' in first_item and 'solution' in first_item:
        # HumanEval format
        return create_humaneval_dataloader(
            tagged_data,
            tokenizer,
            max_len=max_len,
            batch_size=batch_size,
            tag_dropout_rate=tag_dropout_rate,
            shuffle=shuffle
        )
    elif 'example' in first_item:
        # GSM8K format
        ids = pack_gsm8k_with_tags(tagged_data, tokenizer, max_len, tag_dropout_rate)
    else:
        raise ValueError(f"Unknown dataset format. Keys: {first_item.keys()}")
    
    # GSM8K path
    ds = Dataset.from_dict({"input_ids": ids})
    
    pad_token_id = tokenizer.pad_token_id
    dl = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=lambda x: collate([r["input_ids"] for r in x], pad_token_id)
    )
    
    return dl


# ========== Synthetic Data (for schema reuse testing) ==========

def make_paraphrase_pairs(n=20000):
    """
    Generate synthetic paraphrase pairs for schema reuse testing
    
    Creates pairs of semantically similar sentences that should route
    to the same schemas if the router has learned compositional patterns.
    
    Args:
        n: Number of pairs to generate
        
    Returns:
        List of (original, paraphrase) tuples
    """
    names = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Gina", "Hector"]
    items = ["laptop", "router", "server", "notebook", "dataset", "report", "ticket"]
    verbs = ["fixed", "broke", "moved", "updated", "reviewed", "tagged", "validated"]
    places = ["lab", "office", "DC-1", "staging", "prod", "Rack-4", "S3-bucket"]
    times = ["yesterday", "today", "last night", "this morning", "at noon"]
    
    pairs = []
    for _ in range(n):
        a = f"{random.choice(names)} {random.choice(verbs)} the {random.choice(items)} in {random.choice(places)} {random.choice(times)}."
        b = a.replace("the ", "a ").replace(" in ", " at ")
        if random.random() < 0.33:
            b = b.replace("Alice", "Zara").replace("Bob", "Nate").replace("Carol", "Mira")
        pairs.append((a, b))
    
    return pairs
