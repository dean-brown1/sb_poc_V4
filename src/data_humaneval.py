# src/data_humaneval.py


"""
HumanEval Data Preparation

Adapts SchemaBank training for code generation tasks.
"""

import hashlib
import re
from datasets import load_dataset
from typing import List, Dict


def load_humaneval_data(split="test"):
    """
    Load HumanEval dataset
    
    Note: HumanEval only has a 'test' split with 164 problems.
    For training, we'll use the canonical solutions.
    
    Args:
        split: Dataset split (only 'test' available)
        
    Returns:
        HuggingFace dataset
    """
    dataset = load_dataset("openai_humaneval", split=split)
    return dataset


def extract_function_signature(prompt: str) -> str:
    """
    Extract function name and first line of docstring for tagging
    
    Args:
        prompt: Full function prompt with signature and docstring
        
    Returns:
        String for hashing (function_name + first_docstring_line)
    """
    # Extract function name
    func_match = re.search(r'def\s+(\w+)\s*\(', prompt)
    func_name = func_match.group(1) if func_match else ""
    
    # Extract first line of docstring
    docstring_match = re.search(r'"""(.*?)(?:\n|""")', prompt, re.DOTALL)
    if docstring_match:
        first_line = docstring_match.group(1).strip().split('\n')[0]
    else:
        first_line = ""
    
    return f"{func_name}:{first_line}"


def assign_schema_tags_humaneval(
    dataset,
    num_schemas: int = 32,
    tagging_method: str = "hash"
) -> List[Dict]:
    """
    Assign schema tags to HumanEval problems
    
    Args:
        dataset: HumanEval dataset
        num_schemas: Number of schemas to distribute across
        tagging_method: Method for assignment ('hash' or 'semantic')
        
    Returns:
        List of examples with schema tags
    """
    tagged_examples = []
    
    for example in dataset:
        prompt = example['prompt']
        solution = example['canonical_solution']
        
        # Extract tagging key
        tag_key = extract_function_signature(prompt)
        
        # Hash to get 2 schemas
        if tagging_method == "hash":
            hash_val = int(hashlib.md5(tag_key.encode()).hexdigest(), 16)
            schema1 = hash_val % num_schemas
            schema2 = (hash_val // num_schemas) % num_schemas
            
            # Ensure different schemas
            if schema1 == schema2:
                schema2 = (schema2 + 1) % num_schemas
        else:
            # Fallback to simple hash if semantic not implemented
            schema1 = 0
            schema2 = 1
        
        tagged_examples.append({
            'task_id': example['task_id'],
            'prompt': prompt,
            'solution': solution,
            'test': example['test'],
            'entry_point': example['entry_point'],
            'schema_tags': [schema1, schema2],
            'tag_key': tag_key
        })
    
    return tagged_examples


def prepare_humaneval_dataset(
    humaneval_data,
    num_schemas: int = 32,
    tagging_method: str = "hash"
):
    """
    Prepare HumanEval data for SchemaBank training
    
    Creates training examples with schema tags.
    
    Args:
        humaneval_data: Raw HumanEval dataset
        num_schemas: Number of schemas
        tagging_method: Tagging approach
        
    Returns:
        Tagged dataset ready for training
    """
    print(f"Preparing HumanEval dataset with {num_schemas} schemas...")
    print(f"Tagging method: {tagging_method}")
    
    # Assign tags
    tagged_data = assign_schema_tags_humaneval(
        humaneval_data,
        num_schemas=num_schemas,
        tagging_method=tagging_method
    )
    
    print(f"✓ Prepared {len(tagged_data)} problems")
    
    # Show example
    if tagged_data:
        example = tagged_data[0]
        print(f"\nExample tagging:")
        print(f"  Task: {example['task_id']}")
        print(f"  Tag key: {example['tag_key']}")
        print(f"  Schemas: {example['schema_tags']}")
    
    return tagged_data


def format_training_example(example, include_tags=True):
    """
    Format HumanEval example for training
    
    Args:
        example: Tagged example
        include_tags: Whether to include schema tags in prompt
        
    Returns:
        Formatted string for training
    """
    if include_tags:
        schema_str = f"[Schema: {example['schema_tags'][0]},{example['schema_tags'][1]}] "
    else:
        schema_str = ""
    
    # Format: tags + prompt + solution
    text = f"{schema_str}{example['prompt']}{example['solution']}"
    
    return text


def create_humaneval_dataloader(
    tagged_data,
    tokenizer,
    max_len=1024,
    batch_size=1,
    tag_dropout_rate=0.0,
    shuffle=True
):
    """
    Create dataloader for HumanEval training
    
    Similar to GSM8K dataloader but adapted for code.
    
    Args:
        tagged_data: Tagged HumanEval examples
        tokenizer: Tokenizer
        max_len: Maximum sequence length
        batch_size: Batch size
        tag_dropout_rate: Probability of dropping tags (0.0 = keep all)
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    import random
    
    class HumanEvalDataset(Dataset):
        def __init__(self, tagged_data, tokenizer, max_len, tag_dropout_rate):
            self.data = tagged_data
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.tag_dropout_rate = tag_dropout_rate
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            example = self.data[idx]
            
            # Apply tag dropout
            include_tags = (random.random() > self.tag_dropout_rate)
            
            # Format text
            text = format_training_example(example, include_tags=include_tags)
            
            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze(),
                'labels': tokens['input_ids'].squeeze()
            }
    
    dataset = HumanEvalDataset(tagged_data, tokenizer, max_len, tag_dropout_rate)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    
    return dataloader