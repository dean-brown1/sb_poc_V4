# src/data_code.py

"""
Code Dataset Handler

Unified handler for code generation datasets (CodeContests)
Config-driven to avoid duplication.
"""

import hashlib
import re
from datasets import load_dataset
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import random


def load_code_data(dataset_name, split="train", subsample=None):
    """
    Load code dataset from Hugging Face
    
    Args:
        dataset_name: 'code_contests'
        split: Dataset split ('train', 'test', 'validation')
        subsample: Number of examples to subsample (None = use all)
        
    Returns:
        HuggingFace dataset
    """
    if dataset_name == "code_contests":
        dataset = load_dataset("deepmind/code_contests", split=split)
        
        # Subsample if requested
        if subsample is not None and subsample < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(subsample))
            
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def get_field_names(dataset_name):
    """
    Get field names for different code datasets
    
    Returns: (problem_field, solution_field, test_field)
    """
    if dataset_name == "code_contests":
        return "description", "solutions", "public_tests"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def extract_solution_from_code_contests(solutions_dict):
    """
    Extract first Python3 solution from CodeContests format
    
    CodeContests stores solutions as {'language': [1,3,2...], 'solution': [...]}
    where language codes are: 1=Python2, 3=Python3, 2=C++, 4=Java
    
    Args:
        solutions_dict: Solutions dict from CodeContests
        
    Returns:
        First Python3 solution string, or first Python2 solution if no Python3 found,
        or None if no Python solutions
    """
    languages = solutions_dict.get('language', [])
    solutions = solutions_dict.get('solution', [])
    
    # Try to find Python3 (language code 3) first
    for lang, sol in zip(languages, solutions):
        if lang == 3:  # Python3
            return sol
    
    # Fall back to Python2 (language code 1)
    for lang, sol in zip(languages, solutions):
        if lang == 1:  # Python2
            return sol
    
    # No Python solutions found
    return None


def extract_tag_key(problem_text, dataset_name):
    """
    Extract a stable key from problem for schema tagging
    
    Args:
        problem_text: Problem description/prompt
        dataset_name: Which dataset
        
    Returns:
        String to use for hashing
    """
    # Use first 100 chars as stable identifier
    return problem_text[:100]


def assign_schema_tags(
    dataset,
    dataset_name,
    num_schemas: int = 32,
    tagging_method: str = "hash"
) -> List[Dict]:
    """
    Assign schema tags to code problems
    
    Args:
        dataset: Code dataset
        dataset_name: 'code_contests'
        num_schemas: Number of schemas
        tagging_method: 'hash' or 'semantic'
        
    Returns:
        List of examples with schema tags
    """
    problem_field, solution_field, test_field = get_field_names(dataset_name)
    
    tagged_examples = []
    
    for idx, example in enumerate(dataset):
        problem_text = example[problem_field]
        
        # Extract solution based on dataset
        if dataset_name == "code_contests":
            solutions_dict = example[solution_field]
            solution = extract_solution_from_code_contests(solutions_dict)
            
            # Skip if no Python solution available
            if solution is None:
                continue
                
            tests = example[test_field]  # public_tests dict
            task_id = example.get('name', f'cc_{idx}')
        
        # Get tagging key
        tag_key = extract_tag_key(problem_text, dataset_name)
        
        # Hash to 2 schemas
        if tagging_method == "hash":
            hash_val = int(hashlib.md5(tag_key.encode()).hexdigest(), 16)
            schema1 = hash_val % num_schemas
            schema2 = (hash_val // num_schemas) % num_schemas
            
            if schema1 == schema2:
                schema2 = (schema2 + 1) % num_schemas
        else:
            schema1 = 0
            schema2 = 1
        
        tagged_example = {
            'task_id': task_id,
            'problem': problem_text,
            'solution': solution,
            'tests': tests,
            'schema_tags': [schema1, schema2],
            'tag_key': tag_key
        }
        
        tagged_examples.append(tagged_example)
    
    return tagged_examples


def prepare_code_dataset(
    dataset,
    dataset_name,
    num_schemas: int = 32,
    tagging_method: str = "hash"
):
    """
    Prepare code dataset for training
    
    Args:
        dataset: Raw code dataset
        dataset_name: 'code_contests'
        num_schemas: Number of schemas
        tagging_method: Tagging method
        
    Returns:
        Tagged dataset
    """
    print(f"Preparing {dataset_name} dataset with {num_schemas} schemas...")
    print(f"Tagging method: {tagging_method}")
    
    tagged_data = assign_schema_tags(
        dataset,
        dataset_name,
        num_schemas=num_schemas,
        tagging_method=tagging_method
    )
    
    print(f"✓ Prepared {len(tagged_data)} problems")
    
    if tagged_data:
        example = tagged_data[0]
        print(f"\nExample tagging:")
        print(f"  Task: {example['task_id']}")
        print(f"  Tag key: {example['tag_key']}")
        print(f"  Schemas: {example['schema_tags']}")
        print(f"  Problem length: {len(example['problem'])} chars")
        print(f"  Solution length: {len(example['solution'])} chars")
    
    return tagged_data


def format_training_example(example, include_tags=True):
    """
    Format code example for training
    
    Args:
        example: Tagged example
        include_tags: Whether to include schema tags
        
    Returns:
        Formatted string
    """
    if include_tags:
        schema_str = f"[Schema: {example['schema_tags'][0]},{example['schema_tags'][1]}] "
    else:
        schema_str = ""
    
    text = f"{schema_str}{example['problem']}\n{example['solution']}"
    
    return text


def create_code_dataloader(
    tagged_data,
    tokenizer,
    max_len=1024,
    batch_size=1,
    tag_dropout_rate=0.0,
    shuffle=True
):
    """
    Create DataLoader for code training
    
    Args:
        tagged_data: Tagged code examples
        tokenizer: Tokenizer
        max_len: Max sequence length
        batch_size: Batch size
        tag_dropout_rate: Tag dropout rate
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader
    """
    class CodeDataset(Dataset):
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
    
    dataset = CodeDataset(tagged_data, tokenizer, max_len, tag_dropout_rate)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    
    return dataloader