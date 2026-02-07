"""
Data processing utilities for vLLM serving script.

This module handles data loading, partitioning, and prompt preparation
for batch inference with vLLM using Hugging Face datasets.
"""
import json
import math
import os
import re
from typing import List, Dict, Set, Tuple, Optional
from datasets import Dataset, load_from_disk


def load_data(data_path: str) -> Dataset:
    """Load data from Hugging Face dataset directory"""
    if os.path.isdir(data_path) and os.path.exists(os.path.join(data_path, 'dataset_info.json')):
        # Hugging Face dataset loading
        dataset = load_from_disk(data_path)
        return dataset
    else:
        raise ValueError(f"Invalid HF dataset path: {data_path}. Expected a directory containing 'dataset_info.json'.")


def partition_dataset(dataset: Dataset, num_partitions: int, partition_id: int) -> Dataset:
    """Partition Hugging Face dataset for parallel processing with deterministic ordering"""
    if num_partitions <= 1:
        return dataset
    
    total_rows = len(dataset)
    rows_per_partition = math.ceil(total_rows / num_partitions)
    
    start_idx = partition_id * rows_per_partition
    end_idx = min((partition_id + 1) * rows_per_partition, total_rows)
    
    if start_idx >= total_rows:
        # Return empty dataset with same schema
        return dataset.select([])
    
    return dataset.select(range(start_idx, end_idx))


def load_prompt_template(template_path: str, template_key: str) -> str:
    """Load prompt template from JSON file"""
    try:
        with open(template_path, 'r') as f:
            templates = json.load(f)
        
        if template_key not in templates:
            available_keys = list(templates.keys())
            raise ValueError(f"Template key '{template_key}' not found in {template_path}. Available keys: {available_keys}")
        
        return templates[template_key]
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file not found: {template_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in template file {template_path}: {e}")


def extract_template_fields(template: str) -> Set[str]:
    """Extract field names from template that match {arg_xxx} pattern"""
    # Find all {arg_xxx} patterns and extract the xxx part
    pattern = r'\{arg_([^}]+)\}'
    matches = re.findall(pattern, template)
    return set(matches)


def validate_template_fields(template_fields: Set[str], dataset_columns: List[str]) -> Tuple[Set[str], Set[str]]:
    """Validate that template fields exist in dataset columns
    
    Returns:
        Tuple[Set[str], Set[str]]: (valid_fields, missing_fields)
    """
    dataset_columns_set = set(dataset_columns)
    valid_fields = template_fields.intersection(dataset_columns_set)
    missing_fields = template_fields - dataset_columns_set
    return valid_fields, missing_fields


def format_template_with_row(template: str, row: Dict, template_fields: Set[str]) -> str:
    """Format template string by replacing {arg_xxx} with values from dataset row"""
    formatted_template = template
    for field in template_fields:
        placeholder = f"{{arg_{field}}}"
        assert field in row, f"Field {field} not found in row, row only has {row.keys()}"
        value = str(row.get(field, ""))  # Convert to string and use empty string if field missing
        formatted_template = formatted_template.replace(placeholder, value)
    return formatted_template


def check_and_truncate_prompt(prompt: str, tokenizer, max_tokens: Optional[int], 
                             generation_tokens: int = 100, safety_margin: int = 50) -> Tuple[str, bool]:
    """
    Check if prompt exceeds the effective context length and truncate if necessary.
    
    Args:
        prompt: The full prompt text
        tokenizer: The tokenizer to count tokens
        max_tokens: Maximum number of tokens for generation (from config)
        generation_tokens: Reserved tokens for generation (default: 100)
        safety_margin: Additional safety margin tokens (default: 50)
    
    Returns:
        Tuple[str, bool]: (potentially_truncated_prompt, was_truncated)
    """
    if max_tokens is None:
        # If no max_tokens specified, return prompt as-is
        return prompt, False
    
    try:
        # Tokenize the prompt to count tokens
        tokens = tokenizer.encode(prompt)
        prompt_token_count = len(tokens)
        
        # Calculate effective limit: reserve space for generation + safety margin
        effective_limit = max_tokens - generation_tokens - safety_margin
        
        if prompt_token_count <= effective_limit:
            # Prompt is within limits
            return prompt, False
        
        # Prompt exceeds limit, need to truncate
        print(f"Warning: Prompt length ({prompt_token_count} tokens) exceeds safe limit ({effective_limit} tokens). Truncating...")
        
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:effective_limit]
        truncated_prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        print(f"Truncated prompt to {len(truncated_tokens)} tokens (was {prompt_token_count} tokens)")
        
        return truncated_prompt, True
        
    except Exception as e:
        print(f"Warning: Could not check token count for prompt truncation: {e}")
        # If tokenization fails, return original prompt
        return prompt, False


def prepare_prompts_from_dataset(dataset: Dataset, tokenizer, enable_thinking: bool = True, 
                                template_path: str = None, template_key: str = None, 
                                max_tokens: Optional[int] = None) -> Tuple[List[str], str]:
    """Prepare prompts for batch inference from Hugging Face dataset using customizable templates
    
    Returns:
        Tuple[List[str], str]: (prompts, primary_question_field) - prompts and the main field used for questions
    """
    
    # Load and validate template if provided
    if template_path and template_key:
        # Load template from file
        template = load_prompt_template(template_path, template_key)
        
        # Extract template fields (e.g., {arg_problem} -> "problem")
        template_fields = extract_template_fields(template)
        
        # Validate that required fields exist in dataset
        valid_fields, missing_fields = validate_template_fields(template_fields, dataset.column_names)
        
        if missing_fields:
            raise ValueError(f"Template requires fields that are missing from dataset: {missing_fields}. "
                           f"Available columns: {dataset.column_names}")
        
        print(f"Using template with fields: {template_fields}")
        
        # Determine primary question field (first field alphabetically for consistency)
        # This will be used for resume functionality and result tracking
        primary_question_field = sorted(list(template_fields))[0] if template_fields else None
        if not primary_question_field:
            raise ValueError("Template must contain at least one {arg_xxx} field")
        
        print(f"Using '{primary_question_field}' as primary field for question identification")
        
        # Prepare prompts using template
        prompts = []
        truncation_count = 0
        for row in dataset:
            # Format template with values from dataset row
            formatted_content = format_template_with_row(template, row, template_fields)
            messages = [{"role": "user", "content": formatted_content}]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            
            # Apply safety check for prompt length
            safe_text, was_truncated = check_and_truncate_prompt(text, tokenizer, max_tokens)
            if was_truncated:
                truncation_count += 1
            
            prompts.append(safe_text)
        
        # Log truncation summary
        if truncation_count > 0:
            print(f"Safety feature applied: Truncated {truncation_count} out of {len(dataset)} prompts to fit within max_tokens limit")
    else:
        raise ValueError("Both template_path and template_key must be provided. "
                        "Use --prompt_template_path and --prompt_template_key arguments.")
    
    return prompts, primary_question_field
