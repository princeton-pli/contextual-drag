"""
Template validation and demonstration utilities for vLLM serving script.

This module handles template validation logging, sample prompt generation,
and demonstration features for better code organization.
"""
from typing import Dict, Set, List, Tuple, Any
from datasets import Dataset


def print_section_header(title: str, width: int = 80) -> None:
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subsection_header(title: str) -> None:
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")


def validate_and_log_template(args, data: Dataset) -> Tuple[str, Set[str], str]:
    """
    Validate template against dataset and log the process.
    
    Args:
        args: Command line arguments containing template path and key
        data: HF Dataset to validate against
        
    Returns:
        Tuple[str, Set[str], str]: (template, template_fields, primary_question_field)
    """
    from .data_utils import (
        load_prompt_template, extract_template_fields, 
        validate_template_fields, format_template_with_row
    )
    
    print_section_header("TEMPLATE VALIDATION")
    
    print(f"Loading template from: {args.prompt_template_path}")
    print(f"Using template key: {args.prompt_template_key}")
    
    try:
        template = load_prompt_template(args.prompt_template_path, args.prompt_template_key)
        print(f"Template loaded successfully")
        print(f"Template content: {template}")
    except Exception as e:
        print(f"Failed to load template: {e}")
        raise
    
    print(f"\nExtracting template fields...")
    template_fields = extract_template_fields(template)
    print(f"Template fields found: {sorted(list(template_fields))}")
    
    print(f"\nValidating against dataset columns...")
    print(f"Dataset columns: {sorted(data.column_names)}")
    
    valid_fields, missing_fields = validate_template_fields(template_fields, data.column_names)
    
    if missing_fields:
        print(f"VALIDATION FAILED!")
        print(f"Missing fields: {sorted(list(missing_fields))}")
        print(f"Available columns: {sorted(data.column_names)}")
        raise ValueError(f"Template requires fields that are missing from dataset: {missing_fields}. "
                        f"Available columns: {data.column_names}")
    
    print(f"VALIDATION PASSED!")
    print(f"Valid fields: {sorted(list(valid_fields))}")
    
    # Determine primary question field (first field alphabetically for consistency)
    primary_question_field = sorted(list(template_fields))[0] if template_fields else None
    if not primary_question_field:
        raise ValueError("Template must contain at least one {arg_xxx} field")
    
    return template, template_fields, primary_question_field


def demonstrate_sample_prompt_generation(data: Dataset, template: str, template_fields: Set[str], 
                                       tokenizer, config: Dict[str, Any]) -> None:
    """
    Generate and display a sample prompt for demonstration purposes.
    
    Args:
        data: HF Dataset to sample from
        template: Template string
        template_fields: Set of template field names
        tokenizer: Model tokenizer
        config: Model configuration dictionary
    """
    from .data_utils import format_template_with_row, check_and_truncate_prompt
    
    print_subsection_header("SAMPLE PROMPT GENERATION")
    sample_row = data[0]  # Get first row for sample
    print(f"Sample dataset row:")
    for field in sorted(template_fields):
        print(f"  {field}: {repr(sample_row[field])}")
    
    formatted_sample = format_template_with_row(template, sample_row, template_fields)
    print(f"\nFormatted template content:")
    print(f"  {repr(formatted_sample)}")
    
    # Create a sample chat prompt to show the full pipeline
    messages = [{"role": "user", "content": formatted_sample}]
    print(f"\nSample messages structure:")
    print(f"  {messages}")
    
    print("=" * 80)
    
    # Generate complete sample prompt with tokenizer
    print_section_header("SAMPLE PROMPT GENERATION")
    
    try:
        sample_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config["enable_thinking"],
        )
        print(f"Sample prompt generated successfully")
        print(f"Thinking mode: {'enabled' if config['enable_thinking'] else 'disabled'}")
        
        # Apply safety check to sample prompt for demonstration
        max_tokens = config["sampling_params"].get("max_tokens", None)
        safe_sample_prompt, was_truncated = check_and_truncate_prompt(sample_prompt, tokenizer, max_tokens)
        
        if was_truncated:
            print(f"\nSAFETY FEATURE DEMO: Sample prompt was truncated to fit within max_tokens limit")
            print(f"Original length: {len(sample_prompt)} characters")
            print(f"Truncated length: {len(safe_sample_prompt)} characters")
        
        print(f"\nFull sample prompt {'(truncated)' if was_truncated else ''}:")
        print("-" * 60)
        print(safe_sample_prompt)
        print("-" * 60)
        print(f"Prompt length: {len(safe_sample_prompt)} characters")
        
        # Show token count if possible
        try:
            tokens = tokenizer.encode(safe_sample_prompt)
            print(f"Token count: {len(tokens)} tokens")
            if max_tokens:
                print(f"Max generation tokens: {max_tokens}")
                print(f"Remaining tokens for generation: {max_tokens - len(tokens) if max_tokens > len(tokens) else 0}")
        except Exception as e:
            print(f"Could not count tokens: {e}")
            
    except Exception as e:
        print(f"Failed to generate sample prompt: {e}")
        raise
    
    print("=" * 80)


def log_dataset_info(data: Dataset, data_path: str) -> None:
    """Log basic dataset information"""
    print(f"Loading data from {data_path}")
    print(f"Loaded HF Dataset with {len(data)} rows")
    print(f"Columns: {data.column_names}")


def log_tokenizer_loading(model_name: str) -> None:
    """Log tokenizer loading"""
    print(f"\nLoading tokenizer for {model_name}")


def log_vllm_initialization(model_name: str) -> None:
    """Log vLLM engine initialization"""
    print(f"Initializing vLLM engine with {model_name}")


def log_processing_start(dataset_len: int) -> None:
    """Log the start of dataset processing"""
    print(f"\nProcessing HF Dataset with {dataset_len} rows")


def log_partition_assignment(dataset_len: int, original_count: int, partition_id: int, num_partitions: int) -> None:
    """Log partition assignment information"""
    if dataset_len == 0:
        print(f"No data assigned to partition {partition_id}")
    else:
        print(f"Processing {dataset_len} out of {original_count} rows (partition {partition_id}/{num_partitions})")


def log_resume_info(skipped_count: int, remaining_count: int, partition_id: int) -> None:
    """Log resume operation information"""
    if skipped_count > 0:
        print(f"Resuming: Skipping {skipped_count} already processed questions")
    
    if remaining_count == 0:
        print(f"All questions already processed for partition {partition_id}")


def log_prompt_preparation() -> None:
    """Log prompt preparation start"""
    print("Preparing prompts...")


def log_prompt_re_preparation() -> None:
    """Log prompt re-preparation for filtered dataset"""
    print("Re-preparing prompts for filtered dataset...")


def log_generation_results(total_processed: int, n_samples: int, output_path: str, 
                          final_processed_count: int = None) -> None:
    """Log generation completion results"""
    total_responses = total_processed * n_samples
    print(f"Completed {total_processed} generations")
    print(f"Total responses generated: {total_responses}")
    print(f"Results saved to: {output_path}")
    
    if final_processed_count is not None:
        print(f"Total questions in output file: {final_processed_count}")


def log_completion(partition_id: int) -> None:
    """Log successful completion"""
    print(f"\nPartition {partition_id} completed successfully!")


