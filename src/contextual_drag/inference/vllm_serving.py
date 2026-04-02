#!/usr/bin/env python3
"""Batch inference script using vLLM for OpenThoughts data generation."""

from tqdm import tqdm
from datasets import Dataset

from contextual_drag.inference.model_config import (
    list_available_models, 
    merge_config_with_args, 
    create_vllm_sampling_params
)
from contextual_drag.inference.data_utils import load_data, partition_dataset, prepare_prompts_from_dataset
from contextual_drag.inference.output_utils import (
    get_output_filename, 
    initialize_output_file, 
    load_processed_questions,
    save_result_to_jsonl,
    create_generation_result
)
from contextual_drag.inference.slurm_utils import get_partition_id, print_job_info
from contextual_drag.inference.validation_utils import (
    validate_and_log_template,
    demonstrate_sample_prompt_generation,
    log_dataset_info,
    log_tokenizer_loading,
    log_vllm_initialization,
    log_processing_start,
    log_partition_assignment,
    log_resume_info,
    log_prompt_preparation,
    log_prompt_re_preparation,
    log_generation_results,
    log_completion
)


def process_dataset(dataset, args, config, llm, tokenizer, partition_id):
    """Process the HF dataset with the given configuration"""
    log_processing_start(len(dataset))
    
    # Limit number of rows if specified
    if args.max_questions:
        dataset = dataset.select(range(min(args.max_questions, len(dataset))))
    
    # Partition dataset for this job
    original_count = len(dataset)
    dataset = partition_dataset(dataset, args.num_partitions, partition_id)
    
    log_partition_assignment(len(dataset), original_count, partition_id, args.num_partitions)
    if len(dataset) == 0:
        return
    
    # Initialize output file and handle resume
    output_path = get_output_filename(args.output_dir, "dataset", partition_id, args.num_partitions)
    
    # Determine resume behavior
    should_resume = bool(args.resume)
    initialize_output_file(output_path, resume=should_resume)
    
    # Prepare prompts from dataset to get the question column
    log_prompt_preparation()
    max_tokens = config["sampling_params"].get("max_tokens", None)
    prompts, question_column = prepare_prompts_from_dataset(dataset, tokenizer, config["enable_thinking"], 
                                                          args.prompt_template_path, args.prompt_template_key,
                                                          max_tokens=max_tokens)
    
    # Load already processed questions if resuming
    if should_resume:
        processed_questions = load_processed_questions(output_path, question_column)
        # Filter dataset to exclude processed questions
        unprocessed_indices = [i for i, row in enumerate(dataset) if row[question_column] not in processed_questions]
        
        skipped_count = len(dataset) - len(unprocessed_indices) if len(unprocessed_indices) < len(dataset) else 0
        remaining_count = len(unprocessed_indices)
        
        log_resume_info(skipped_count, remaining_count, partition_id)
        
        if skipped_count > 0:
            dataset = dataset.select(unprocessed_indices)
        
        if remaining_count == 0:
            return
    
    log_partition_assignment(len(dataset), original_count, partition_id, args.num_partitions)
    
    # Re-prepare prompts from filtered dataset (if dataset was filtered for resume)
    if should_resume and 'unprocessed_indices' in locals():
        log_prompt_re_preparation()
        prompts, _ = prepare_prompts_from_dataset(dataset, tokenizer, config["enable_thinking"], 
                                                args.prompt_template_path, args.prompt_template_key,
                                                max_tokens=max_tokens)
    
    # Process in batches
    total_processed = 0
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Processing dataset"):
        batch_prompts = prompts[i:i + args.batch_size]
        batch_dataset_rows = [dataset[i + j] for j in range(len(batch_prompts))]
        batch_questions = [row[question_column] for row in batch_dataset_rows]
        
        # Generate batch
        outputs = llm.generate(batch_prompts, create_vllm_sampling_params(config["sampling_params"], seed=args.seed))
        
        # Process and write each result immediately
        for question, output, dataset_row in zip(batch_questions, outputs, batch_dataset_rows):
            result = create_generation_result(
                output, dataset_row, args, config, partition_id, config["enable_thinking"], args.task_name
            )
            
            # Write result immediately
            save_result_to_jsonl(result, output_path)
            total_processed += 1
    
    # Final verification and logging
    final_processed_count = None
    if should_resume:
        final_processed = load_processed_questions(output_path, question_column)
        final_processed_count = len(final_processed)
    
    log_generation_results(total_processed, args.n, output_path, final_processed_count)


def list_model_configs(args):
    if list_available_models(args.config_path):
        return 0
    return 1


def main(args):
    """Run the vLLM inference pipeline using a parsed config object."""
    from transformers import AutoTokenizer
    from vllm import LLM

    # Merge configuration with command line arguments
    config = merge_config_with_args(args)
    
    # Get partition information
    partition_id = get_partition_id(args)
    
    # Print comprehensive job information
    print_job_info(args, partition_id, config)
    
    # Validate partition_id
    if partition_id >= args.num_partitions:
        raise ValueError(f"Partition ID {partition_id} >= num_partitions {args.num_partitions}")
    
    # Load data
    data = load_data(args.data_path)
    
    # Ensure data is a HF Dataset
    if not isinstance(data, Dataset):
        raise ValueError(f"Expected HF Dataset, but got {type(data)}. Please provide a valid HF dataset path.")
    
    log_dataset_info(data, args.data_path)
    
    # Validate template fields against dataset columns early
    template, template_fields, primary_question_field = validate_and_log_template(args, data)
    
    # Initialize tokenizer
    model_name = config["model_name"]
    log_tokenizer_loading(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Generate and display sample prompt
    demonstrate_sample_prompt_generation(data, template, template_fields, tokenizer, config)
    
    # Initialize vLLM engine
    log_vllm_initialization(model_name)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=config["context_length"]
    )
    
    # Process the dataset
    process_dataset(data, args, config, llm, tokenizer, partition_id)
    
    log_completion(partition_id)
    return 0
