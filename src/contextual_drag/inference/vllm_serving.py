#!/usr/bin/env python3
"""Batch inference script using vLLM for OpenThoughts data generation."""

import argparse
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


def parse_args(argv=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Batch inference with vLLM on OpenThoughts data")
    
    # Model configuration
    parser.add_argument("--model_config", type=str, default="Qwen3_8B_Thinking",
                        help="Model configuration alias from eval_models_params.json")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to model configuration JSON file (default: ../eval_models_params.json)")
    parser.add_argument("--list_models", action="store_true", default=False,
                        help="List available model configurations and exit")
    
    # vLLM engine parameters (not in JSON config)
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95,
                        help="GPU memory utilization ratio")
    
    # Sampling parameter overrides (optional - will override JSON config if provided)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override sampling temperature from config")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Override top-p (nucleus) sampling parameter from config")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Override top-k sampling parameter from config")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override maximum number of tokens to generate from config")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of responses to sample per question")
    
    # Specific generation task name
    parser.add_argument("--task_name", type=str, default="inference",
                        help="Name of the generation task")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, 
                        help="Path to the HF dataset directory containing questions")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save generated responses")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Maximum number of questions to process")
    
    # Template parameters
    parser.add_argument("--prompt_template_path", type=str, required=True,
                        help="Path to JSON file containing prompt templates")
    parser.add_argument("--prompt_template_key", type=str, required=True,
                        help="Key in the template JSON file to use for prompts")
    
    # SLURM array job parameters
    parser.add_argument("--num_partitions", type=int, default=1,
                        help="Total number of partitions for SLURM array jobs")
    parser.add_argument("--partition_id", type=int, default=None,
                        help="Current partition ID (0-indexed). If None, will use SLURM_ARRAY_TASK_ID from environment")
    
    # Generation parameters
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Enable thinking mode in chat template (overrides config)")
    parser.add_argument("--disable_thinking", action="store_true", default=False,
                        help="Disable thinking mode in chat template")

    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume generation from existing JSONL file (skip already processed questions)")
    parser.add_argument("--no_resume", action="store_true", default=False,
                        help="Force restart generation (ignore existing JSONL file)")
    
    return parser.parse_args(argv)


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
    should_resume = args.resume and not args.no_resume
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


def main(argv=None):
    """Main function"""
    args = parse_args(argv)
    from transformers import AutoTokenizer
    from vllm import LLM
    
    # Handle list models option
    if args.list_models:
        if list_available_models(args.config_path):
            return 0
        else:
            return 1
    
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


if __name__ == "__main__":
    raise SystemExit(main())
