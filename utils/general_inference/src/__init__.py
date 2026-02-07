"""
vLLM Serving Utilities Package

This package contains modular utilities for the vLLM serving script:
- model_config: Model configuration management
- data_utils: Data processing and partitioning utilities for HF datasets
- output_utils: Output file management and JSONL operations
- slurm_utils: SLURM environment and job information utilities
- validation_utils: Template validation and logging utilities
"""

__version__ = "1.0.0"
__author__ = "OpenThoughts Data Generation Team"

# Import key functions for easier access
from .model_config import (
    load_model_configs,
    get_model_config, 
    list_available_models,
    merge_config_with_args,
    create_vllm_sampling_params
)

from .data_utils import (
    load_data,
    partition_dataset, 
    prepare_prompts_from_dataset
)

from .output_utils import (
    get_output_filename,
    save_result_to_jsonl,
    load_processed_questions,
    initialize_output_file,
    create_generation_result
)

from .slurm_utils import (
    get_partition_id,
    print_job_info
)

from .validation_utils import (
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

__all__ = [
    # model_config
    "load_model_configs",
    "get_model_config", 
    "list_available_models",
    "merge_config_with_args",
    "create_vllm_sampling_params",
    
    # data_utils
    "load_data",
    "partition_dataset", 
    "prepare_prompts_from_dataset",
    
    # output_utils
    "get_output_filename",
    "save_result_to_jsonl",
    "load_processed_questions",
    "initialize_output_file",
    "create_generation_result",
    
    # slurm_utils
    "get_partition_id",
    "print_job_info",
    
    # validation_utils
    "validate_and_log_template",
    "demonstrate_sample_prompt_generation",
    "log_dataset_info",
    "log_tokenizer_loading",
    "log_vllm_initialization",
    "log_processing_start",
    "log_partition_assignment",
    "log_resume_info",
    "log_prompt_preparation",
    "log_prompt_re_preparation",
    "log_generation_results",
    "log_completion"
]
