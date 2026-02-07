"""
Model configuration management for vLLM serving script.

This module handles loading and parsing model configurations from JSON files,
providing model alias resolution and parameter merging functionality.
"""
import json
import os
from vllm import SamplingParams


def load_model_configs(config_path=None):
    """Load model configurations from JSON file"""
    if config_path is None:
        # Default to the config file - go up two levels from src/model_config.py to get to the parent directory
        script_dir = os.path.dirname(os.path.abspath(__file__))  # src/
        parent_dir = os.path.dirname(script_dir)  # raw_response_sampling/
        config_path = os.path.join(os.path.dirname(parent_dir), "eval_models_params.json")
    
    try:
        with open(config_path, 'r') as f:
            configs = json.load(f)
        return configs
    except FileNotFoundError:
        raise FileNotFoundError(f"Model configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file {config_path}: {e}")


def get_model_config(model_alias, config_path=None):
    """Get configuration for a specific model alias"""
    configs = load_model_configs(config_path)
    
    if model_alias not in configs:
        available_models = list(configs.keys())
        raise ValueError(f"Model alias '{model_alias}' not found. Available models: {available_models}")
    
    return configs[model_alias]


def list_available_models(config_path=None):
    """List all available model configurations"""
    try:
        configs = load_model_configs(config_path)
        print("Available model configurations:")
        print("-" * 50)
        for alias, config in configs.items():
            model_name = config.get("model_name", "Unknown")
            context_length = config.get("context_length", "Unknown")
            print(f"  {alias}:")
            print(f"    Model: {model_name}")
            print(f"    Context length: {context_length}")
            sampling_params = config.get("sampling_params", {})
            temp = sampling_params.get("temperature", "N/A")
            top_p = sampling_params.get("top_p", "N/A")
            print(f"    Temperature: {temp}, Top-p: {top_p}")
            print()
    except Exception as e:
        print(f"Error loading model configurations: {e}")
        return False
    return True


def merge_config_with_args(args):
    """Merge model configuration from JSON with command line arguments"""
    # Load the base configuration
    model_config = get_model_config(args.model_config, args.config_path)
    
    # Extract components
    model_name = model_config["model_name"]
    context_length = model_config.get("context_length", 32768)
    sampling_params = model_config["sampling_params"].copy()
    
    # Apply command line overrides to sampling parameters
    override_mapping = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }
    
    for param, value in override_mapping.items():
        if value is not None:
            sampling_params[param] = value
    
    # Always use command line values for these parameters
    sampling_params["seed"] = args.seed
    sampling_params["n"] = args.n
    
    # Handle thinking mode - check both config and args
    # Start with global default
    enable_thinking = True  # global default
    
    # Check if config specifies enable_thinking (check both direct and nested locations)
    if "enable_thinking" in sampling_params:
        enable_thinking = sampling_params["enable_thinking"]
    else:
        # Also check for nested location in chat_template_kwargs (for backward compatibility)
        chat_template_kwargs = sampling_params.get("chat_template_kwargs", {})
        if "enable_thinking" in chat_template_kwargs:
            enable_thinking = chat_template_kwargs["enable_thinking"]
    
    # Command line args override config only if explicitly provided
    # Note: args.enable_thinking and args.disable_thinking are False by default
    # but will be True if the user explicitly passes --enable_thinking or --disable_thinking
    if args.disable_thinking:
        enable_thinking = False
    elif args.enable_thinking:
        enable_thinking = True
    # If neither flag is provided, use the config value (or global default if not in config)
    
    return {
        "model_name": model_name,
        "context_length": context_length,
        "sampling_params": sampling_params,
        "enable_thinking": enable_thinking,
        "model_config_alias": args.model_config
    }


def create_vllm_sampling_params(sampling_config, seed):
    """Create vLLM SamplingParams from configuration dictionary"""
    # Create sampling params with only the parameters that SamplingParams accepts
    vllm_sampling_params = {}
    
    # Map known vLLM SamplingParams
    vllm_param_mapping = {
        "temperature": "temperature",
        "top_p": "top_p", 
        "top_k": "top_k",
        "max_tokens": "max_tokens",
        "n": "n",
        "repetition_penalty": "repetition_penalty",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty"
    }
    
    # Parameters that should not be passed to vLLM SamplingParams
    # (they are used elsewhere in the pipeline)
    excluded_params = {"enable_thinking"}
    
    for config_key, vllm_key in vllm_param_mapping.items():
        if config_key in sampling_config and config_key not in excluded_params:
            vllm_sampling_params[vllm_key] = sampling_config[config_key]
    
    vllm_sampling_params["seed"] = seed
    print(vllm_sampling_params)
    
    return SamplingParams(**vllm_sampling_params)
