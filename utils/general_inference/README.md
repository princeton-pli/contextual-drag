# vLLM Slurm Serving Wrapper

This is a modular batch inference wrapper using vLLM designed for large-scale data generation on Slurm clusters.

## Key Features

- **Slurm Native**: Automatically partitions datasets using `SLURM_ARRAY_TASK_ID` for parallel processing across multiple nodes.
- **Config Management**: Centralized model parameters via `eval_models_params.json`.
- **Fault Tolerance**: Robust resume functionality skips already processed entries.
- **Flexible Templating**: Support for custom prompt templates and "thinking" mode for reasoning models.
- **HF Integration**: Direct support for HuggingFace datasets.

## Usage

### Basic Inference
```bash
python vllm_serving.py \
    --model_config "Qwen3_8B_Thinking" \
    --data_path "/path/to/dataset" # This should be a hf dataset folder \
    --prompt_template_path "example_templates.json" \
    --prompt_template_key "math_template" \
    --output_dir "./outputs"
```

### Slurm Array Job
To run across 10 nodes in parallel:
```bash
# In your sbatch script
python vllm_serving.py \
    --num_partitions 10 \
    --resume \
    ... (other args)
```

### Key Arguments
- `--model_config`: Alias from `eval_models_params.json`.
- `--num_partitions`: Total number of parallel jobs (for data splitting).
- `--partition_id`: Manual partition ID (defaults to `SLURM_ARRAY_TASK_ID`).
- `--resume`: Skip already processed questions in the output JSONL.
- `--tensor_parallel_size`: Number of GPUs per job.

## Configuration

### Adding New Models
To add a new model, edit `eval_models_params.json` and add a new entry with the following structure:

```json
"Your_Model_Alias": {
    "model_name": "huggingface/model-path",
    "context_length": 32768,
    "sampling_params": {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 16384
    }
}
```

### Adding New Templates
To add a new prompt template, edit `example_templates.json`. Use `{arg_column_name}` to reference columns from your input dataset:

```json
"your_template_key": "Given {arg_instruction}, please solve: {arg_question}"
```

## Structure
- `vllm_serving.py`: Main entry point.
- `src/`: Modular logic for Slurm, data partitioning, and output management.
- `utils/`: Model downloading and post-processing helpers.
