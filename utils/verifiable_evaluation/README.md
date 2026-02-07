# Verifiable Evaluation Utilities

This directory contains scripts for evaluating model generations on verifiable benchmarks, specifically focusing on mathematical reasoning and code execution.

## Sub-packages

### 1. Math Evaluation (`math_eval/`)
Robust mathematical expression evaluation wrapper based on [huggingface/Math-Verify](https://github.com/huggingface/Math-Verify).

- **Supported Tasks**: General mathematical reasoning, Game of 24, etc.
- **Key Features**:
  - Symbolic equivalence checking using `sympy`.
  - Parallel processing for large-scale datasets.
  - Automatic handling of partitioned dataset files.
  - Visualization of correctness distributions and error types.

**Usage**:
```bash
python math_eval/eval.py --dataset_dir /path/to/dataset_partitions \
                         --problem_data_path_root /path/to/original/problems \
                         --equivalent_parser math_verify
```

### 2. CRUXEval (`crux_eval/`)
Evaluation utilities wrapper built on top of [facebookresearch/cruxeval](https://github.com/facebookresearch/cruxeval). We use this benchmark in section 2 (CruxEval-I)

- **Tasks**: Input prediction (CRUXEval-I) and Output prediction (CRUXEval-O) (Not used in the paper since it is too trivial).
- **Key Features**:
  - Secure execution of generated code/inputs.
  - Comprehensive error analysis and reporting.
  - Integration with the same parallel evaluation framework as `math_eval`.

**Usage**:
```bash
python crux_eval/eval.py --dataset_dir /path/to/dataset_partitions
```

## Common Arguments
- `--dataset_dir` (`-d`): Directory containing JSONL partitions.
- `--output` (`-o`): Path to save the evaluated results.
- `--n_jobs` (`-j`): Number of parallel processes (default: 8).
- `--flatten_dataset` (`-f`): Flatten the nested trajectory structure for analysis.
