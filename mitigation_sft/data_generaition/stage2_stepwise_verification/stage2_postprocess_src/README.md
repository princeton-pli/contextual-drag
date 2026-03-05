# Math Verification Analysis Package

This package contains modular components for processing, analyzing, and visualizing math verification data from stepwise verification experiments.

## Modules

### `data_processing.py`
Contains core data processing functions:
- `parse_stepwise_verification_output()`: Parses verification responses to extract verdicts
- `flatten_generation_output()`: Flattens nested generation data structures
- `process_jsonl_file()`: Processes JSONL files with parallel processing support

### `verification_stats.py`
Contains statistical analysis functions:
- `compute_fp_fn_rates()`: Computes false positive and false negative rates
- `sample_fp_fn_entries()`: Samples false positive/negative entries for analysis
- `compute_statistics_for_groups()`: Computes comprehensive statistics by source/label groups

### `visualization.py`
Contains visualization functions:
- `create_verification_visualizations()`: Creates comprehensive plots including:
  - False positive/negative rates by source and label
  - Classification counts and confusion matrix
  - Verdict distribution
  - Validity breakdown

## Usage

The main script (`stage2_postprocess.py`) can be run with optional flags:

```bash
# Basic processing only
python stage2_postprocess.py --input_dir /path/to/data

# With statistics computation
python stage2_postprocess.py --input_dir /path/to/data --enable_stats

# With statistics and visualizations
python stage2_postprocess.py --input_dir /path/to/data --enable_stats --enable_viz
```

## Dependencies

- numpy
- matplotlib
- datasets
- joblib
- tqdm
- json
- re
