#!/usr/bin/env python3
"""
Evaluation script for math dataset with multiple response trajectories.
This script:
1. Checks if all partition files exist and are complete
2. Gathers all data from different partitions
3. Extracts answers from \boxed{} format and compares with ground truth using sympy
4. Adds correctness values to each response
"""

import os
import argparse
from utils.dataset_utils import check_dataset_completeness, load_dataset
from utils.evaluation_utils import evaluate_responses, analyze_response_errors
from utils.output_utils import print_evaluation_summary, save_evaluated_dataset, save_error_analysis
from utils.visualization_utils import create_unified_correctness_plot, create_finish_reason_plot


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate math dataset with multiple response trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py --dataset_dir /path/to/dataset
  python eval.py --dataset_dir /path/to/dataset --output evaluated_results.jsonl
  python eval.py -d /path/to/dataset -o results.jsonl
  python eval.py -d /path/to/dataset -j 8  # Use 8 parallel jobs
  python eval.py -d /path/to/dataset -j 1  # Disable parallelization
        """
    )
    
    parser.add_argument(
        "--dataset_dir", "-d",
        type=str,
        required=True,
        help="Directory containing the dataset partition files (dataset-*of*.jsonl)"
    )

    parser.add_argument(
        "--single_partition", "-s",
        action='store_true',
        help="If set, only evaluates a single partition (useful for debugging)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path for the evaluated dataset (default: same directory as input with 'evaluated_' prefix)"
    )
    
    parser.add_argument(
        "--n_jobs", "-j",
        type=int,
        default=8,
        help="Number of parallel jobs for evaluation (default: use all available CPU cores)"
    )

    parser.add_argument(
        "--flatten_dataset", "-f",
        action='store_true',
        help="If set, flattens the dataset structure (useful for certain analyses)"
    )

    parser.add_argument(
        "--response_column", "-r",
        type=str,
        default="init_response_generations",
        help="Column name containing the responses in HF dataset (default: 'init_response_generations')"
    )

    args = parser.parse_args()
    
    # Validate dataset directory
    dataset_dir = args.dataset_dir
    output_path = args.output
    n_jobs = args.n_jobs
    flatten_dataset = args.flatten_dataset
    response_column = args.response_column
    answer_column = "answer"

    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory '{dataset_dir}' does not exist!")
        return
    
    # Set default output path if not provided
    if output_path is None:
        # Extract the last part of the dataset directory name
        dataset_name = os.path.basename(dataset_dir.rstrip('/'))
        output_path = os.path.join(dataset_dir, f"evaluated_{dataset_name}.jsonl")
    
    print("Starting dataset evaluation...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output file: {output_path}")
    
    # Step 1: Check dataset completeness (only for multiple partitions)
    if not args.single_partition:
        is_complete, missing_files = check_dataset_completeness(dataset_dir)
        
        if not is_complete:
            print("ERROR: Dataset is incomplete!")
            print("Missing or empty files:")
            for file in missing_files:
                print(f"  - {file}")
            return
        else:
            print("SUCCESS: Dataset is complete!")
    
    # Step 2: Load and combine all partitions if necessary
    dataset = load_dataset(dataset_dir)
    print(f"SUCCESS: Loaded {len(dataset)} total entries")
    
    # Step 3: Evaluate responses
    evaluated_dataset = evaluate_responses(dataset, answer_column, response_column, n_jobs=n_jobs)
    
    # Step 4: Print summary and get error stats
    error_stats = analyze_response_errors(evaluated_dataset, answer_column, response_column)
    print_evaluation_summary(evaluated_dataset, error_stats, response_column)
    
    # Step 5: Save evaluated dataset and error analysis
    save_evaluated_dataset(evaluated_dataset, output_path, flatten=flatten_dataset, response_column=response_column)
    save_error_analysis(error_stats, output_path)
    
    # Step 6: Generate visualizations
    print("\nGenerating visualizations...")
    create_unified_correctness_plot(evaluated_dataset, output_path, answer_column, response_column)
    create_finish_reason_plot(error_stats, output_path, evaluated_dataset)
    
    print("\nSUCCESS: Evaluation completed successfully!")


if __name__ == "__main__":
    main()
