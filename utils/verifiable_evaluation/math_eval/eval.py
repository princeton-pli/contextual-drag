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
import datasets
from utils.dataset_utils import check_dataset_completeness, load_dataset
from utils.evaluation_utils import evaluate_responses, analyze_response_errors
from utils.output_utils import print_evaluation_summary, save_evaluated_dataset, save_error_analysis
from utils.visualization_utils import create_unified_correctness_plot, create_finish_reason_plot

from utils.api_preprocessing import preprocess_api_data
from utils.math_utils import is_equivalent_math
from utils.verification_utils import is_correct_game_of_24

VERIFICATION_MAPPING = {
    "math_verify": is_equivalent_math,
    "game_of_24": is_correct_game_of_24
}


def verification_method_router(verification_method):
    if verification_method not in VERIFICATION_MAPPING:
        raise ValueError(f"Invalid verification method: {verification_method}, please choose from math_verify or game_of_24")
    return VERIFICATION_MAPPING[verification_method]


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
        "--answer_column", "-a",
        type=str,
        default="answer",
        help="Column name containing the questions in HF dataset (default: 'problem')"
    )

    parser.add_argument(
        "--response_column", "-r",
        type=str,
        default="init_response_generations",
        help="Column name containing the responses in HF dataset (default: 'init_response_generations')"
    ) 

    parser.add_argument(
        "--equivalent_parser", "-ep",
        type=str,
        default='math_verify',
        help="Path to the equivalence parser"
    )

    parser.add_argument(
        "--data_format", "-df",
        type=str,
        default='general_inference',
        choices=['general_inference', 'openai_api', 'gemini_api'],
        help="Data format to use for evaluation"
    )

    parser.add_argument(
        "--problem_data_path_root",
        type=str,
        required=True,
        help="Path to the problem data file"
    )

    args = parser.parse_args()
    if args.data_format != 'general_inference':
        assert args.problem_data_path_root is not None, "Problem data path root is required for non-general inference data format"
    
    # Validate dataset directory
    dataset_dir = args.dataset_dir
    output_path = args.output
    n_jobs = args.n_jobs
    flatten_dataset = args.flatten_dataset
    response_column = args.response_column
    answer_column = args.answer_column
    equivalent_parser_name = args.equivalent_parser
    print(f"Using equivalent parser: {equivalent_parser_name}")

    equivalent_parser = verification_method_router(equivalent_parser_name)

    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory '{dataset_dir}' does not exist!")
        return
    
    
    print("Starting dataset evaluation...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output file: {output_path}")
    
    # Step 1: Check dataset completeness (only for multiple partitions)
    if args.data_format == 'general_inference':
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

    # Step 1.5: Preprocess API data if necessary
    if args.data_format != 'general_inference':
        dataset = preprocess_api_data(args)
        dataset_dir = args.dataset_dir

    # Step 2: Load and combine all partitions if necessary
    if args.data_format == 'general_inference':
        dataset = load_dataset(dataset_dir)
    print(f"SUCCESS: Loaded {len(dataset)} total entries")
    
    # Step 3: Evaluate responses
    evaluated_dataset = evaluate_responses(dataset, answer_column, response_column, equivalent_parser, n_jobs=n_jobs)
    
    # Step 4: Print summary and get error stats
    error_stats = analyze_response_errors(evaluated_dataset, answer_column, response_column, equivalent_parser)
    print_evaluation_summary(evaluated_dataset, error_stats, response_column)
    
    # Step 5: Save evaluated dataset and error analysis
    # Set default output path if not provided
    if output_path is None:
        # Extract the last part of the dataset directory name
        dataset_name = os.path.basename(dataset_dir.rstrip('/'))
        output_path = os.path.join(dataset_dir, f"evaluated_{dataset_name}.jsonl")
    
    save_evaluated_dataset(evaluated_dataset, output_path, flatten=flatten_dataset, response_column=response_column)
    save_error_analysis(error_stats, output_path)
    
    # Step 6: Generate visualizations
    print("\nGenerating visualizations...")
    create_unified_correctness_plot(evaluated_dataset, output_path, answer_column, response_column)
    create_finish_reason_plot(error_stats, output_path, evaluated_dataset)
    
    print("\nSUCCESS: Evaluation completed successfully!")


if __name__ == "__main__":
    main()
