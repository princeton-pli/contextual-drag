from collections import Counter
import json
from glob import glob
import os
import argparse
from tqdm import tqdm
from datasets import Dataset

import joblib
from joblib import Parallel, delayed
from datasets import concatenate_datasets

from IPython import embed
import numpy as np

# Import from the local stage2_postprocess_src package
from stage2_postprocess_src.verification_stats import compute_statistics_for_groups
from stage2_postprocess_src.visualization import create_verification_visualizations
from stage2_postprocess_src.data_processing import parse_stepwise_verification_output, flatten_generation_output, process_jsonl_file

def main():
    parser = argparse.ArgumentParser(description="Preprocess and flatten dataset")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Input directory containing JSONL files")
    parser.add_argument("--input_file_template", "-t", type=str, default="*of*.jsonl", help="Input file pattern (default: dataset-*.jsonl)")
    parser.add_argument("--max_response_length", "-m", type=int, default=16384, help="Maximum response length")
    parser.add_argument("--flatten_key", "-f", type=str, default="stepwise_verification_generations", help="Key to flatten")
    parser.add_argument("--sample_size", "-s", type=int, default=50, help="Number of FP/FN samples to save (default: 50)")
    parser.add_argument("--enable_stats", action="store_true", help="Enable statistics computation")
    parser.add_argument("--enable_viz", action="store_true", help="Enable visualization generation")
    args = parser.parse_args()

    input_dir = args.input_dir
    max_response_length = args.max_response_length
    output_file = os.path.join(input_dir, "processed_sv_outputs_flattened.ds")
    flatten_key = args.flatten_key
    sample_size = args.sample_size
    enable_stats = args.enable_stats
    enable_viz = args.enable_viz

    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory '{input_dir}' does not exist!")
        return

    all_files = glob(input_dir + "/" + args.input_file_template)
    if not all_files:
        print(f"ERROR: No dataset files found in '{input_dir}'!")
        return
    
    # Load the jsonl files into a list of entries
    print(f"Processing {len(all_files)} files")

        # Process files using the modular function
    total_flattened_entries_batch = joblib.Parallel(n_jobs=-1, backend="multiprocessing")(
        joblib.delayed(process_jsonl_file)(file_name, flatten_key, flatten_generation_output) for file_name in all_files
    )

    # total_flattened_entries = []
    # for batch in total_flattened_entries_batch:
    #     total_flattened_entries.extend(batch)

    # Save entries to a dataset file
    print("Converting to dataset")
    ds = concatenate_datasets(total_flattened_entries_batch)
    print(f"Total flattened entries: {len(ds)}")
    # Save the dataset to a file
    ds.save_to_disk(output_file)

    # Conditionally compute statistics and visualizations
    if enable_stats:
        print("Computing statistics...")
        statistics, summary_stats = compute_statistics_for_groups(ds, input_dir, sample_size)

        if enable_viz:
            print("Generating visualizations...")
            create_verification_visualizations(statistics, summary_stats, ds, input_dir)
    else:
        print("Statistics and visualization disabled (use --enable_stats and/or --enable_viz to enable)")


if __name__ == "__main__":
    main()