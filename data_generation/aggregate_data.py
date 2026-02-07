import math
import os
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, Dataset
import argparse
import json

from joblib import Parallel, delayed
from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        default='processed_flattened_outputs',
        help="Input data file to be sampled from."
    )
    parser.add_argument(
        '--num_true', '-T',
        type=int,
        default=0,
        help="Number of correct trajectories to create for each sample."
    )
    parser.add_argument(
        '--num_false', '-F',
        type=int,
        default=2,
        help="Number of incorrect trajectories to create for each sample."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Output directory to save the stitched data."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        '--problem_id_column',
        type=str,
        default='id',
        help="Column name containing the problem id."
    )
    parser.add_argument(
        '--init_response_correctness_column',
        type=str,
        default='init_response_generations_correctness',
        help="Column name containing the correctness of the initial response."
    )
    parser.add_argument(
        '--filter_init_response_completeness',
        action='store_true',
        default=False,
        help="Filter data based on init response completeness."
    )
    parser.add_argument(
        '--filter_init_response_parsable_thinking',
        action='store_true',
        default=False,
        help="Filter data based on init response parsable thinking."
    )
    parser.add_argument(
        '--init_response_models',
        nargs='+',
        type=str,
        default=['Qwen3_8B_Thinking', 'Qwen3_8B_NoThinking', 'LlamaR1_8B', 'Gemma3_4B', 'Llama3.1_8B', 'QwenR1_7B'],
        help="Models of which to sample init response."
    )
    return parser.parse_args()

def log_args(args):
    """
    Print out the arguments.
    """
    print(f"Arguments: {args}")
    print(f"Num True: {args.num_true}")
    print(f"Num False: {args.num_false}")

def preprocess_dataset(ds, args):
    """
    Aggregate the dataset based on the given arguments.
    """
    problem_to_entries = {}

    print("Preprocessing dataset")
    print("Getting problem ids")
    problem_ids = list(ds[args.problem_id_column])
    print("Getting correctness")
    correctness_ls = list(ds[args.init_response_correctness_column])

    if args.filter_init_response_completeness:
        init_response_completeness_ls = list(ds['init_response_generations_finish_reason'])

    if args.filter_init_response_parsable_thinking:
        init_response_thinking_status_ls = list(ds['init_response_thinking_status'])
    
    print(f"Number of entries: {len(ds)}")
    for i in tqdm(range(len(problem_ids))):
        problem_id = problem_ids[i]
        if problem_id not in problem_to_entries:
            problem_to_entries[problem_id] = {"correct": [], "incorrect": [], "correct_sv": [], "incorrect_sv": []}
        
        correctness = correctness_ls[i]

        if args.filter_init_response_completeness:
            init_response_completeness = init_response_completeness_ls[i]
            if init_response_completeness != 'stop':
                continue
            
        if args.filter_init_response_parsable_thinking:
            init_response_thinking_status = init_response_thinking_status_ls[i]
            if init_response_thinking_status != 'parsable_thinking':
                continue

        if correctness:
            problem_to_entries[problem_id]["correct"].append(i)
        else:
            problem_to_entries[problem_id]["incorrect"].append(i)
    
    return problem_to_entries

def combination(m, n):
    """
    Calculate the number of combinations of m choose n.
    """
    return math.comb(m, n)

def filter_problem(problem_to_entries, args):
    """
    Validate the problem to entry mapping.
    """
    total_valid_combos = 0
    correct_key = "correct"
    incorrect_key = "incorrect"
    valid_problem_ids = []
    for problem_id in problem_to_entries:
        num_trajs = args.num_true + args.num_false
        if len(problem_to_entries[problem_id][correct_key]) >= num_trajs and len(problem_to_entries[problem_id][incorrect_key]) >= num_trajs:
        # if len(problem_to_entries[problem_id][incorrect_key]) >= num_trajs:
            valid_problem_ids.append(problem_id)
            total_valid_combos += combination(len(problem_to_entries[problem_id][correct_key]), args.num_true) * combination(len(problem_to_entries[problem_id][incorrect_key]), args.num_false)
    print(f"Total valid combinations: {total_valid_combos}")
    return valid_problem_ids


def sample_combos(ds, problem_to_entries, sampled_problem_ids, args):
    """
    Sample the problems from the dataset.
    """
    sampled_problems = []
    correct_key = "correct"
    incorrect_key = "incorrect"

    n_correct = args.num_true
    n_incorrect = args.num_false

    def sample_combo(problem_id):
        correct_indices = np.random.choice(problem_to_entries[problem_id][correct_key], n_correct, replace=False)
        incorrect_indices = np.random.choice(problem_to_entries[problem_id][incorrect_key], n_incorrect, replace=False)
        indices = correct_indices.tolist() + incorrect_indices.tolist()
        np.random.shuffle(indices)

        key_ind = indices[0]

        # Basic Metadata for the problem
        new_entry = {
            "id": ds[key_ind]["id"],
            "problem": ds[key_ind]["problem"],
            "answer": ds[key_ind]["answer"],
            "source": ds[key_ind]["source"],
            "domain": ds[key_ind]["domain"],
            "label": ds[key_ind]["label"],
            "llama8b_solve_rate": ds[key_ind]["llama8b_solve_rate"],
        }

        # Add the trajectories and their metadata
        for i, traj_ind in enumerate(indices):
            new_entry[f"traj{i+1}"] = ds[traj_ind]["init_response_final"][:32768]
            new_entry[f"traj{i+1}_correctness"] = ds[traj_ind][args.init_response_correctness_column]
            new_entry[f"traj{i+1}_metadata"] = ds[traj_ind]
        return new_entry

    sampled_problems = [sample_combo(problem_id) for problem_id in tqdm(sampled_problem_ids)]
    # sample_combo)(problem_id) for problem_id in sampled_problem_ids)
    return sampled_problems

if __name__ == "__main__":

    args = parse_args()
    log_args(args)
    np.random.seed(args.seed)

    # Go through the input directory and load the dataset
    assert os.path.exists(args.input_dir)
    assert os.path.exists(os.path.join(args.input_dir, 'dataset_info.json')), "Input directory is not a valid Hugging Face dataset"
    ds = load_from_disk(args.input_dir)

    print(f"Sampling from {args.init_response_models}")
    ds = ds.filter(lambda x: x['init_response_generations_metadata']['model_config_alias'] in args.init_response_models, num_proc=16)
    print(f"Filtered dataset to {len(ds)} samples")

    # Ensure that there is "id" column in the dataset
    assert args.problem_id_column in ds.column_names, "Dataset does not have an 'id' column"
    assert args.init_response_correctness_column in ds.column_names, "Dataset does not have an 'init_response_generations_correctness' column"

    problem_to_entries = preprocess_dataset(ds, args)
    print(f"Preprocessed {len(problem_to_entries)} problems")
    print(f"Average number of trajectories: {np.mean([len(problem_to_entries[problem_id]['correct'] + problem_to_entries[problem_id]['incorrect']) for problem_id in problem_to_entries])}")
    valid_problem_ids = filter_problem(problem_to_entries, args)
    print(f"Valid problems: {len(valid_problem_ids)}")

    n_samples = len(valid_problem_ids)
    sampled_problem_ids = valid_problem_ids

    aggregated_data = sample_combos(ds, problem_to_entries, sampled_problem_ids, args)
    output_ds = Dataset.from_list(aggregated_data)
    print(f"Sampled {len(aggregated_data)} combinations")

    if args.output_dir is None:
        # use the parent directory of the input directory
        args.output_dir = os.path.dirname(args.input_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file_name = f"minimal_aggregated_data_T{args.num_true}_F{args.num_false}.ds"
    output_file_path = os.path.join(args.output_dir, output_file_name)
    output_ds.save_to_disk(output_file_path)

    print(f"Saved {n_samples} aggregated data to {output_file_path}")