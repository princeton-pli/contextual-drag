import math
import os
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk, Dataset
import argparse
import json
from glob import glob
import re

import torch
from joblib import Parallel, delayed
from IPython import embed
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        default='processed_flattened_sv_outputs',
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
        '--n_samples',
        type=int,
        default=10000,
        help="Number of samples to generate."
    )

    # Whether to filter data based on stepwise verification correctness
    parser.add_argument(
        '--filter_sv_correctness',
        action='store_true',
        default=False,
        help="Filter data based on stepwise verification correctness."
    )
    parser.add_argument(
        '--filter_init_response_completeness',
        action='store_true',
        default=False,
        help="Filter data based on init response completeness."
    )
    parser.add_argument(
        '--stepwise_verification_response_column',
        type=str,
        default='stepwise_verification_generations_generated_response',
        help="Column name containing the response of the stepwise verification."
    )
    parser.add_argument(
        '--stepwise_verification_correctness_column',
        type=str,
        default='stepwise_verification_correctness',
        help="Column name containing the correctness of the stepwise verification."
    )
    parser.add_argument(
        '--stepwise_verification_verdict_column',
        type=str,
        default='stepwise_verification_verdict',
        help="Column name containing the verdict of the stepwise verification."
    )
    parser.add_argument(
        '--init_response_models',
        nargs='+',
        type=str,
        default=['Qwen3_8B_Thinking', 'Qwen3_8B_NoThinking', 'LlamaR1_8B', 'Gemma3_4B', 'Llama3.1_8B', 'QwenR1_7B'],
        help="Models of which to sample init response."
    )
    parser.add_argument(
        '--correct_response_model',
        type=str,
        default='Qwen3_8B_Thinking',
        help="Model of which to sample correct response."
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='train',
        help="Data split to sample from."
    )
    parser.add_argument(
        '--stepwise_verification_verdict_boolean_column',
        type=str,
        default='stepwise_verification_boolean_verdict',
        help="Column name containing the boolean verdict of the stepwise verification."
    )
    parser.add_argument(
        '--remove_sources',
        nargs='+',
        type=str,
        default=[],
        help="Sources to remove from the dataset."
    )
    return parser.parse_args()

def log_args(args):
    """
    Print out the arguments.
    """
    print(f"Arguments: {args}")
    print(f"Filter SV Correctness: {args.filter_sv_correctness}")
    print(f"Stepwise Verification Response Column: {args.stepwise_verification_response_column}")
    print(f"Stepwise Verification Correctness Column: {args.stepwise_verification_correctness_column}")
    print(f"Num True: {args.num_true}")
    print(f"Num False: {args.num_false}")
    print(f"N Samples: {args.n_samples}")
    print(f"Init Response Models: {args.init_response_models}")
    print(f"Correct Response Model: {args.correct_response_model}")
    print(f"Data Split: {args.data_split}")

def preprocess_dataset(ds, args):
    """
    Aggregate the dataset based on the given arguments.
    """
    problem_to_entries = {}
    problem_to_correct_responses = {}

    print("Preprocessing dataset", flush=True)
    print("Getting problem ids", flush=True)
    problem_ids = list(ds[args.problem_id_column])
    print("Getting correctness", flush=True)
    correctness_ls = list(ds[args.init_response_correctness_column])
    print("Getting model metadata", flush=True)
    model_metadata_ls = list(ds['init_response_generations_metadata'])

    if args.filter_init_response_completeness:
        init_response_completeness_ls = list(ds['init_response_generations_finish_reason'])
    
    model_alias_ls = [metadata['model_config_alias'] for metadata in model_metadata_ls]
    stop_reasons = list(ds['init_response_generations_finish_reason'])

    if args.filter_sv_correctness:
        sv_correctness_ls = list(ds[args.stepwise_verification_correctness_column])

    print(f"Filtering SV Correctness: {args.filter_sv_correctness}")

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

        if correctness:
            problem_to_entries[problem_id]["correct"].append(i)
            if model_alias_ls[i] == args.correct_response_model:
                if stop_reasons[i] == 'stop':
                    if problem_id not in problem_to_correct_responses:
                        problem_to_correct_responses[problem_id] = []
                    problem_to_correct_responses[problem_id].append(i)

            if args.filter_sv_correctness:
                if sv_correctness_ls[i]:
                    problem_to_entries[problem_id]["correct_sv"].append(i)
        else:
            problem_to_entries[problem_id]["incorrect"].append(i)
            if args.filter_sv_correctness:
                if sv_correctness_ls[i]:
                    problem_to_entries[problem_id]["incorrect_sv"].append(i)
    
    return problem_to_entries, problem_to_correct_responses

def combination(m, n):
    """
    Calculate the number of combinations of m choose n.
    """
    return math.comb(m, n)

def filter_problem(problem_to_entries, problem_to_correct_responses, args):
    """
    Validate the problem to entry mapping.
    """
    total_valid_combos = 0
    correct_key = "correct_sv" if args.filter_sv_correctness else "correct"
    incorrect_key = "incorrect_sv" if args.filter_sv_correctness else "incorrect"
    valid_problem_ids = []
    for problem_id in problem_to_entries:
        if problem_id in problem_to_correct_responses:
            if len(problem_to_entries[problem_id][correct_key]) >= args.num_true and len(problem_to_entries[problem_id][incorrect_key]) >= args.num_false:
                valid_problem_ids.append(problem_id)
                total_valid_combos += combination(len(problem_to_entries[problem_id][correct_key]), args.num_true) * combination(len(problem_to_entries[problem_id][incorrect_key]), args.num_false)
    print(f"Total valid combinations: {total_valid_combos}")
    return valid_problem_ids

def remove_trailing_formatting_thoughts(thinking_trace):
    # Split by "."
    thinking_trace_split = thinking_trace.split(".")
    if thinking_trace_split[-1] != "":
        return thinking_trace
    last_segment = thinking_trace_split[-2]
    filtered_last_segment = re.sub(r"[^a-zA-Z0-9\s]", "", last_segment).lower()
    key_words = [' correct', ' incorrect', ' wrong', ' error', ' verdict', ' mistake', ' unclear']
    if any(keyword in filtered_last_segment for keyword in key_words):
        return thinking_trace
    return '.'.join(thinking_trace_split[:-2]) + '.'
    
def parse_sv_response_templates(sv_response, remove_trailing_formatting=True):
    # If not a valid thinking response, return None

    if sv_response.startswith("analysis") and sv_response[8] != " ":
        # In GPT-OSS format
        if sv_response.count("assistantfinal") > 1:
            return None, None
        if "assistantfinal" in sv_response:
            thinking_trace = sv_response[8:].split("assistantfinal")[-2]
            final_output = sv_response.split("assistantfinal")[-1]
            return thinking_trace, final_output
        else:
            return None, None

    if "<think>" not in sv_response or "</think>" not in sv_response:
        return None, None
    # If more than one think or end think, return None
    if sv_response.count("<think>") > 1 or sv_response.count("</think>") > 1:
        return None, None

    thinking_trace = sv_response.split("<think>")[1].split("</think>")[0]
    final_output = sv_response.split("</think>")[-1]
    # Remove trailing newlines and spaces
    thinking_trace = thinking_trace.strip(' \n')
    if remove_trailing_formatting:
        cleaned_thinking_trace = remove_trailing_formatting_thoughts(thinking_trace)
    else:
        cleaned_thinking_trace = thinking_trace
    final_output = final_output.strip(' \n')

    return thinking_trace, final_output

def sample_combos(ds, problem_to_entries, sampled_problem_ids, args):
    """
    Sample the problems from the dataset.
    """
    sampled_problems = []
    correct_key = "correct_sv" if args.filter_sv_correctness else "correct"
    incorrect_key = "incorrect_sv" if args.filter_sv_correctness else "incorrect"

    n_correct = args.num_true
    n_incorrect = args.num_false

    sampled_pairs = []

    for problem_id in sampled_problem_ids:
        correct_indices = np.random.choice(problem_to_entries[problem_id][correct_key], n_correct, replace=False).tolist()
        incorrect_indices = np.random.choice(problem_to_entries[problem_id][incorrect_key], n_incorrect, replace=False).tolist()
        indices = correct_indices + incorrect_indices

        # randomly shuffle the indices
        np.random.shuffle(indices)

        # record the indices of the correct trajectories
        correct_positions = []
        for i in correct_indices:
            correct_positions.append(indices.index(i))

        # randomly sample the index of the correct trajectory
        if n_correct > 0:
            incontext_correct_index = np.random.choice(correct_positions)
        else:
            incontext_correct_index = -1
            
        sampled_pairs.append((problem_id, indices, incontext_correct_index))

    def sample_combo(problem_id, indices, incontext_correct_index):

        key_ind = indices[0]
        incontext_correct_ind = indices[incontext_correct_index]

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
            new_entry[f"traj{i+1}_sv_verdict"] = ds[traj_ind][args.stepwise_verification_verdict_column]
            new_entry[f"traj{i+1}_sv_verdict_boolean"] = ds[traj_ind][args.stepwise_verification_verdict_boolean_column]
            
            sv_output = ds[traj_ind][args.stepwise_verification_response_column]
            thinking_trace, final_output = parse_sv_response_templates(sv_output)
            new_entry[f"traj{i+1}_sv_thinking_trace"] = thinking_trace
            new_entry[f"traj{i+1}_sv_final_output"] = final_output
            
            new_entry[f"traj{i+1}_metadata"] = ds[traj_ind]
            new_entry[f"traj{i+1}_sv_correctness"] = ds[traj_ind][args.stepwise_verification_correctness_column]

        # If there is a correct index, add the correct response
        if incontext_correct_index != -1:
            # The correct trajectory is selected to be the incontext correct trajectory
            new_entry["correct_traj_id"] = f"{incontext_correct_index + 1}"
            incontext_correct_traj_final_output = ds[incontext_correct_ind]["init_response_final"][:32768]
            new_entry["incontext_correct_traj_final_output"] = incontext_correct_traj_final_output
        else:
            new_entry["correct_traj_id"] = None
            new_entry["incontext_correct_traj_thinking_trace"] = None
            new_entry["incontext_correct_traj_final_output"] = None

        # Add the ground truth response
        assert problem_id in problem_to_correct_responses, "Problem id not found in problem to correct responses"
        gt_index = int(np.random.choice(problem_to_correct_responses[problem_id]))
        new_entry["traj_gt"] = ds[gt_index]["init_response_generations_generated_response"]
        gt_thinking_trace, gt_final_output = parse_sv_response_templates(new_entry["traj_gt"], remove_trailing_formatting=False)
        new_entry["traj_gt_thinking_trace"] = gt_thinking_trace
        new_entry["traj_gt_final_output"] = gt_final_output
        new_entry["traj_gt_metadata"] = ds[gt_index]
        # new_entry["traj_gt_sv_correctness"] = ds[correct_indices[0]][args.stepwise_verification_correctness_column]
        return new_entry

    # Sequentially sample the problems
    # sampled_problems = [sample_combo(problem_id, indices, incontext_correct_index) for problem_id, indices, incontext_correct_index in tqdm(sampled_pairs)]
    
    # Parallelly sample the problems
    sampled_problems = Parallel(n_jobs=-1)(
        delayed(sample_combo)(problem_id, indices, incontext_correct_index) for problem_id, indices, incontext_correct_index in tqdm(sampled_pairs)
    )
    return sampled_problems

def load_split_ids(split):
    splits = split.split('+')
    for split in splits:
        assert split in ['sft', 'rl', 'val'], "Data split must be one of sft, rl, val"
    split_ids = []
    for split in splits:
        split_file = split_template.format(split=split)
        with open(split_file, 'r') as f:
            split_ids.extend(json.load(f))
    return split_ids

if __name__ == "__main__":

    split_template = "/scratch/gpfs/ARORA/xz4134/Research_src/Aggregation/in-context-aggregation/data/big_math_rl_verified/train_split/detailed_splits/{split}_ids.json"

    args = parse_args()
    log_args(args)

    np.random.seed(args.seed)
    split_ids = load_split_ids(args.data_split)

    # Go through the input directory and load the dataset
    assert os.path.exists(args.input_dir)
    assert os.path.exists(os.path.join(args.input_dir, 'dataset_info.json')), "Input directory is not a valid Hugging Face dataset"
    ds = load_from_disk(args.input_dir)

    print(f"Sampling from {args.init_response_models}")
    # filter out the unwanted sources
    if args.remove_sources:
        print(f"Removing sources: {args.remove_sources}")
        print(f"Size of dataset before filtering: {len(ds)}")
        ds = ds.filter(lambda x: x['source'] not in args.remove_sources, num_proc=16)
        print(f"Size of dataset after filtering: {len(ds)}")
    ds = ds.filter(lambda x: x['id'] in split_ids, num_proc=16)
    ds = ds.filter(lambda x: x['init_response_generations_metadata']['model_config_alias'] in args.init_response_models, num_proc=16)
    # init_response_models = list(ds['init_response_generations_metadata']['model_config_alias'])
    print(f"Filtered dataset to {len(ds)} samples")

    # Ensure that there is "id" column in the dataset
    assert args.problem_id_column in ds.column_names, "Dataset does not have an 'id' column"
    assert args.init_response_correctness_column in ds.column_names, "Dataset does not have an 'init_response_generations_correctness' column"
    if args.filter_sv_correctness:
        assert args.stepwise_verification_correctness_column in ds.column_names, "Dataset does not have an 'stepwise_verification_correctness' column"

    problem_to_entries, problem_to_correct_responses = preprocess_dataset(ds, args)
    print(f"Preprocessed {len(problem_to_entries)} problems")
    print(f"Average number of trajectories: {np.mean([len(problem_to_entries[problem_id]['correct'] + problem_to_entries[problem_id]['incorrect']) for problem_id in problem_to_entries])}")
    valid_problem_ids = filter_problem(problem_to_entries, problem_to_correct_responses, args)
    print(f"Valid problems: {len(valid_problem_ids)}", flush=True)

    remaining_samples = len(valid_problem_ids)

    full_rounds = args.n_samples // remaining_samples
    remaining_samples = args.n_samples % remaining_samples
    sampled_problem_ids = valid_problem_ids * full_rounds
    sampled_problem_ids.extend(np.random.choice(valid_problem_ids, remaining_samples, replace=False))

    aggregated_data = sample_combos(ds, problem_to_entries, sampled_problem_ids, args)
    output_ds = Dataset.from_list(aggregated_data)
    print(f"Sampled {len(aggregated_data)} combinations")

    if args.output_dir is None:
        # use the parent directory of the input directory
        args.output_dir = os.path.dirname(args.input_dir)

    args.output_dir = args.output_dir.replace('stage2_stepwise_verification', 'stage3_generate_aggregation')
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_name = f"{args.data_split}_N{args.n_samples}_T{args.num_true}_F{args.num_false}_SV{args.filter_sv_correctness}_REMOVE{'_'.join(args.remove_sources)}.ds"
    if args.filter_init_response_completeness:
        output_file_name =output_file_name.replace('.ds', '_IRComplete.ds')

    output_file_path = os.path.join(args.output_dir, output_file_name)
    output_ds.save_to_disk(output_file_path)
    
    sample_datapoint = output_ds[0]
    sample_output_path = output_file_path.replace('.ds', '_sample.json')
    with open(sample_output_path, 'w') as f:
        json.dump(sample_datapoint, f, indent=4)

    print(f"Saved aggregated data to {output_file_path}")