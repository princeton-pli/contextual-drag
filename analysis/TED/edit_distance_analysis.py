from edit_distances import edit_distance
from sample_preprocessing import build_processed_anchored_data, load_init_responses

import argparse
import numpy as np
import json
from joblib import Parallel, delayed
import copy
import os
import ast
from IPython import embed

import matplotlib.pyplot as plt

def compute_edit_distance(processed_entry, metric="levenshtein"):

    for traj_answers in processed_entry:
        for init_response_metadata in processed_entry[traj_answers]['init_response']:
            init_response_answer = init_response_metadata[0]

            edit_distances = []

            for traj_answer in ast.literal_eval(traj_answers):
                edit_dist = edit_distance(traj_answer, init_response_answer, metric=metric)
                edit_distances.append(edit_dist)

            if "edit_distance" not in init_response_metadata[1]:
                init_response_metadata[1]["edit_distance"] = {}
            init_response_metadata[1]["edit_distance"].update({metric: np.mean(edit_distances)})

        for anchored_metadata in processed_entry[traj_answers]['anchored_responses']:
            anchored_answer = anchored_metadata[0]

            edit_distances = []
            for traj_answer in ast.literal_eval(traj_answers):
                edit_dist = edit_distance(traj_answer, anchored_answer, metric=metric)
                edit_distances.append(edit_dist)

            if "edit_distance" not in anchored_metadata[1]:
                anchored_metadata[1]["edit_distance"] = {}
            anchored_metadata[1]["edit_distance"].update({metric: np.mean(edit_distances)})

    return processed_entry

def batched_compute_edit_distance_parallel(processed_data, metric):
    print(f"Computing edit distance for {metric} metric. Total number of problems: {len(processed_data)}")
    entries_list = [(k, v) for k, v in processed_data.items()]
    def compute_edit_distance_wrapper(problem_id, problem_entries):
        return (problem_id, compute_edit_distance(problem_entries, metric=metric))
    computed_entries = Parallel(n_jobs=20)(
        delayed(compute_edit_distance_wrapper)(problem_id, problem_entries) for problem_id, problem_entries in entries_list
    )
    computed_data = {problem_id: computed_entry for problem_id, computed_entry in computed_entries}
    return computed_data

def gather_distance_stats(
        entry,
        filter_stop=True,
        filter_verification=False,
        filter_incorrect=False,
        metric="tree",
    ):
    # Apply filters to the entry
    for anchor in entry:
        for key in ["anchored_responses", "init_response"]:
            remained_ls = []
            
            for item in entry[anchor][key]:
                if filter_stop and item[1]['finish_reason'] != 'stop':
                    continue
                if filter_verification:
                    if 'verdict' in item[1]:
                        if item[1]['verdict'] != False:
                            continue
                if filter_incorrect and item[1]['correctness'] != False:
                    continue
                remained_ls.append(item)
            entry[anchor][key] = remained_ls
            mean_distance_ls = [x[1]['edit_distance'][metric] for x in remained_ls]
            entry[anchor][key + "_distance_stats"] = np.mean(mean_distance_ls) if len(mean_distance_ls) > 0 else None

    # Compute the distance stats
    anchored_responses_distance_stats = []
    init_response_distance_stats = []
    for anchor in entry:
        valid_anchor = entry[anchor]["anchored_responses_distance_stats"] is not None and entry[anchor]["init_response_distance_stats"] is not None
        if valid_anchor:
            anchored_responses_distance_stats.append(entry[anchor]["anchored_responses_distance_stats"])
            init_response_distance_stats.append(entry[anchor]["init_response_distance_stats"])
    return {
        "anchored_responses": np.mean(anchored_responses_distance_stats) if len(anchored_responses_distance_stats) > 0 else None,
        "init_response": np.mean(init_response_distance_stats) if len(init_response_distance_stats) > 0 else None,
    }

def compute_stats_for_dataset(processed_data, **kwargs):
    ret = []
    for _, entry in processed_data.items():
        distance_stats = gather_distance_stats(entry, **kwargs)
        ret.append(distance_stats)
    return ret

def visualize_stats(stats):
    anchored_stats = []
    init_response_stats = []
    for item in stats:
        if item["anchored_responses"] is None or item["init_response"] is None:
            continue
        anchored_stats.append(item["anchored_responses"])
        init_response_stats.append(item["init_response"])
    print(f"Anchored responses stats: {np.mean(anchored_stats)}")
    print(f"Init response stats: {np.mean(init_response_stats)}")

    plt.scatter(anchored_stats, init_response_stats)
    plt.xlabel("Anchored responses")
    plt.ylabel("Init response")
    plt.title("Stats")
    plt.savefig("stats.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name.", default="GPT_OSS_120B")
    parser.add_argument("--anchored_data_template", type=str, help="Path to the anchored data file.", default="../../outputs/2f/24-game/{model_name}/evaluated_{model_name}_flattened.jsonl")
    parser.add_argument("--init_response_data_template", type=str, help="Path to the init response data file.", default="../../outputs/initial_sampling/24-game/{model_name}/evaluated_{model_name}_flattened.jsonl")
    parser.add_argument("--metric", type=str, help="Metric to use for edit distance.", default="tree")
    parser.add_argument("--iteration", type=str, help="Iteration. Format: {run_id}_{num_round}.", default=None)
    parser.add_argument("--task", type=str, help="Task name.", default="")
    args = parser.parse_args()

    if '2f' in args.anchored_data_template:
        cache_dir = "direct2"
        num_trajs = 2
    elif '1f' in args.anchored_data_template:
        cache_dir = "direct"
        num_trajs = 1
    else:
        run_id, num_round = args.iteration.split("_")
        cache_dir = f"recursive{run_id}"
        num_trajs = 1
    save_location = f"cache/{cache_dir}/{args.task}/{args.model_name}_{num_round}.json"
    print(f"Saving processed data to {save_location}")
    if os.path.exists(save_location):
        with open(save_location, "r") as f:
            processed_data = json.load(f)
        print(f"Loaded processed data from {save_location}")
    else:
        print(f"Building processed data for {args.anchored_data_template.format(model_name=args.model_name)}")
        processed_data = build_processed_anchored_data(args.anchored_data_template.format(model_name=args.model_name), num_trajs=num_trajs, iteration=num_round)
        print(f"Loading init responses for {args.init_response_data_template.format(model_name=args.model_name)}")
        processed_data = load_init_responses(processed_data, args.init_response_data_template.format(model_name=args.model_name))

        if args.metric == "binary":
            processed_data = batched_compute_edit_distance_parallel(processed_data, metric=args.metric)
        else:
            for metric in ["levenshtein", "tree", "binary"]:
                processed_data = batched_compute_edit_distance_parallel(processed_data, metric=metric)

        save_directory = os.path.dirname(save_location)
        os.makedirs(save_directory, exist_ok=True)
        with open(save_location, "w") as f:
            json.dump(processed_data, f)

if __name__ == "__main__":
    main()