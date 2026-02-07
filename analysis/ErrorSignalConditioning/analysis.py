import argparse
import json
import re
from pathlib import Path

import numpy as np
from datasets import Dataset
from joblib import Parallel, delayed

raw_path_template = "../../outputs/initial_sampling/{task_name}/{model_name}/evaluated_{model_name}.jsonl"
conditioned_path_template = "../../outputs/prompted/{task_name}/{model_name}/evaluated_{model_name}.jsonl"
TASK_ORDER = ['aime24', 'aime25', 'hmmt24', 'hmmt25'] #, 'gpqa', 'mmlu', 'crux-i', '24-game']
TASK_DISPLAY = {
    'aime24': 'AIME24',
    'aime25': 'AIME25',
    'hmmt24': 'HMMT24',
    'hmmt25': 'HMMT25',
    'gpqa': 'GPQA',
    'mmlu': 'MMLU',
    'crux-i': 'Crux-I',
    '24-game': 'Game of 24',
}

def parse_verification_output(output):
    if output is None or len(output.strip()) == 0:
        return None
    output = output.lower().replace('*', '').replace('\n', '')
    
    incorrect_keywords = ["draft solution is incorrect", "draft is incorrect", "draft incorrect", "\\boxed{\\text{incorrect}}"]
    correct_keywords = ["draft solution is correct", "draft is correct", "draft correct", "\\boxed{\\text{correct}}"]

    for kw in correct_keywords:
        if kw in output:
            return False
    for kw in incorrect_keywords:
        if kw in output:
            return True

    pattern = r"<overall_verdict>\s*([^\n]*)</overall_verdict>"
    verdict_results = re.findall(pattern, output, re.DOTALL)
    verdict_result = verdict_results[-1].strip() if verdict_results else None
    if verdict_result is None:
        return None
    return verdict_result.lower() == "incorrect"

def parse_verification_for_entry(entry):
    responses = entry['init_response_generations']
    valid_verdict = False
    for response in responses:
        raw_response = response['generated_response']
        verdict = parse_verification_output(raw_response)
        response['verdict'] = verdict
        if verdict is True and response['finish_reason'] == 'stop':
            valid_verdict = True
    
    entry['valid_verdict'] = valid_verdict
    return entry

def filter_valid_verdict(entry):
    responses = entry['init_response_generations']
    candidate_indices = []
    for i, response in enumerate(responses):
        if response['verdict'] and response['finish_reason'] == 'stop':
            candidate_indices.append(i)
    valid_responses = [responses[i] for i in candidate_indices]
    entry['init_response_generations'] = valid_responses
    return entry

def get_average_correctness(ds):
    correctness_ls = []
    for entry in ds:
        correctness_ls.append(
            np.mean([response['correctness'] == True for response in entry['init_response_generations']])
        )
    return correctness_ls

def get_individual_correctness(ds):
    correctness_ls = []
    for entry in ds:
        correctness_ls.append(
            [response['correctness'] == True for response in entry['init_response_generations']]
        )
    return correctness_ls

def merge_ids(ds):
    id2entry = {}
    for entry in ds:
        entry_id = entry['id']
        if entry_id not in id2entry:
            id2entry[entry_id] = entry
        else:
            id2entry[entry_id]['init_response_generations'].extend(entry['init_response_generations'])
    return Dataset.from_list(list(id2entry.values()))

def analyze_correct_verification_conditioning(task_name, model_name):

    # Load conditioned evaluation results
    eval_file_name = conditioned_path_template.format(task_name=task_name, model_name=model_name)
    cached_ls = []
    for line in open(eval_file_name, "r"):
        entry = json.loads(line)
        cached_ls.append(entry)
    
    raw_ds = Dataset.from_list(cached_ls)
    raw_ds = merge_ids(raw_ds)
    raw_ds_ids = set(v['id'] for v in raw_ds)
    raw_ds_w_verdict = raw_ds.map(parse_verification_for_entry)
    # only keep entries with valid verdict
    filtered_raw_ds = raw_ds_w_verdict.filter(lambda x: x['valid_verdict'])
    filtered_correct_verdict_ds = filtered_raw_ds.map(filter_valid_verdict)

    filtered_ids = [v['id'] for v in filtered_correct_verdict_ds]
    filtered_ids_set = set(filtered_ids)
    print(f"Filtered {len(filtered_ids)} out of {len(raw_ds)}")

    # Load raw evaluation results
    eval_file_name = raw_path_template.format(task_name=task_name, model_name=model_name)
    cached_ls = []
    for line in open(eval_file_name, "r"):
        entry = json.loads(line)
        cached_ls.append(entry)
    raw_init_sampling_ds = Dataset.from_list(cached_ls)
    raw_init_sampling_ds = merge_ids(raw_init_sampling_ds)
    raw_init_sampling_ds_matched = raw_init_sampling_ds.filter(lambda x: x['id'] in raw_ds_ids)
    raw_init_sampling_ds_filtered = raw_init_sampling_ds_matched.filter(lambda x: x['id'] in filtered_ids_set)

    assert len(filtered_raw_ds) == len(raw_init_sampling_ds_filtered), f"Filtered {len(filtered_raw_ds)} out of {len(raw_ds)} but {len(raw_init_sampling_ds_filtered)} out of {len(raw_init_sampling_ds)}"
    
    ave_correctness_raw = get_average_correctness(filtered_raw_ds)
    ave_correctness_filtered = get_average_correctness(filtered_correct_verdict_ds)
    ave_correctness_raw_init_sampling = get_average_correctness(raw_init_sampling_ds_matched)
    ave_correctness_filtered_init_sampling = get_average_correctness(raw_init_sampling_ds_filtered)

    individual_correctness_raw = get_individual_correctness(filtered_raw_ds)
    individual_correctness_filtered = get_individual_correctness(filtered_correct_verdict_ds)
    individual_correctness_raw_init_sampling = get_individual_correctness(raw_init_sampling_ds_matched)
    individual_correctness_filtered_init_sampling = get_individual_correctness(raw_init_sampling_ds_filtered)

    assert len(ave_correctness_raw) == len(individual_correctness_raw)
    assert len(ave_correctness_filtered) == len(individual_correctness_filtered)
    assert len(ave_correctness_raw_init_sampling) == len(individual_correctness_raw_init_sampling)
    assert len(ave_correctness_filtered_init_sampling) == len(individual_correctness_filtered_init_sampling)

    individual_correctness_raw_init_sampling_rescaled = []
    for i in range(len(ave_correctness_raw)):
        dup = len(individual_correctness_filtered[i])
        individual_correctness_raw_init_sampling_rescaled.append([ave_correctness_raw_init_sampling[i]] * dup)

    flattened_correctness_filtered = []
    for x in individual_correctness_filtered:
        flattened_correctness_filtered.extend(x)

    flattened_correctness_raw_init_sampling_rescaled = []
    for x in individual_correctness_raw_init_sampling_rescaled:
        flattened_correctness_raw_init_sampling_rescaled.extend(x)

    return {
        "correctness_raw": np.mean(ave_correctness_raw),
        "correctness_filtered": np.mean(ave_correctness_filtered),
        "correctness_raw_init_sampling": np.mean(ave_correctness_raw_init_sampling),
        "correctness_filtered_init_sampling": np.mean(ave_correctness_filtered_init_sampling),
        "num_problems": len(raw_init_sampling_ds_matched),
        "num_problems_filtered": len(raw_init_sampling_ds_filtered),
    }

def parse_args():
    default_json = Path(__file__).with_name("correct_verification_conditioning_results_prompted.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=default_json, help="Where to write metrics JSON.")
    return parser.parse_args()


def main():
    tasks = ['aime24', 'aime25', 'hmmt24', 'hmmt25', 'gpqa', 'mmlu', 'crux-i', '24-game']
    models = ['Qwen3_8B', 'Qwen3_32B', 'GPT_OSS_120B', 'Nemotron_7B', 'GPT_OSS_20B', 'Nemotron_32B'] #, 'Qwen2.5_7B', 'Qwen2.5_32B']
    args = parse_args()

    task_model_pairs = [(task, model) for task in tasks for model in models]
    results = Parallel(n_jobs=-1)(
        delayed(analyze_correct_verification_conditioning)(task, model) for task, model in task_model_pairs
    )

    save_dict = {task: {model: {} for model in models} for task in tasks}
    for i, (task, model) in enumerate(task_model_pairs):
        result = results[i]
        save_dict[task][model] = {
            "correctness_raw": result['correctness_raw'],
            "correctness_filtered": result['correctness_filtered'],
            "correctness_raw_init_sampling": result['correctness_raw_init_sampling'],
            "correctness_filtered_init_sampling": result['correctness_filtered_init_sampling'],
            "num_problems": result['num_problems'],
            "num_problems_filtered": result['num_problems_filtered'],
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(save_dict, args.output_json.open("w"), indent=2)

if __name__ == "__main__":
    main()