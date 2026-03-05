import datasets
from transformers import AutoTokenizer
import json
import re
from tqdm import tqdm
import numpy as np
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from IPython import embed

import os
import sys
root_dir = '../utils'
if root_dir not in sys.path:
    sys.path.append(root_dir)
from general_inference.src.data_utils import *
global tokenizer

count_mapping = {
    1: 'first',
    2: 'second',
    3: 'third',
    4: 'fourth',
    5: 'fifth',
    6: 'sixth',
    7: 'seventh',
    8: 'eighth'
}

trim_words = [
    "let's produce",
    "final answer",
    "now output",
    "let's output",
    "boxed",
    "we need to output",
    "format",
    "let's output",
    "thus output",
    "thus final",
    "thus answer",
    "thus produce",
    "thus, output",
    "thus, final",
    "thus, answer",
    "thus, produce",
    "so final",
    "thus verdict",
    "hence answer",
    "so output",
    "we output",
    "we'll output",
    "write answer",
    "now produce",
    "final output",
    "final answer",
    "final message",
    "output verdict",
    "provide verdict",
    "output fix",
    "provide fix",
]

think_tokens = ('<think>\n', '\n</think>\n\n')
remove_think_pattern = re.compile(f"{re.escape(think_tokens[0])}(.*?){re.escape(think_tokens[1])}", re.DOTALL)
think_pattern = re.compile(f"({re.escape(think_tokens[0])}(.*?){re.escape(think_tokens[1])})", re.DOTALL)

def remove_trailing_line(response):

    # Remove trailing newlines and spaces
    try:
        response = response.strip('\n')
    except:
        print(f"Error stripping trailing newline: {response}")
        return response

    # Get the last line of the response:
    last_line = response.split('\n\n')[-1].lower()
    len_last_line = len(last_line)

    num_words_last_line = len(last_line.split(' '))
    trim_flag = False

    # If the last line is too short, return the response

    for word in trim_words:
        if word in last_line:
            trim_flag = True
            break
    if num_words_last_line <= 2:
        trim_flag = True
    
    if trim_flag and num_words_last_line <= 10:
        # print(f"Trimming trailing line: {last_line}")
        response = response[:-len_last_line].strip('\n')
    # else:
    #     print(f"Not trimming trailing line: {last_line}")

    # Get the last sentence of the response
    trim_flag = False
    last_sentence = response.strip('.').split('.')[-1]
    last_sentence_lower = last_sentence.lower()
    num_words_last_sentence = len(last_sentence_lower.split(' '))
    len_last_sentence = len(last_sentence_lower)

    if num_words_last_sentence <= 2:
        trim_flag = True
    
    for word in trim_words:
        if word in last_sentence:
            trim_flag = True
            break

    if "</fix>" in last_sentence:
        trim_flag = False

    if trim_flag and num_words_last_sentence <= 10:
        response = last_sentence.join(response.split(last_sentence)[:-1])

    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_merged_ds_only', action='store_true', default=False)
    parser.add_argument('--data_config_path', type=str, default='/scratch/gpfs/ARORA/xz4134/Research_src/Aggregation/in-context-aggregation/data_generation_scripts/big_math_rl/stage4_format_training_data/configs/sft_n10k_svmix/111mix_lmax16384_n10000.json')
    parser.add_argument('--template_path', type=str, default='/scratch/gpfs/ARORA/xz4134/Research_src/Aggregation/in-context-aggregation/data_generation_scripts/big_math_rl/stage4_format_training_data/sft_templates.json')
    parser.add_argument('--output_path', type=str, default='/scratch/gpfs/ARORA/xz4134/Research_src/Aggregation/in-context-aggregation/data_generation_scripts/big_math_rl/stage4_format_training_data/outputs/')
    return parser.parse_args()

def log_config(config):
    for key in config:
        if key != 'training_sources':
            print(f"{key}:\t{config[key]}")
    for source in config["training_sources"]:
        print(f"Training source:\t{source['path']}\t{source['weight']}")

def get_weighted_concatenated_dataset(training_sources):

    # Make sure the training sources are not repetitive
    training_sources_unique = list(set([x["path"] for x in training_sources]))
    assert len(training_sources) == len(training_sources_unique), "Training sources are repetitive"
    
    ds_list = []
    weight_vec, size_vec = np.zeros(len(training_sources)), np.zeros(len(training_sources))
    for i, ds_config in enumerate(training_sources):
        ds_list.append(datasets.load_from_disk(ds_config["path"]))
        weight_vec[i] = ds_config["weight"]
        size_vec[i] = len(ds_list[-1])

    reformated_size_vec = size_vec / weight_vec
    minimum_ds = np.argmin(reformated_size_vec)

    # Based on the assumption that we should use all data from the dataset with the smallest size
    # We can compute the subsample ratio for each dataset
    normalized_weight_vec = weight_vec / weight_vec[minimum_ds]
    size_vec = normalized_weight_vec * size_vec[minimum_ds]
    size_vec = size_vec.astype(int)

    for i, ds in enumerate(ds_list):
        ds_list[i] = ds.select(np.arange(size_vec[i]))

    ds = datasets.concatenate_datasets(ds_list)
    return ds

def validate_prompt_and_ds(template, ds):
    print(f"Validating prompt template and dataset...")
    template_fields = extract_template_fields(template)
    valid_fields, missing_fields = validate_template_fields(template_fields, ds.column_names)

    print(f"Template:\n\n{template}\n")
    print('\n'.join(sorted(list(template_fields))) + "\n" + "="*100 + "\n\n")
    print(missing_fields)
    if missing_fields:
        raise ValueError(f"Prompt template requires fields that are missing from dataset: {missing_fields}. "
                        f"Available columns: {ds.column_names}")
    print(f"VALIDATION PASSED")
    return template_fields

def substitute_thinking_trace_index(trace, index):
    index_str = count_mapping[index]
    if trace is None:
        print(f"Thinking trace is None for index {index}")
        return ""
    trace = trace.replace(f"The solution", f"The {index_str} solution")
    trace = trace.replace(f"The draft", f"The {index_str} draft")
    trace = trace.replace(f"the solution", f"the {index_str} solution")
    trace = trace.replace(f"the draft", f"the {index_str} draft")
    trace = trace.replace(f"Draft", f"The {index_str} draft")
    return trace

def remap_thinking_trace_index(entry, replace_only_for_multidraft=False):
    replace_candidates = []
    for i in range(1, 9):
        if f"traj{i}_sv_thinking_trace" in entry:
            replace_candidates.append([i, f"traj{i}_sv_thinking_trace"])

    for i, trace_key in replace_candidates:
        entry[trace_key] = remove_trailing_line(entry[trace_key])

    if replace_only_for_multidraft and len(replace_candidates) == 1:
        return entry
    for i, trace_key in replace_candidates:
        entry[trace_key] = substitute_thinking_trace_index(entry[trace_key], i)
    return entry

def format_entry(
    entry,
    prompt_template,
    prompt_template_fields,
    answer_reuse_template,
    answer_reuse_template_fields,
    answer_new_template,
    answer_new_template_fields,
    remap_sv_index,
    replace_only_for_multidraft
):
    prompt = format_template_with_row(prompt_template, entry, prompt_template_fields)
    correct_traj_id = entry['correct_traj_id']
    if remap_sv_index:
        entry = remap_thinking_trace_index(entry, replace_only_for_multidraft)
    if correct_traj_id is None:
        response = format_template_with_row(answer_new_template, entry, answer_new_template_fields)
    else:
        response = format_template_with_row(answer_reuse_template, entry, answer_reuse_template_fields)
    return {
        'id': entry['id'],
        'prompt': prompt,
        'response': response,
    }

def count_tokens(example):
    prompt = example['prompt']
    response = example['response']
    return {'prompt_n_tokens': len(tokenizer.encode(prompt)), 'response_n_tokens': len(tokenizer.encode(response))}

def visualize_data(ds, visualization_path):
    prompt_n_tokens = list(ds['prompt_n_tokens'])
    response_n_tokens = list(ds['response_n_tokens'])
    total_n_tokens = [p + r for p, r in zip(prompt_n_tokens, response_n_tokens)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram for prompt tokens
    axes[0, 0].hist(prompt_n_tokens, bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Prompt Token Count Distribution')
    axes[0, 0].set_xlabel('Number of Tokens')
    axes[0, 0].set_ylabel('Frequency')
    
    # Histogram for response tokens
    axes[0, 1].hist(response_n_tokens, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Response Token Count Distribution')
    axes[0, 1].set_xlabel('Number of Tokens')
    axes[0, 1].set_ylabel('Frequency')
    
    # Histogram for total tokens
    axes[1, 0].hist(total_n_tokens, bins=50, alpha=0.7, color='red')
    axes[1, 0].set_title('Total Token Count Distribution')
    axes[1, 0].set_xlabel('Number of Tokens')
    axes[1, 0].set_ylabel('Frequency')
    
    # Scatter plot for correlation
    axes[1, 1].scatter(prompt_n_tokens, response_n_tokens, alpha=0.5, s=10)
    axes[1, 1].set_title('Prompt vs Response Token Count')
    axes[1, 1].set_xlabel('Prompt Tokens')
    axes[1, 1].set_ylabel('Response Tokens')
    
    plt.tight_layout()
    plt.savefig(visualization_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Token distribution analysis saved to {visualization_path}")

def ds2gpt_style_jsonl(ds, output_path):
    ds_jsonl = [
        {
            'messages': [
                {'role': 'user', 'content': entry['prompt']},
                {'role': 'assistant', 'content': entry['response']},
            ],
        } for entry in ds
    ]
    with open(output_path, 'w') as f:
        for entry in ds_jsonl:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved {len(ds_jsonl)} examples to {output_path}")

def main():
    args = parse_args()
    data_config_path = args.data_config_path
    output_path = args.output_path

    config = json.load(open(data_config_path, 'r'))
    remap_sv_index = config["remap_sv_index"]
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    
    training_sources = config["training_sources"]
    total_size = config["total_size"]
    ds_name = config["ds_name"]
    np.random.seed(config["mixing_seed"])
    output_ds_path = os.path.join(output_path, ds_name)

    ds = get_weighted_concatenated_dataset(training_sources)
    if len(ds) > total_size:
        ds = ds.select(np.random.choice(len(ds), total_size, replace=False))

    if args.dump_merged_ds_only:
        # Shuffle the dataset
        print(f"Shuffling the merged dataset with seed {config['mixing_seed']}. To format the dataset for llamafactory sft, please remove the --dump_merged_ds_only flag.")
        ds = ds.shuffle(seed=config["mixing_seed"])
        merged_ds_path = os.path.join(output_ds_path, f'merged_stage3_data.ds') 
        ds.save_to_disk(merged_ds_path)
        print(f"Saved merged dataset to {merged_ds_path}")
        print(f"Merged dataset size: {len(ds)}")
        return

    log_config(config)
    print(f"Concatenated dataset size: {len(ds)}")

    template = json.load(open(args.template_path, 'r'))

    prompt_template = template[config["aggregate_prompt_template"]]
    prompt_template_fields = validate_prompt_and_ds(prompt_template, ds)

    answer_reuse_template = template[config["answer_reuse_template"]]
    answer_reuse_template_fields = validate_prompt_and_ds(answer_reuse_template, ds)

    answer_new_template = template[config["answer_new_template"]]
    answer_new_template_fields = validate_prompt_and_ds(answer_new_template, ds)

    if "replace_only_for_multidraft" in config:
        replace_only_for_multidraft = config["replace_only_for_multidraft"]
    else:
        replace_only_for_multidraft = False

    formatted_data = [format_entry(
        entry,
        prompt_template,
        prompt_template_fields,
        answer_reuse_template,
        answer_reuse_template_fields,
        answer_new_template,
        answer_new_template_fields,
        remap_sv_index,
        replace_only_for_multidraft
    ) for entry in tqdm(ds)]

    ds_shard_size = 100
    ds_shards = [
        formatted_data[i:i+ds_shard_size]
        for i in range(0, len(formatted_data), ds_shard_size)
    ]

    def count_tokens_shard(ds_shard):
        return [count_tokens(x) for x in ds_shard]
    
    length_shards = Parallel(n_jobs=min(40, len(ds_shards)), verbose=10)(
        delayed(count_tokens_shard)(ds_shard) for ds_shard in ds_shards
    )

    # Flatten ds_shards and length_shards
    ds_raw_ls = []
    total_length_ls = []
    for ds_shard, length_shard in zip(ds_shards, length_shards):
        for x, y in zip(ds_shard, length_shard):
            x.update(y)
        ds_raw_ls.extend(ds_shard)
        total_length_ls.extend([x['prompt_n_tokens'] + x['response_n_tokens'] for x in ds_shard])
    
    # Convert raw data to dataset
    ds_raw = datasets.Dataset.from_list(ds_raw_ls)
    os.makedirs(output_ds_path, exist_ok=True)

    # Filter out long data
    # Get all indices of total length that is within the limit
    valid_indices = [i for i, x in enumerate(total_length_ls) if x <= config['length_limit']]
    filtered_ds_raw = ds_raw.select(valid_indices)
    
    # Shuffle the filtered dataset
    print(f"Shuffling the filtered dataset with seed {config['mixing_seed']}")
    filtered_ds_raw = filtered_ds_raw.shuffle(seed=config["mixing_seed"])
    filtered_ds_raw.to_parquet(os.path.join(output_ds_path, f'filtered_data_lmax{config["length_limit"]}.parquet'))
    ds2gpt_style_jsonl(filtered_ds_raw, os.path.join(output_ds_path, f'filtered_data_lmax{config["length_limit"]}.jsonl'))
    
    visualize_data(ds_raw, visualization_path=f'{output_ds_path}/raw_token_distribution_analysis.png')
    visualize_data(filtered_ds_raw, visualization_path=f'{output_ds_path}/filtered_token_distribution_analysis.png')

    example_ds = filtered_ds_raw.select(np.random.choice(len(filtered_ds_raw), 100, replace=False))
    example_ds.to_parquet(os.path.join(output_ds_path, f'filtered_data_lmax{config["length_limit"]}_example.parquet'))
    ds2gpt_style_jsonl(example_ds, os.path.join(output_ds_path, f'filtered_data_lmax{config["length_limit"]}_example.jsonl'))

if __name__ == "__main__":
    main()