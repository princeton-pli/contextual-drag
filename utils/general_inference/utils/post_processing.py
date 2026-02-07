from collections import Counter
import json
from glob import glob
import os
import argparse
from tqdm import tqdm
from datasets import Dataset

def parse_thinking_steps(response: str, prompt: str, max_response_length: int):

    if "<think>" not in response:
        non_thinking_response = response
        thinking_status = 'no_thinking'
    
    else:
        non_thinking_response = response.split("</think>")[-1]

        if response.count("<think>") != response.count("</think>"):
            thinking_status = 'malformed_thinking'
        else:
            thinking_status = 'parsable_thinking'
    
    if len(non_thinking_response) > max_response_length:
        non_thinking_response = non_thinking_response[:max_response_length]
        thinking_status = 'truncated_' + thinking_status
        # print(f"Truncated {thinking_status} response to {len(non_thinking_response)} characters")
    
    return non_thinking_response, thinking_status

def preprocess_entry(entry, max_response_length):

    necessary_keys = ['question', 'answer', 'generated_response']
    for key in necessary_keys:
        if key not in entry:
            raise ValueError(f"Missing key: {key}")

    # Further processing can be done here
    final_response, thinking_status = parse_thinking_steps(entry['generated_response'], max_response_length)
    entry['init_response_final'] = final_response
    entry['init_response_thinking_status'] = thinking_status
    return entry

def main():
    parser = argparse.ArgumentParser(description="Preprocess and flatten dataset")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Input directory containing JSONL files")
    parser.add_argument("--input_file_template", "-t", type=str, default="*/*/*flattened.jsonl", help="Input file pattern (default: dataset-*.jsonl)")
    parser.add_argument("--max_response_length", "-m", type=int, default=8192, help="Maximum response length")
    args = parser.parse_args()

    input_dir = args.input_dir
    max_response_length = args.max_response_length
    output_file = os.path.join(input_dir, "processed_flattened_init_responses.ds")

    print(f"\nPreprocessing dataset with max response length {max_response_length}")

    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory '{input_dir}' does not exist!")
        return

    all_files = glob(input_dir + "/" + args.input_file_template)
    if not all_files:
        print(f"ERROR: No dataset files found in '{input_dir}'!")
        return

    processed_entries = []
    for file_path in all_files:
        print(f"\nProcessing file: {file_path}")

        total_entries = 0
        thinking_parsing_status = []

        with open(file_path, 'r') as f:
            for line in f:
                total_entries += 1
                entry = json.loads(line)
                processed_entry = preprocess_entry(entry, max_response_length)
                processed_entries.append(processed_entry)

                thinking_parsing_status.append(processed_entry['init_response_thinking_status'])

        # Compute the percentage of entries with each thinking parsing status
        thinking_parsing_status_counts = Counter(thinking_parsing_status)
        total_entries = sum(thinking_parsing_status_counts.values())
        thinking_parsing_status_percentages = {status: count / total_entries * 100 for status, count in thinking_parsing_status_counts.items()}
        max_status_len = max(len(s) for s in thinking_parsing_status_percentages)
        for status, percentage in thinking_parsing_status_percentages.items():
            print(f"{status:<{max_status_len}} : {percentage:6.2f}%")

    # Save the processed entries to a hf dataset
    dataset = Dataset.from_list(processed_entries)
    dataset.save_to_disk(output_file)

    print(f"Preprocessing complete. Processed {len(processed_entries)} entries.")
    print(f"Output saved to '{output_file}'.")

if __name__ == "__main__":
    main()