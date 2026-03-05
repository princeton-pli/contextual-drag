import json
import re
from tqdm import tqdm
from datasets import Dataset


def parse_stepwise_verification_output(entry: dict) -> tuple[str, str, bool]:
    """
    Parse the stepwise verification output from a dataset entry.

    Args:
        entry: Dataset entry containing verification response

    Returns:
        tuple: (boolean_verdict, stepwise_verification_verdict, stepwise_verification_correctness)
    """
    response_key = "stepwise_verification_generations_generated_response"
    if response_key not in entry:
        raise ValueError(f"Key {response_key} not found in entry")
    response = entry[response_key]
    s_lower = response.lower()
    # Remove all symbols (e.g. *, #, etc.)
    # s_lower = re.sub(r'[^a-zA-Z0-9\s\n]', '', s_lower)

    verified_correctness = entry["init_response_generations_correctness"]

    verdict_pattern = r"<overall_verdict>\s*([^\n]*)</overall_verdict>"
    verdict_results = re.findall(verdict_pattern, s_lower, re.DOTALL)
    verdict_result = verdict_results[-1].strip() if verdict_results else None

    # If no template match is found, return None
    if verdict_result is None:
        boolean_verdict = None
        stepwise_verification_verdict = "unparsable_verdict"
        stepwise_verification_correctness = False
        return boolean_verdict, stepwise_verification_verdict, stepwise_verification_correctness

    # Remove all symbols (e.g. *, #, etc.) from the verdict_result
    verdict_result = re.sub(r'[^\w\s]', '', verdict_result)
    verdict_result = verdict_result.split('overall_verdict')[-1].strip()
    verdict_result = verdict_result.split('overall_confidence')[0].strip()
    boolean_verdict = verdict_result == "correct"

    if verdict_result == "correct":
        stepwise_verification_verdict = "correct"
    elif verdict_result == "incorrect":
        stepwise_verification_verdict = "incorrect"
    elif "partial" in verdict_result:
        stepwise_verification_verdict = "partially_correct"
    else:
        stepwise_verification_verdict = "parsable_ood_verdict"
        print(f"Unparsable verdict: {verdict_result}")

    stepwise_verification_correctness = boolean_verdict == verified_correctness

    return boolean_verdict, stepwise_verification_verdict, stepwise_verification_correctness


def flatten_generation_output(entry, flatten_key):
    """
    Flatten a generation output entry based on a specified key.

    Args:
        entry: Dataset entry to flatten
        flatten_key: Key containing the generations to flatten

    Returns:
        list: List of flattened entries
    """
    if flatten_key not in entry:
        raise ValueError(f"Key {flatten_key} not found in entry")
    generations = entry.pop(flatten_key)
    flattened_generations = []
    for generation in generations:
        generation_flattened = {f"{flatten_key}_{k}": v for k, v in generation.items()}
        flattened_generations.append({**entry, **generation_flattened})
        break
    return flattened_generations


def process_jsonl_file(file_name, flatten_key, data_processing_func):
    """
    Process a single JSONL file and flatten its entries.

    Args:
        file_name: Path to the JSONL file
        flatten_key: Key to flatten in each entry
        data_processing_func: Function to process each entry

    Returns:
        list: List of processed and flattened entries
    """
    total_flattened_entries = []

    with open(file_name, 'r') as f:
        for line in tqdm(f):
            entry = json.loads(line)
            flattened_entries = data_processing_func(entry, flatten_key)

            for flattened_entry in flattened_entries:
                boolean_verdict, stepwise_verification_verdict, stepwise_verification_correctness = parse_stepwise_verification_output(flattened_entry)
                flattened_entry["stepwise_verification_verdict"] = stepwise_verification_verdict
                flattened_entry["stepwise_verification_boolean_verdict"] = boolean_verdict
                flattened_entry["stepwise_verification_correctness"] = stepwise_verification_correctness
            total_flattened_entries.extend(flattened_entries)
    
    print(f"Converting to dataset for {file_name} with length {len(total_flattened_entries)}")
    flattened_ds = Dataset.from_list(total_flattened_entries)
    print(f"{file_name} converted to dataset with length {len(flattened_ds)}")

    return flattened_ds
