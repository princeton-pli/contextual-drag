"""Output and file management utilities for vLLM serving."""

import json
from pathlib import Path
from typing import Set, Dict, Any, Optional


def get_output_filename(output_dir: str, dataset_name: str, partition_id: int, num_partitions: int) -> str:
    """Generate output filename based on dataset and partition info"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if num_partitions > 1:
        filename = f"{dataset_name}-{partition_id}of{num_partitions}.jsonl"
    else:
        filename = f"{dataset_name}.jsonl"
    
    return str(output_dir / filename)


def save_result_to_jsonl(result: Dict[str, Any], output_file: str) -> None:
    """Append a single result to JSONL file"""
    with open(output_file, 'a') as f:
        json.dump(result, f)
        f.write('\n')


def load_processed_questions(output_path: str, question_column: str = "question") -> Set[str]:
    """Load already processed questions from existing JSONL file"""
    processed_questions = set()
    output_path = Path(output_path)
    
    if not output_path.exists():
        return processed_questions
    
    try:
        with output_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line.strip())
                    # Try multiple ways to get the question identifier
                    question = result.get(question_column) or result.get('question') or result.get('problem')
                    if question:
                        processed_questions.add(question)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                    continue
    except Exception as e:
        print(f"Warning: Error reading existing file {output_path}: {e}")
        return set()
    
    return processed_questions


def initialize_output_file(output_path: str, resume: bool = False) -> None:
    """Initialize output file (create empty file or validate existing for resume)"""
    output_path = Path(output_path)
    if resume and output_path.exists():
        print(f"Resuming from existing file: {output_path}")
        # Validate file can be read
        try:
            with output_path.open("r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            print(f"Found {line_count} existing results in output file")
        except Exception as e:
            print(f"Warning: Error reading existing file: {e}")
            print("Creating new file...")
            with output_path.open("w", encoding="utf-8") as f:
                pass
    else:
        # Create new empty file
        with output_path.open("w", encoding="utf-8") as f:
            pass
        print(f"Initialized new output file: {output_path}")


def create_generation_result(
        output,
        dataset_row: Dict[str, Any],
        args,
        config: Dict[str, Any],
        partition_id: int,
        enable_thinking: bool,
        task_name: str
    ) -> Dict[str, Any]:
    
    """Create a standardized result dictionary for a single generation from HF dataset row"""
    
    # Preserve all original columns from the HF dataset row
    original_data = dict(dataset_row) if dataset_row else {}
    
    # Multiple responses format
    responses = []
    for j, response_output in enumerate(output.outputs):
        response_data = {
            "response_id": j,
            "generated_response": response_output.text,
            "finish_reason": response_output.finish_reason,
        }
        responses.append(response_data)
    
    result = {
        f"{task_name}_generations": responses,
        f"{task_name}_prompt": output.prompt,
        **original_data,  # Preserve all original columns from the dataset
        f"{task_name}_generations_metadata": {
            "num_responses": len(responses),
            "template_path": getattr(args, 'prompt_template_path', None),
            "template_key": getattr(args, 'prompt_template_key', None),
            "enable_thinking": enable_thinking,
            "partition_id": partition_id,
            "num_partitions": args.num_partitions,
            "model_config_alias": config["model_config_alias"]
        }
    }
    
    return result
