from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any

import scriptconfig as scfg


@dataclass(frozen=True)
class CommandSpec:
    config_cls: type[scfg.DataConfig]
    module_name: str
    description: str
    extra_argv: tuple[str, ...] = ()


def _config_to_argv(config: scfg.DataConfig) -> list[str]:
    argv: list[str] = []
    for key, value in config.asdict().items():
        if value is None:
            continue
        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(option)
        elif isinstance(value, (list, tuple)):
            if value:
                argv.append(option)
                argv.extend(str(item) for item in value)
        else:
            argv.extend([option, str(value)])
    return argv


class InferenceRunCLI(scfg.DataConfig):
    model_config = scfg.Value("Qwen3_8B_Thinking", help="Model configuration alias.")
    config_path = scfg.Value(None, help="Optional path to model configuration JSON.")
    list_models = scfg.Value(False, isflag=True, help="List model configurations and exit.")
    tensor_parallel_size = scfg.Value(1, type=int, help="Number of GPUs for tensor parallelism.")
    gpu_memory_utilization = scfg.Value(0.95, type=float, help="GPU memory utilization ratio.")
    temperature = scfg.Value(None, type=float, help="Override sampling temperature.")
    top_p = scfg.Value(None, type=float, help="Override top-p.")
    top_k = scfg.Value(None, type=int, help="Override top-k.")
    max_tokens = scfg.Value(None, type=int, help="Override generation max tokens.")
    seed = scfg.Value(42, type=int, help="Sampling seed.")
    n = scfg.Value(1, type=int, help="Number of responses per prompt.")
    task_name = scfg.Value("inference", help="Generation task name.")
    data_path = scfg.Value(None, required=True, help="Path to the HF dataset directory.")
    output_dir = scfg.Value("./outputs", help="Directory to save generated responses.")
    batch_size = scfg.Value(32, type=int, help="Batch size for inference.")
    max_questions = scfg.Value(None, type=int, help="Optional cap on questions to process.")
    prompt_template_path = scfg.Value(None, required=True, help="Path to the prompt template JSON.")
    prompt_template_key = scfg.Value(None, required=True, help="Template key inside the template JSON.")
    num_partitions = scfg.Value(1, type=int, help="Total number of partitions.")
    partition_id = scfg.Value(None, type=int, help="Current partition id.")
    enable_thinking = scfg.Value(False, isflag=True, help="Enable thinking mode.")
    disable_thinking = scfg.Value(False, isflag=True, help="Disable thinking mode.")
    resume = scfg.Value(False, isflag=True, help="Resume generation from existing JSONL output.")
    no_resume = scfg.Value(False, isflag=True, help="Restart generation from scratch.")


class InferenceListModelsCLI(scfg.DataConfig):
    config_path = scfg.Value(None, help="Optional path to model configuration JSON.")


class EvalMathCLI(scfg.DataConfig):
    dataset_dir = scfg.Value(None, required=True, help="Directory containing evaluation dataset JSONL partitions.")
    single_partition = scfg.Value(False, isflag=True, help="Evaluate a single partition only.")
    output = scfg.Value(None, help="Optional output path.")
    n_jobs = scfg.Value(8, type=int, help="Number of parallel jobs.")
    flatten_dataset = scfg.Value(False, isflag=True, help="Save flattened evaluation output.")
    answer_column = scfg.Value("answer", help="Answer column name.")
    response_column = scfg.Value("init_response_generations", help="Response column name.")
    equivalent_parser = scfg.Value("math_verify", help="Verification backend.")
    data_format = scfg.Value(
        "general_inference",
        choices=["general_inference", "openai_api", "gemini_api"],
        help="Input data format.",
    )
    problem_data_path_root = scfg.Value(None, help="Problem dataset path template for API-backed evaluation.")


class EvalCruxCLI(scfg.DataConfig):
    dataset_dir = scfg.Value(None, required=True, help="Directory containing evaluation dataset JSONL partitions.")
    single_partition = scfg.Value(False, isflag=True, help="Evaluate a single partition only.")
    output = scfg.Value(None, help="Optional output path.")
    n_jobs = scfg.Value(8, type=int, help="Number of parallel jobs.")
    flatten_dataset = scfg.Value(False, isflag=True, help="Save flattened evaluation output.")
    response_column = scfg.Value("init_response_generations", help="Response column name.")


class InitialSamplingPostprocessCLI(scfg.DataConfig):
    input_dir = scfg.Value(None, required=True, help="Input directory containing flattened JSONL files.")
    input_file_template = scfg.Value("*/*/*flattened.jsonl", help="Glob pattern for input files.")
    max_response_length = scfg.Value(16384, type=int, help="Maximum retained response length.")


class MinimalAggregateFlattenCLI(scfg.DataConfig):
    input_ds_path = scfg.Value(None, required=True, help="Input dataset path.")
    output_ds_path = scfg.Value(None, help="Optional explicit output dataset path.")


class AggregateDataCLI(scfg.DataConfig):
    input_dir = scfg.Value("processed_flattened_outputs", help="Input dataset directory.")
    num_true = scfg.Value(0, type=int, help="Number of correct trajectories per sample.")
    num_false = scfg.Value(2, type=int, help="Number of incorrect trajectories per sample.")
    output_dir = scfg.Value(None, help="Output directory.")
    seed = scfg.Value(42, type=int, help="Random seed.")
    problem_id_column = scfg.Value("id", help="Problem id column.")
    init_response_correctness_column = scfg.Value(
        "init_response_generations_correctness", help="Correctness column name."
    )
    filter_init_response_completeness = scfg.Value(False, isflag=True, help="Require stop finish reason.")
    filter_init_response_parsable_thinking = scfg.Value(False, isflag=True, help="Require parsable thinking.")
    init_response_models = scfg.Value(
        ["Qwen3_8B_Thinking", "Qwen3_8B_NoThinking", "LlamaR1_8B", "Gemma3_4B", "Llama3.1_8B", "QwenR1_7B"],
        nargs="+",
        help="Model aliases to sample from.",
    )


class AggregateCruxDataCLI(scfg.DataConfig):
    input_dir = scfg.Value("processed_flattened_outputs", help="Input dataset directory.")
    num_true = scfg.Value(0, type=int, help="Number of correct trajectories per sample.")
    num_false = scfg.Value(2, type=int, help="Number of incorrect trajectories per sample.")
    output_dir = scfg.Value(None, help="Output directory.")
    seed = scfg.Value(42, type=int, help="Random seed.")
    problem_id_column = scfg.Value("id", help="Problem id column.")
    init_response_correctness_column = scfg.Value(
        "init_response_generations_correctness", help="Correctness column name."
    )
    filter_init_response_parsable_thinking = scfg.Value(False, isflag=True, help="Require parsable thinking.")
    data_split = scfg.Value("none", help="Data split selector.")
    init_response_models = scfg.Value(None, nargs="+", help="Model aliases to sample from.")
    filter_init_response_completeness = scfg.Value(False, isflag=True, help="Require stop finish reason.")
    split_root = scfg.Value(None, help="Directory containing split JSON files.")
    execution_mode = scfg.Value(None, choices=["workspace", "installed"], help="Explicit execution mode.")


class AggregateIterativeDataCLI(scfg.DataConfig):
    input_dir = scfg.Value("processed_flattened_outputs", help="Input dataset directory.")
    num = scfg.Value(0, type=int, help="Number of trajectories per sample.")
    output_dir = scfg.Value(None, help="Output directory.")
    seed = scfg.Value(42, type=int, help="Random seed.")
    problem_id_column = scfg.Value("id", help="Problem id column.")
    filter_init_response_completeness = scfg.Value(False, isflag=True, help="Require stop finish reason.")
    filter_init_response_parsable_thinking = scfg.Value(False, isflag=True, help="Require parsable thinking.")
    data_split = scfg.Value("none", help="Data split selector.")
    init_response_models = scfg.Value(
        ["Qwen3_8B_Thinking", "Qwen3_8B_NoThinking", "LlamaR1_8B", "Gemma3_4B", "Llama3.1_8B", "QwenR1_7B"],
        nargs="+",
        help="Model aliases to sample from.",
    )
    round_num = scfg.Value(0, type=int, help="Round number for recursive aggregation.")
    split_root = scfg.Value(None, help="Directory containing split JSON files.")
    execution_mode = scfg.Value(None, choices=["workspace", "installed"], help="Explicit execution mode.")


class Stage1PostprocessIterativeCLI(scfg.DataConfig):
    input_dir = scfg.Value(None, required=True, help="Input directory containing flattened JSONL files.")
    input_file_template = scfg.Value("*/*/*flattened.jsonl", help="Glob pattern for input files.")
    output_dir = scfg.Value(None, help="Optional output directory.")
    max_response_length = scfg.Value(16384, type=int, help="Maximum retained response length.")
    round_num = scfg.Value(0, type=int, help="Round number.")


COMMANDS: dict[tuple[str, ...], CommandSpec] = {
    ("inference", "run"): CommandSpec(InferenceRunCLI, "contextual_drag.inference.vllm_serving", "Run vLLM inference."),
    ("inference", "list-models"): CommandSpec(
        InferenceListModelsCLI,
        "contextual_drag.inference.vllm_serving",
        "List packaged model configs.",
        extra_argv=("--list_models",),
    ),
    ("eval", "math"): CommandSpec(EvalMathCLI, "contextual_drag.evaluation.math.eval", "Run math evaluation."),
    ("eval", "crux"): CommandSpec(EvalCruxCLI, "contextual_drag.evaluation.crux.eval", "Run crux evaluation."),
    ("data", "initial-sampling-postprocess"): CommandSpec(
        InitialSamplingPostprocessCLI,
        "contextual_drag.data.initial_sampling_postprocess",
        "Postprocess initial sampling outputs.",
    ),
    ("data", "minimal-aggregate-flatten"): CommandSpec(
        MinimalAggregateFlattenCLI,
        "contextual_drag.data.minimal_aggregate_flatten",
        "Flatten a T2 dataset into T1 variants.",
    ),
    ("data", "aggregate"): CommandSpec(
        AggregateDataCLI,
        "contextual_drag.data.aggregate_data",
        "Aggregate general benchmark data.",
    ),
    ("data", "aggregate-crux"): CommandSpec(
        AggregateCruxDataCLI,
        "contextual_drag.data.aggregate_crux_data",
        "Aggregate crux benchmark data.",
    ),
    ("data", "aggregate-iterative"): CommandSpec(
        AggregateIterativeDataCLI,
        "contextual_drag.data.aggregate_data_iterative",
        "Aggregate iterative benchmark data.",
    ),
    ("data", "stage1-postprocess-iterative"): CommandSpec(
        Stage1PostprocessIterativeCLI,
        "contextual_drag.data.stage1_postprocess_iterative",
        "Postprocess iterative stage1 outputs.",
    ),
}


def _overall_help() -> str:
    lines = [
        "Usage: python -m contextual_drag <group> <command> [options]",
        "",
        "Groups:",
        "  inference  run, list-models",
        "  eval       math, crux",
        "  data       initial-sampling-postprocess, minimal-aggregate-flatten,",
        "             aggregate, aggregate-crux, aggregate-iterative, stage1-postprocess-iterative",
        "",
        "Use `python -m contextual_drag <group> --help` to see commands in a group.",
        "Use `python -m contextual_drag <group> <command> --help` for command-specific help.",
    ]
    return "\n".join(lines)


def _group_help(group: str) -> str:
    lines = [f"Usage: python -m contextual_drag {group} <command> [options]", "", "Commands:"]
    for path, spec in COMMANDS.items():
        if path[0] == group:
            lines.append(f"  {path[1]:28} {spec.description}")
    return "\n".join(lines)


def _dispatch(spec: CommandSpec, argv: list[str]) -> int:
    config = spec.config_cls.cli(argv=argv, special_options=False, strict=True)
    forwarded_argv = _config_to_argv(config) + list(spec.extra_argv)
    module = importlib.import_module(spec.module_name)
    result = module.main(forwarded_argv)
    return int(result or 0)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(_overall_help())
        return 0

    group = argv[0]
    if len(argv) == 1 or argv[1] in {"-h", "--help"}:
        print(_group_help(group))
        return 0

    command = argv[1]
    spec = COMMANDS.get((group, command))
    if spec is None:
        print(_overall_help())
        return 1

    try:
        return _dispatch(spec, argv[2:])
    except SystemExit as ex:
        code = ex.code
        return int(code) if isinstance(code, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
