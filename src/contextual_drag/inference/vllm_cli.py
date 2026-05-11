from __future__ import annotations

import scriptconfig as scfg


class InferenceRunCLI(scfg.DataConfig):
    __command__ = "run"

    model_config = scfg.Value("Qwen3_8B_Thinking", help="Model configuration alias.")
    config_path = scfg.Value(None, help="Optional path to model configuration JSON.")
    tensor_parallel_size = scfg.Value(1, type=int, help="Number of GPUs for tensor parallelism.")
    gpu_memory_utilization = scfg.Value(0.95, type=float, help="GPU memory utilization ratio.")
    temperature = scfg.Value(None, type=float, help="Override sampling temperature.")
    top_p = scfg.Value(None, type=float, help="Override top-p.")
    top_k = scfg.Value(None, type=int, help="Override top-k.")
    max_tokens = scfg.Value(None, type=int, help="Override generation max tokens.")
    seed = scfg.Value(42, type=int, help="Sampling seed.")
    n = scfg.Value(1, type=int, help="Number of responses per prompt.")
    task_name = scfg.Value(
        "init_response",
        help=(
            "Generation task name; output columns are written as "
            "`<task_name>_generations`, `<task_name>_prompt`, etc. "
            "Defaults to 'init_response' so that downstream `eval math` / `eval crux` "
            "find the column they expect without an explicit --response_column override."
        ),
    )
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
    resume = scfg.Value(True, isflag=True, help="Resume generation from existing JSONL output.")

    @classmethod
    def main(cls, argv=True, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.inference import vllm_serving

        return vllm_serving.main(args)


class InferenceListModelsCLI(scfg.DataConfig):
    __command__ = "list-models"

    config_path = scfg.Value(None, help="Optional path to model configuration JSON.")

    @classmethod
    def main(cls, argv=True, **kwargs):
        args = cls.cli(argv=argv, data=kwargs, strict=True, special_options=False)
        from contextual_drag.inference import vllm_serving

        return vllm_serving.list_model_configs(args)
