#!/bin/bash

# Default values
TASK=${TASK:-"aime24"}
MODEL=${MODEL:-"GPT_OSS_120B"}
NUM_PARTITIONS=${NUM_PARTITIONS:-1}
N_SAMPLES=${N_SAMPLES:-16}
TEMPLATE_KEY=${TEMPLATE_KEY:-"qwen_math_prompt"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/${TASK}/initial_sampling/${MODEL}"}
DATA_PATH=${DATA_PATH:-"data/${TASK}/${TASK}.ds"}
NUM_GPUS=${NUM_GPUS:-1}

export TRANSFORMERS_OFFLINE=True
export TIKTOKEN_ENCODINGS_BASE="misc/gpt-oss-harmony/encodings"

# Generate initial responses to problems
python3 general_inference/vllm_serving.py \
    --model_config "$MODEL" \
    --num_partitions $NUM_PARTITIONS \
    --output_dir "$OUTPUT_DIR" \
    --n $N_SAMPLES \
    --batch_size 16384 \
    --data_path "$DATA_PATH" \
    --tensor_parallel_size "$NUM_GPUS" \
    --gpu_memory_utilization 0.95 \
    --seed 42 \
    --task_name "init_response" \
    --prompt_template_path "prompt_templates/init_response_prompt_templates.json" \
    --prompt_template_key "$TEMPLATE_KEY"


# Evaluate
python verifiable_evaluation/math_eval/eval.py -f -s -d "$OUTPUT_DIR"

# Postprocess the output
python data_generation/initial_sampling_postprocess.py -i "$OUTPUT_DIR"