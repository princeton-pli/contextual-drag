#!/bin/bash

MODELS=("GPT_OSS_120B" "GPT_OSS_20B" "Qwen3_32B" "Qwen3_8B" "Nemotron_32B" "Nemotron_7B" "LlamaR1_8B" "QwenR1_7B" "Llama3.1_70B" "Llama3.1_8B" "Qwen2.5_32B" "Qwen2.5_7B")
LARGE_MODELS=("Qwen3_32B" "Nemotron_32B" "Llama3.1_70B" "Qwen2.5_32B")

TASKS=("aime24" "aime25" "hmmt24" "hmmt25" "gpqa" "mmlu" "crux-i" "24-game")

NUM_PARTITIONS=1
N_SAMPLES=8

# Create logs directory if it doesn't exist
mkdir -p ./logs
launch_script_path="evals/eval.sh"

for MODEL in "${MODELS[@]}"; do

    if [[ " ${LARGE_MODELS[@]} " =~ " ${MODEL} " ]]; then
        NUM_GPUS=4
    else
        NUM_GPUS=1
    fi

    for TASK in "${TASKS[@]}"; do

        TEMPLATE_KEY="1f"
        if [ "$TASK" == "crux-i" ]; then
            TEMPLATE_KEY="1f_crux_input"
        elif [ "$TASK" == "gpqa" ] || [ "$TASK" == "mmlu" ]; then
            TEMPLATE_KEY="1f_qa_mc"
        fi

        JOB_NAME="${TASK}-1f-N${N_SAMPLES}-${MODEL}"
        echo "Starting ${JOB_NAME}"

        bash $launch_script_path \
            --task $TASK \
            --model $MODEL \
            --num-partitions $NUM_PARTITIONS \
            --n-samples $N_SAMPLES \
            --template-path "prompt_templates/1f_templates.json" \
            --template-key $TEMPLATE_KEY \
            --output-dir outputs/1f/${TASK}/${MODEL} \
            --data-path outputs/initial_sampling/${TASK}/1f.ds \
            --num-gpus $NUM_GPUS
    done
done
