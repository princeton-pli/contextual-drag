#!/bin/bash

MODELS=("GPT_OSS_120B" "Nemotron_32B" "Nemotron_7B")
LARGE_MODELS=("Qwen3_32B" "Nemotron_32B" "Llama3.1_70B" "Qwen2.5_32B")
TASKS=("hmmt25" "gpqa" "crux-i")

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

        EVAL_DIR="evals/position/${TASK}/${MODEL}"
        case "$TASK" in
            crux-i)  TEMPLATE_KEY="position_crux_input" ;;
            gpqa|mmlu) TEMPLATE_KEY="position_qa_mc" ;;
            *)       TEMPLATE_KEY="position" ;;
        esac

        JOB_NAME="${TASK}-position-N${N_SAMPLES}-${MODEL}"
        echo "Starting ${JOB_NAME}"

        bash $launch_script_path \
            --task $TASK \
            --model $MODEL \
            --num-partitions $NUM_PARTITIONS \
            --n-samples $N_SAMPLES \
            --template-path "prompt_templates/ablation_templates.json" \
            --template-key $TEMPLATE_KEY \
            --output-dir $EVAL_DIR \
            --data-path outputs/initial_sampling/${TASK}/1f.ds \
            --num-gpus $NUM_GPUS
    done
done