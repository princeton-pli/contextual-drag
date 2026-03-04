#!/bin/bash

MODEL="GPT_OSS_20B"
LARGE_MODELS=("Qwen3_32B" "Nemotron_32B" "Llama3.1_70B" "Qwen2.5_32B")
TASK="aime24"

NUM_PARTITIONS=1
N_SAMPLES=16
RUN_ID=$1

if [[ " ${LARGE_MODELS[@]} " =~ " ${MODEL} " ]]; then
    NUM_GPUS=4
else
    NUM_GPUS=1
fi

# Create logs directory if it doesn't exist
mkdir -p ./logs
launch_script_path="evals/iterative_eval.sh"

for STEP in {1..15}; do

    OUTPUT_BASE="outputs_ablations/iterative_${RUN_ID}"
    
    PREV_STEP=$((STEP - 1))
    if [[ "$STEP" == "1" ]]; then
        INPUT_DIR="outputs/initial_sampling/${TASK}"
    else
        INPUT_DIR="${OUTPUT_BASE}/step${PREV_STEP}/${TASK}"
    fi
        
    PROCESS_COMMAND="stage1_postprocess_iterative.py -t */*flattened.jsonl -i ${INPUT_DIR} -r ${STEP} -o ${OUTPUT_BASE}/step${STEP}/${TASK}"
    GENERATE_INPUT="${OUTPUT_BASE}/step${STEP}/${TASK}/processed_flattened_step${STEP}_responses.ds"
    GENERATE_COMMAND="aggregate_data_iterative.py -N 1 --init_response_models ${MODEL} --filter_init_response_completeness --filter_init_response_parsable_thinking --seed ${RUN_ID} -r ${STEP} -i ${GENERATE_INPUT}"

    DATA_PATH="${OUTPUT_BASE}/step${PREV_STEP}/${TASK}/step${STEP}.ds"

    TEMPLATE_KEY="1f"
    if [ "$TASK" == "crux-i" ]; then
        TEMPLATE_KEY="1f_crux_input"
    elif [ "$TASK" == "gpqa" ] || [ "$TASK" == "mmlu" ]; then
        TEMPLATE_KEY="1f_qa_mc"
    fi

    EVAL_DIR="outputs_ablations/iterative_${RUN_ID}/step${STEP}/${TASK}/${MODEL}"
    EVAL_S_FLAG=""
    if [ "$NUM_PARTITIONS" == "1" ]; then
        EVAL_S_FLAG="-s"
    fi
    EVAL_COMMAND="utils/verifiable_evaluation/math_eval/eval.py -f${EVAL_S_FLAG:+ ${EVAL_S_FLAG}} -d ${EVAL_DIR} -r step${STEP}_response_generations"

    JOB_NAME="${TASK}-iterative_${RUN_ID}-${STEP}-N${N_SAMPLES}-${MODEL}"
    echo "Starting ${JOB_NAME}"

    bash $launch_script_path \
        --task $TASK \
        --model $MODEL \
        --num-partitions $NUM_PARTITIONS \
        --n-samples $N_SAMPLES \
        --template-path prompt_templates/1f_templates.json \
        --template-key $TEMPLATE_KEY \
        --output-dir $EVAL_DIR \
        --data-path $DATA_PATH \
        --num-gpus $NUM_GPUS \
        --eval $EVAL_COMMAND \
        --task-name step${STEP} \
        --process-command $PROCESS_COMMAND \
        --generate-command $GENERATE_COMMAND
done
