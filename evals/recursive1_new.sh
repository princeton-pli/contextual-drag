#!/bin/bash

MODEL="GPT_OSS_20B"
# MODEL="NewOSS_20B_n1_11_80k_DR80k_lr1e-5_e1"
LARGE_MODELS=("Qwen3_32B" "Nemotron_32B" "Llama3.1_70B" "Qwen2.5_32B")
# TASKS=("aime24" "aime25" "hmmt24" "hmmt25")
TASKS=("aime24")

NUM_PARTITIONS=1
N_SAMPLES=16
RUN_ID=$1

# Create logs directory if it doesn't exist
mkdir -p ./logs
launch_script_path="/scratch/gpfs/ARORA/yc6206/multi-llm/in-context-aggregation/evals_ablations/recursive_eval.sh"

for STEP in {1..15}; do
        
    if [[ " ${LARGE_MODELS[@]} " =~ " ${MODEL} " ]]; then
        NUM_GPUS=4
    else
        NUM_GPUS=1
    fi

    for TASK in "${TASKS[@]}"; do

        OUTPUT_BASE="outputs_ablations/iterative_${RUN_ID}"
        
        if [[ "$STEP" == "1" ]]; then
            INPUT_DIR="outputs/initial_sampling/${TASK}"
            ROUND=0
        else
            PREV_STEP=$((STEP - 1))
            INPUT_DIR="${OUTPUT_BASE}/step${PREV_STEP}/${TASK}"
        fi
        
        PROCESS_COMMAND="stage1_postprocess_recursive.py -t */*flattened.jsonl -i ${INPUT_DIR} -r ${STEP} -o ${OUTPUT_BASE}/step${STEP}/${TASK}"
        GENERATE_INPUT="${OUTPUT_BASE}/step${STEP}/${TASK}/processed_flattened_step${STEP}_responses.ds"
        GENERATE_COMMAND="aggregate_data_recursive.py -N 1 --init_response_models ${MODEL} --filter_init_response_completeness --filter_init_response_parsable_thinking --seed ${RUN_ID} -r ${STEP} -i ${GENERATE_INPUT}"

        JOB_NAME="${TASK}-iterative_${RUN_ID}-${STEP}-N${N_SAMPLES}-${MODEL}"

        echo "Submitting job for ${JOB_NAME}"


        DATA_PATH="${OUTPUT_BASE}/step${PREV_STEP}/${TASK}/minimal_aggregated_data_N1_step${STEP}.ds"

        TEMPLATE_KEY="1f"
        if [ "$TASK" == "crux-i" ]; then
            TEMPLATE_KEY="1f_crux_input"
        elif [ "$TASK" == "gpqa" ] || [ "$TASK" == "mmlu" ]; then
            TEMPLATE_KEY="1f_qa_mc"
        fi

        if [ "$NUM_PARTITIONS" == "1" ]; then
            if [ "$TASK" == "crux-i" ]; then
                EVAL_COMMAND="data_generation_scripts/big_math_rl/verifiable_evaluation/crux_eval/eval.py -f -s -d outputs_ablations/recursive1_retry2_${RUN_ID}/round${STEP}/${TASK}/${MODEL} -r round${STEP}_response_generations" 
            elif [ "$TASK" == "24-game" ]; then
                EVAL_COMMAND="data_generation_scripts/big_math_rl/verifiable_evaluation/math_eval/eval.py -f -s -d outputs_ablations/recursive1_retry2_${RUN_ID}/round${STEP}/${TASK}/${MODEL} -ep game_of_24 -r round${STEP}_response_generations"
            else
                EVAL_COMMAND="data_generation_scripts/big_math_rl/verifiable_evaluation/math_eval/eval.py -f -s -d outputs_ablations/recursive1_retry2_${RUN_ID}/round${STEP}/${TASK}/${MODEL} -r round${STEP}_response_generations"
            fi
        elif [ "$TASK" == "24-game" ]; then
            EVAL_COMMAND="data_generation_scripts/big_math_rl/verifiable_evaluation/math_eval/eval.py -f -d outputs_ablations/recursive1_retry2_${RUN_ID}/round${STEP}/${TASK}/${MODEL} -ep game_of_24 -r round${STEP}_response_generations"
        elif [ "$TASK" == "crux-i" ]; then
            EVAL_COMMAND="data_generation_scripts/big_math_rl/verifiable_evaluation/crux_eval/eval.py -f -d outputs_ablations/recursive1_retry2_${RUN_ID}/round${STEP}/${TASK}/${MODEL} -r round${STEP}_response_generations"
        else
            EVAL_COMMAND="data_generation_scripts/big_math_rl/verifiable_evaluation/math_eval/eval.py -f -d outputs_ablations/recursive1_retry2_${RUN_ID}/round${STEP}/${TASK}/${MODEL} -r round${STEP}_response_generations"
        fi

        if [ "$STEP" == "1" ]; then              
            prev_jid=$(sbatch \
                --job-name="$JOB_NAME" \
                --gres="gpu:$NUM_GPUS" \
                --cpus-per-task=8 \
                --mem="100G" \
                --time="2:00:00" \
                --partition=pli-c \
                --array=0-$((NUM_PARTITIONS - 1)) \
                --output="./logs/${JOB_NAME}-%A_%a.out" \
                --wrap="bash $launch_script_path \
                    --task $TASK \
                    --model $MODEL \
                    --num-partitions $NUM_PARTITIONS \
                    --n-samples $N_SAMPLES \
                    --template-path "data_generation_scripts/big_math_rl/stage4_format_training_data/sft_templates.json" \
                    --template-key $TEMPLATE_KEY \
                    --output-dir outputs_ablations/recursive1_retry2_${RUN_ID}/round${STEP}/${TASK}/${MODEL} \
                    --data-path $DATA_PATH \
                    --num-gpus $NUM_GPUS \
                    --eval '$EVAL_COMMAND' \
                    --task-name round${STEP} \
                    --process-command '$PROCESS_COMMAND' \
                    --generate-command '$GENERATE_COMMAND'" | awk '{print $4}')
        else
            prev_jid=$(sbatch \
                --job-name="$JOB_NAME" \
                --gres="gpu:$NUM_GPUS" \
                --cpus-per-task=8 \
                --mem="100G" \
                --time="2:00:00" \
                --partition=pli-c \
                --array=0-$((NUM_PARTITIONS - 1)) \
                --output="./logs/${JOB_NAME}-%A_%a.out" \
                --dependency=afterok:${prev_jid} \
                --wrap="bash $launch_script_path \
                    --task $TASK \
                    --model $MODEL \
                    --num-partitions $NUM_PARTITIONS \
                    --n-samples $N_SAMPLES \
                    --template-path "data_generation_scripts/big_math_rl/stage4_format_training_data/sft_templates.json" \
                    --template-key $TEMPLATE_KEY \
                    --output-dir outputs_ablations/recursive1_retry2_${RUN_ID}/round${STEP}/${TASK}/${MODEL} \
                    --data-path $DATA_PATH \
                    --num-gpus $NUM_GPUS \
                    --eval '$EVAL_COMMAND' \
                    --task-name round${STEP} \
                    --process-command '$PROCESS_COMMAND' \
                    --generate-command '$GENERATE_COMMAND'" | awk '{print $4}')
        fi
    done
done
