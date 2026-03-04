#!/bin/bash

MODELS=("GPT_OSS_120B" "GPT_OSS_20B" "Nemotron_7B" "Nemotron_32B" "Qwen3_8B" "Qwen3_32B")
LARGE_MODELS=("Qwen3_32B" "Nemotron_32B" "Llama3.1_70B" "Qwen2.5_32B")
TASKS=("aime24" "aime25" "hmmt24" "hmmt25" "gpqa" "mmlu" "crux-i" "24-game")

declare -A STEPS_MAP=( ["1"]="revise1" ["2"]="solve" )
STEP=${STEPS_MAP[$1]}
NUM_PARTITIONS=1

if [ "$STEP" == "solve" ]; then
    N_SAMPLES=8
else
    N_SAMPLES=1
fi

# Create logs directory if it doesn't exist
mkdir -p ./logs
launch_script_path="context_denoising/eval.sh"

for MODEL in "${MODELS[@]}"; do
    
    if [[ " ${LARGE_MODELS[@]} " =~ " ${MODEL} " ]]; then
        NUM_GPUS=4
    else
        NUM_GPUS=1
    fi

    for TASK in "${TASKS[@]}"; do

        EVAL_DIR="context_denoising/revise1/${STEP}/${TASK}/${MODEL}"

        case "$STEP" in
            solve)
                DATA_PATH="context_denoising/revise1/revise1/${TASK}/${MODEL}/processed_flattened_init_responses.ds"
                case "$TASK" in
                    crux-i)  TEMPLATE_KEY="filter1_${STEP}_crux_input" ;;
                    gpqa|mmlu) TEMPLATE_KEY="filter1_${STEP}_qa_mc" ;;
                    *)       TEMPLATE_KEY="filter1_${STEP}" ;;
                esac
                S_FLAG=""
                [ "$NUM_PARTITIONS" == "1" ] && S_FLAG="-s "
                case "$TASK" in
                    crux-i)
                        EVAL_COMMAND="utils/verifiable_evaluation/crux_eval/eval.py -f ${S_FLAG}-d $EVAL_DIR -r ${STEP}_response_generations"
                        ;;
                    24-game)
                        EVAL_COMMAND="utils/verifiable_evaluation/math_eval/eval.py -f ${S_FLAG}-d $EVAL_DIR -ep game_of_24 -r ${STEP}_response_generations"
                        ;;
                    *)
                        EVAL_COMMAND="utils/verifiable_evaluation/math_eval/eval.py -f ${S_FLAG}-d $EVAL_DIR -r ${STEP}_response_generations"
                        ;;
                esac
                ;;
            *)
                DATA_PATH="outputs/initial_sampling/${TASK}/1f.ds"
                case "$TASK" in
                    crux-i)  TEMPLATE_KEY="revise_revise1_${STEP}_crux_input" ;;
                    gpqa|mmlu) TEMPLATE_KEY="revise_revise1_${STEP}_qa_mc" ;;
                    *)       TEMPLATE_KEY="revise_revise1_${STEP}" ;;
                esac
                EVAL_COMMAND="utils/format_multiturn_data/postprocess.py -i $EVAL_DIR -t dataset.jsonl -n ${STEP}_response --response_final_column_name filtered_traj1"
                ;;
        esac

        JOB_NAME="${TASK}-revise1-${STEP}-N${N_SAMPLES}-${MODEL}"
        echo "Starting ${JOB_NAME}"

        bash $launch_script_path \
            --task $TASK \
            --model $MODEL \
            --num-partitions $NUM_PARTITIONS \
            --n-samples $N_SAMPLES \
            --template-path "prompt_templates/context_denoising_templates.json" \
            --template-key $TEMPLATE_KEY \
            --output-dir $EVAL_DIR \
            --data-path $DATA_PATH \
            --num-gpus $NUM_GPUS \
            --eval '$EVAL_COMMAND' \
            --task-name $STEP
    done
done