#!/bin/bash

# Default values
TASK=""
MODEL=""
NUM_PARTITIONS=""
N_SAMPLES=""
TEMPLATE_PATH=""
TEMPLATE_KEY=""
OUTPUT_DIR=""
DATA_PATH=""
NUM_GPUS=""
EVAL=""
TASK_NAME=""
PROCESS_COMMAND=""
GENERATE_COMMAND=""

# Function to display usage information
show_help() {
  cat << EOF
Usage: $0 [OPTIONS]

Options:
  --task TASK                    Task name (required)
  --model MODEL                  Model name (required)
  --num-partitions NUM           Number of partitions (required)
  --n-samples NUM                Number of samples (required)
  --template-path PATH           Template path (required)
  --template-key KEY             Template key (required)
  --output-dir DIR               Output directory (required)
  --data-path PATH               Data path (required)
  --num-gpus NUM                 Number of GPUs (required)
  -h, --help                     Show this help message

Example:
  $0 --task algebra --model GPT_OSS_120B --num-partitions 4 \\
     --n-samples 16 --template-key math_prompt \\
     --output-dir outputs/algebra/GPT_OSS_120B \\
     --data-path data/algebra/algebra.ds \\
     --num-gpus 1
EOF
}

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --task) TASK="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --num-partitions) NUM_PARTITIONS="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --template-path) TEMPLATE_PATH="$2"; shift 2 ;;
    --template-key) TEMPLATE_KEY="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --data-path) DATA_PATH="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --eval) EVAL="$2"; shift 2 ;;
    --task-name) TASK_NAME="$2"; shift 2 ;;
    --process-command) PROCESS_COMMAND="$2"; shift 2 ;;
    --generate-command) GENERATE_COMMAND="$2"; shift 2 ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Error: Unknown option '$1'"; echo ""; show_help; exit 1 ;;
  esac
done

# Validate required arguments
REQUIRED_VARS=("TASK" "MODEL" "NUM_PARTITIONS" "N_SAMPLES" "TEMPLATE_PATH" "TEMPLATE_KEY" "OUTPUT_DIR" "DATA_PATH" "NUM_GPUS")
MISSING_ARGS=()

for var in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!var}" ]]; then
    MISSING_ARGS+=("$var")
  fi
done

if [[ ${#MISSING_ARGS[@]} -gt 0 ]]; then
  echo "Error: Missing required arguments: ${MISSING_ARGS[*]}"
  echo ""
  show_help
  exit 1
fi

export HF_HOME=".cache"
export TRANSFORMERS_OFFLINE=True
export TIKTOKEN_ENCODINGS_BASE="misc/gpt-oss-harmony/encodings"

# Process the previous round's outputs
cd data_generation_scripts/big_math_rl/stage1_init_response_sampling || { echo "Failed to change directory"; exit 1; }
python3 ${PROCESS_COMMAND}

cd data_generation_scripts/big_math_rl/stage3_generate_aggregation || { echo "Failed to change directory"; exit 1; }
python3 ${GENERATE_COMMAND}

PROJECT_DIR="."
cd $PROJECT_DIR || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

# Run the script with model alias framework
python3 utils/general_inference/vllm_serving.py \
    --model_config "$MODEL" \
    --num_partitions $NUM_PARTITIONS \
    --output_dir "$OUTPUT_DIR" \
    --n $N_SAMPLES \
    --batch_size 16384 \
    --data_path "$DATA_PATH" \
    --tensor_parallel_size "$NUM_GPUS" \
    --gpu_memory_utilization 0.95 \
    --seed 42 \
    --task_name "${TASK_NAME}_response" \
    --prompt_template_path "$TEMPLATE_PATH" \
    --prompt_template_key "$TEMPLATE_KEY"


# Evaluate
python3 ${EVAL}

echo "Job completed for partition $SLURM_ARRAY_TASK_ID"
