#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=900G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=ALL

set -euo pipefail

mkdir -p logs

echo "DATASET: $DATASET"
echo "LR: $LR"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# -------------------------
# Runtime-configurable params (override via env vars or sbatch --export)
# -------------------------
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
EPOCHS="${EPOCHS:-1.0}"
SAVE_STEPS="${SAVE_STEPS:-500}"
GRADIENT_ACCUM="${GRADIENT_ACCUM:-8}"
JOB_SUFFIX="${JOB_SUFFIX:-STRESS_TEST}"

MODEL_PATH="${MODEL_PATH:-unsloth/gpt-oss-20b-BF16}"
DATASET="${DATASET:-STRESS_TEST}"
LR="${LR:-5.0e-5}"
SCHEDULER="${SCHEDULER:-cosine}"

LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-../utils/LLaMA-Factory}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${LLAMA_FACTORY_DIR}/examples/deepspeed/ds_z3_config.json}"

MODEL_NAME_BASE="$(basename "$MODEL_PATH")"
MODEL_NAME_SANITIZED="$(echo "$MODEL_NAME_BASE" | sed 's/[^a-zA-Z0-9_-]/-/g')"
DATASET_SANITIZED="$(echo "$DATASET" | sed 's/[^a-zA-Z0-9_-]/-/g')"
JOB_NAME_RUNTIME="LFSFT_${DATASET_SANITIZED}_${MODEL_NAME_SANITIZED}_lr${LR}_${SCHEDULER}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${JOB_NAME_RUNTIME}}"

# --- Environment Setup ---
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$CUDA_HOME/bin:${PATH}"

# Workaround for libcurand if necessary
mkdir -p ~/local/cuda/lib64
ln -sf /usr/local/cuda-12.8/lib64/libcurand.so.10 ~/local/cuda/lib64/libcurand.so
export LD_LIBRARY_PATH=~/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=~/local/cuda/lib64:${LIBRARY_PATH:-}

echo "Started at $(date)"

set -x

cd "$LLAMA_FACTORY_DIR"

export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_DIR/wandb"
export DISABLE_VERSION_CHECK=1
export TRANSFORMERS_OFFLINE=True

torchrun \
  --nproc_per_node="$GPUS_PER_NODE" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  src/train.py \
  --model_name_or_path "$MODEL_PATH" \
  --trust_remote_code true \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --dataset "$DATASET" \
  --template gpt \
  --cutoff_len 24576 \
  --max_samples 100000000 \
  --overwrite_cache true \
  --preprocessing_num_workers 64 \
  --dataloader_num_workers 64 \
  --output_dir "$OUTPUT_DIR" \
  --logging_steps 2 \
  --save_steps "$SAVE_STEPS" \
  --plot_loss true \
  --overwrite_output_dir true \
  --save_only_model true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GRADIENT_ACCUM" \
  --learning_rate "$LR" \
  --num_train_epochs "$EPOCHS" \
  --lr_scheduler_type "$SCHEDULER" \
  --warmup_ratio 0.1 \
  --bf16 true \
  --ddp_timeout 180000000 \
  --flash_attn fa2 \
  --enable_liger_kernel true