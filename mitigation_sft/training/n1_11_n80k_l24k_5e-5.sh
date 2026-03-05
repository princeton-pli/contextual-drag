DATASET=GPTOSS_SFT_n1_11_lmax24k_n80k
LR=5e-5

NUM_GPUS=8
MEMORY=$((NUM_GPUS * 100))G
NUM_CPUS=$((NUM_GPUS * 8))
SLURM_JOB_NAME="LFSFT_OSS20B_n1_11_n80k_l24k_5e-5"
LOG_PATH="logs/${SLURM_JOB_NAME}-%j.out"
ERROR_PATH="logs/${SLURM_JOB_NAME}-%j.err"

sbatch --export=ALL,DATASET=$DATASET,LR=$LR,GPUS_PER_NODE=$NUM_GPUS \
  --job-name=$SLURM_JOB_NAME \
  --output=$LOG_PATH \
  --error=$ERROR_PATH \
  --gres=gpu:$NUM_GPUS \
  --mem=$MEMORY \
  --cpus-per-task=$NUM_CPUS \
  run_single_node_slurm.sh
