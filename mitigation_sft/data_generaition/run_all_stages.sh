#!/bin/bash
set -e

# Stage 0: Sample problems from BigMathRL dataset
echo "=== Running Stage 0: Dataset Preprocessing ==="
cd stage0_dataset_preprocessing
python3 generate_subset.py
cd ..

# Stage 1: Run inference on 5 models and verify correctness
echo "=== Running Stage 1: Initial Response Sampling ==="
cd stage1_init_response_sampling
bash batch_exec.sh
# Postprocess after inference and math-verify to combine outputs
python3 stage1_postprocess.py -i outputs/SR_Training
cd ..

# Stage 2: GPT-OSS 20B verification on Stage 1 responses
echo "=== Running Stage 2: GPT-OSS 20B Verification ==="
cd stage2_stepwise_verification
bash scripts/gpt_oss_20B/run_verification.sh
cd ..

# Stage 3: Aggregation into 40k 1T and 40k 1F settings
echo "=== Running Stage 3: Data Aggregation ==="
cd stage3_aggregation
bash scripts/gpt_oss_20b/oss20b_40k_each.sh
cd ..

# Stage 4: Synthesize stitched reasoning traces
echo "=== Running Stage 4: Format Training Data ==="
cd stage4_format_training_data
bash scripts_final/gptoss/SFT_Mix_n1_11_lmax24k_n80k_gptoss.sh
cd ..

echo "=== Pipeline Completed Successfully! ==="
