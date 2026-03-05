export TRANSFORMERS_OFFLINE=True
export TIKTOKEN_ENCODINGS_BASE="/scratch/gpfs/ARORA/xz4134/Research_src/Aggregation/in-context-aggregation/misc/gpt-oss-harmony/encodings"

# Run the script with model alias framework
python3 ../utils/general_inference/vllm_serving.py \
    --model_config "GPT_OSS_20B" \
    --num_partitions 1 \
    --output_dir ./outputs/GPT_OSS_20B/init_sampling_verified \
    --n 1 \
    --batch_size 32768 \
    --data_path "../stage1_init_response_sampling/outputs/SR_Training/training_mix.ds" \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.95 \
    --seed 42 \
    --task_name "stepwise_verification" \
    --prompt_template_path "stepwise_verification_prompt_templates.json" \
    --prompt_template_key "stepwise_verification_template"

echo "Done with verification inference"

python3 stage2_postprocess.py --enable_stats --enable_viz -i ./outputs/GPT_OSS_20B/init_sampling_verified

echo "Done with postprocessing"