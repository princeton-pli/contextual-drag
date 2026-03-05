export TRANSFORMERS_OFFLINE=True

# Run the script with model alias framework
python3 ../utils/general_inference/vllm_serving.py \
    --model_config "LlamaR1_8B" \
    --num_partitions 1 \
    --output_dir ./outputs/SR_Training/LlamaR1_8B/init_sampling \
    --n 4 \
    --batch_size 16384 \
    --data_path "../stage0_dataset_preprocessing/bigmathrl_subset.ds" \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.95 \
    --seed 42 \
    --task_name "init_response" \
    --prompt_template_path "init_response_prompt_templates.json" \
    --prompt_template_key "qwen_math_prompt" 

echo "Done with inference"

python3 ../utils/verifiable_evaluation/math_eval/eval.py -f -s -d ./outputs/SR_Training/LlamaR1_8B/init_sampling -j 48

echo "Done with evaluation"
