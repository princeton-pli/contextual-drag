init_response_model=(GPT_OSS_20B Qwen3_8B_Thinking Nemotron_7B LlamaR1_8B QwenR1_7B)
correct_response_model=GPT_OSS_20B

# T0 F1: 20000
python3 aggregate_data.py -i ../stage2_stepwise_verification/outputs/GPT_OSS_20B/init_sampling_verified/processed_sv_outputs_flattened.ds -T 0 -F 1 --filter_sv_correctness --filter_init_response_completeness --data_split sft --n_samples=40000 --init_response_models "${init_response_model[@]}" --correct_response_model "${correct_response_model}"
# T1 F0: 20000
python3 aggregate_data.py -i ../stage2_stepwise_verification/outputs/GPT_OSS_20B/init_sampling_verified/processed_sv_outputs_flattened.ds -T 1 -F 0 --filter_sv_correctness --filter_init_response_completeness --data_split sft --n_samples=40000 --init_response_models "${init_response_model[@]}" --correct_response_model "${correct_response_model}"