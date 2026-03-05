# bench_anchor_models = {
#             'aime24': ['nemotron_32b', 'nemotron_7b', 'qwen3_32b'],
#             'aime25': ['gpt_oss_120b', 'nemotron_32b', 'nemotron_7b'],
#             'hmmt24': ['gpt_oss_120b', 'nemotron_32b', 'nemotron_7b'],
#             'hmmt25': ['nemotron_32b', 'nemotron_7b', 'gpt_oss_120b'],
#             'gpqa': ['gpt_oss_120b', 'nemotron_32b', 'qwen3_32b'],
#             'mmlu': ['nemotron_32b', 'qwen3_32b', 'gpt_oss_120b'],
#             'crux-i': ['nemotron_32b', 'qwen3_32b', 'qwen3_8b'],
#             '24-game': ['nemotron_32b', 'qwen3_32b', 'qwen3_8b'],
#         }

python aggregate_data_minimal.py -i outputs/initial_sampling/aime24/processed_flattened_init_responses.ds -T 2 -F 0 --init_response_models Nemotron_32B Nemotron_7B Qwen3_32B --filter_init_response_completeness --filter_init_response_parsable_thinking --output_dir outputs/2t/aime24/

python aggregate_data_minimal.py -i outputs/initial_sampling/aime25/processed_flattened_init_responses.ds -T 2 -F 0 --init_response_models GPT_OSS_120B Nemotron_32B Nemotron_7B --filter_init_response_completeness --filter_init_response_parsable_thinking --output_dir outputs/2t/aime25/

python aggregate_data_minimal.py -i outputs/initial_sampling/hmmt24/processed_flattened_init_responses.ds -T 2 -F 0 --init_response_models GPT_OSS_120B Nemotron_32B Nemotron_7B --filter_init_response_completeness --filter_init_response_parsable_thinking --output_dir outputs/2t/hmmt24/

python aggregate_data_minimal.py -i outputs/initial_sampling/hmmt25/processed_flattened_init_responses.ds -T 2 -F 0 --init_response_models Nemotron_32B Nemotron_7B GPT_OSS_120B --filter_init_response_completeness --filter_init_response_parsable_thinking --output_dir outputs/2t/hmmt25/

python aggregate_data_minimal.py -i outputs/initial_sampling/gpqa/processed_flattened_init_responses.ds -T 2 -F 0 --init_response_models GPT_OSS_120B Nemotron_32B Qwen3_32B --filter_init_response_completeness --filter_init_response_parsable_thinking --output_dir outputs/2t/gpqa/

python aggregate_data_minimal.py -i outputs/initial_sampling/mmlu/processed_flattened_init_responses.ds -T 2 -F 0 --init_response_models Nemotron_32B Qwen3_32B GPT_OSS_120B --filter_init_response_completeness --filter_init_response_parsable_thinking --output_dir outputs/2t/mmlu/

python aggregate_data_minimal.py -i outputs/initial_sampling/crux-i/processed_flattened_init_responses.ds -T 2 -F 0 --init_response_models Nemotron_32B Qwen3_32B Qwen3_8B --filter_init_response_completeness --filter_init_response_parsable_thinking --output_dir outputs/2t/crux-i/

python aggregate_data_minimal.py -i outputs/initial_sampling/24-game/processed_flattened_init_responses.ds -T 2 -F 0 --init_response_models Nemotron_32B Qwen3_32B Qwen3_8B --filter_init_response_completeness --filter_init_response_parsable_thinking --output_dir outputs/2t/24-game/