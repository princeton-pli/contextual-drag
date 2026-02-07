MODELS=("GPT_OSS_20B" "Qwen3_8B" "Nemotron_7B" "GPT_OSS_120B" "Qwen3_32B" "Nemotron_32B")
for model_name in "${MODELS[@]}"; do
    python edit_distance_analysis.py --model_name "$model_name" \
    --anchored_data_template "../../outputs/1f/24-game/{model_name}/evaluated_{model_name}_flattened.jsonl" \
    --init_response_data_template "../../outputs/initial_sampling/24-game/{model_name}/evaluated_{model_name}_flattened.jsonl" \
    --metric "tree"
done