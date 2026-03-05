export TRANSFORMERS_OFFLINE=True

python3 sft_data_generation_oss_jsonl.py \
  --data_config_path=configs/sft_final_n1_11_80kmix_gptoss/SFT_Mix_n1_11_lmax24k_n80k.json \
  --template_path=sft_templates_final.json \
  --output_path=outputs