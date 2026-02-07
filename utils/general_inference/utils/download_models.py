import json
import os
from huggingface_hub import snapshot_download

# Path to the eval_models_params.json file
PARAMS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "eval_models_params.json"
)

# Load model names from the JSON config
with open(PARAMS_PATH, "r") as f:
    model_params = json.load(f)

# Collect all unique model_id values
model_ids = set()
for model_cfg in model_params.values():
    if "model_name" in model_cfg:
        model_ids.add(model_cfg["model_name"])

# Download each model
for model_id in model_ids:
    print(f"Downloading: {model_id}")
    snapshot_download(
        repo_id=model_id,
        max_workers=16
        # Optionally, you can set local_dir=f"./models/{model_id.replace('/', '__')}"
    )