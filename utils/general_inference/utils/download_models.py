from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from contextual_drag.config.resources import inference_model_config_resource


model_params = inference_model_config_resource()

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
