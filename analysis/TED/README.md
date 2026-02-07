# Tree Edit Distance (TED) Analysis

This folder contains scripts for analyzing the Tree Edit Distance between draft responses and anchored responses. (Results used in Section 2.2 and Section 3)

## Usage

### 1. Compute Edit Distances
Run `edit_distance_analysis.py` to compute and cache edit distance metrics (Levenshtein, Tree, and Binary) for specified models.

```bash
python edit_distance_analysis.py --model_name "MODEL_NAME"
```

### 2. Visualize Results
Run `visualize_anchored_main.py` to generate comparison plots from the cached data.

```bash
python visualize_anchored_main.py
```

Results are saved in the `figures/` directory.
