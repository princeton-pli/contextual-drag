# Error Signal Conditioning Analysis

This folder contains scripts for analyzing how error signals (both prompted and self-detected) condition the model's performance. (Results used in Section 3)

## Usage

### 1. Run Analysis
Run `analysis.py` to process the evaluation results and generate the conditioning metrics (we parse the model's verification results on the context, and compute the accuracy conditioned on the correctly verified ones).

```bash
python analysis.py
```

You can specify a different output path using `--output-json`:
```bash
python analysis.py --output-json correct_verification_conditioning_results_prompted.json
```

### 2. Visualize Results
Run `visualize_drop.py` to generate heatmap visualizations comparing prompted and self-detected error awareness.

```bash
python visualize_drop.py
```

The visualization is saved as `awareness_visualization.pdf`.
