# Contextual Drag: How Errors in the Context Affect LLM Reasoning

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.04288-b31b1b.svg?style=flat)](https://arxiv.org/abs/2602.04288)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Email me](https://img.shields.io/badge/Contact-6fcf97?logo=gmail)](mailto:yuncheng@princeton.com)


<div>
  <img width="90%" src=assets/context-drag-overview.png>
</div>
</div>

Central to many self-improvement pipelines for large language models (LLMs) is the assumption that models can improve by reflecting on past mistakes. We study a phenomenon termed *contextual drag*: the presence of failed attempts in the context biases subsequent generations toward structurally similar errors.

Across evaluations of 11 proprietary and open-weight models on 8 reasoning tasks, contextual drag induces 10-20% performance drops, and iterative self-refinement in models with severe contextual drag can collapse into self-deterioration. Structural analysis using tree edit distance reveals that subsequent reasoning trajectories inherit structurally similar error patterns from the context.

We demonstrate that neither external feedback nor successful self-verification suffices to eliminate this effect. While mitigation strategies such as fallback-behavior fine-tuning and context denoising yield partial improvements, they fail to fully restore baseline performance, positioning contextual drag as a persistent failure mode in current reasoning architectures

Authors: [Yun Cheng](https://kapikantzari.github.io/) (`yuncheng at princeton dot edu`), [Xingyu Zhu](https://ultimatejupiter.github.io/), [Haoyu Zhao](https://hyzhao.me/), [Sanjeev Arora](https://www.cs.princeton.edu/~arora/)


## 1. Evaluating Contextual Drag on Benchmarks

To reproduce the contextual drag evaluations, follow these steps to prepare the data and run the inference scripts.

**Step 0: Build environments**
Install the required dependencies:
```bash
pip install -r requirements.txt
```

**Step 1: Get initial responses**

Generate a pool of initial model responses for each benchmark. This step is necessary to identify correct and incorrect reasoning paths for subsequent context construction:
```bash
bash data_generation/initial_sampling.sh
```
*Note: This script performs vLLM-based inference, evaluates the results, and post-processes them into a flattened dataset format.*

**Step 2: Generate Contextual Drag Datasets**

Construct datasets where the model is presented with 2 failed attempts (2F) or 2 successful attempts (2T) in its context:
```bash
# Generate datasets with 2 failed attempts in context
bash data_generation/2f_datagen/datagen.sh

# Generate datasets with 2 successful attempts in context
bash data_generation/2t_datagen/datagen.sh
```

To evaluate the effect of a single in context draft (1T/1F), use the flattening script to downsample and reformat the 2F/2T datasets e.g.:
```bash
bash data_generation/minimal_aggregate_flatten.sh ../outputs/2f
```

### Run evaluations (Main Table)

Once the datasets are generated, run the main evaluations to measure how these contexts "drag" the model's performance. These scripts iterate through multiple models and tasks to produce the results shown in the paper's main table:
```bash
# Evaluate performance with 1 attempt in context
bash evals/1f.sh

# Evaluate performance with 2 attempts in context
bash evals/2f.sh
```

## 2. Quantitative Analysis of Contextual Drag

### Tree-Edit Distance (TED) Analysis (section 2.3)
We use Tree Edit Distance to quantify the structural inheritance of errors. This analysis reveals how subsequent reasoning trajectories mimic the logical structure of the errors provided in the context.
- **Location**: [`analysis/TED`](analysis/TED)
- **Key Scripts**: `edit_distance_analysis.py` (computation) and `visualize_anchored_main.py` (plotting).
- See [analysis/TED/README.md](analysis/TED/README.md) for detailed usage.

### Error Signal Conditioning (section 3)
This analysis investigates whether model awareness of an error (either through external prompting or self-detection) mitigates the contextual drag effect.
- **Location**: [`analysis/ErrorSignalConditioning`](analysis/ErrorSignalConditioning)
- **Key Scripts**: `analysis.py` (metrics generation) and `visualize_drop.py` (heatmap generation).
- See [analysis/ErrorSignalConditioning/README.md](analysis/ErrorSignalConditioning/README.md) for details.

## 3. Attempts for Mitigating Contextual Drag

### Contextual Denoising (section 4.1)
We explore context denoising as a strategy to mitigate the impact of failed attempts by filtering or re-weighting the context.
- See [context_denoising/README.md](context_denoising/README.md) for more information.

### Mitigation via SFT (Fallback-behavior Fine-tuning) (section 4.2)
We demonstrate a mitigation strategy using Supervised Fine-Tuning (SFT) on synthetic reasoning traces to encourage "fallback" behavior when errors are detected in the context.
- **Location**: [`mitigation_sft`](mitigation_sft)
- See [mitigation_sft/README.md](mitigation_sft/README.md) for the complete data generation and training pipeline.

## Citation
```bibtex
@misc{cheng2026contextualdrag,
      title={Contextual Drag: How Errors in the Context Affect LLM Reasoning}, 
      author={Yun Cheng and Xingyu Zhu and Haoyu Zhao and Sanjeev Arora},
      year={2026},
      eprint={2602.04288},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.04288}, 
}
```