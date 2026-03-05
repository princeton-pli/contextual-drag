# Contextual Drag Mitigation SFT

This repository contains the complete pipeline for mitigating contextual drag via SFT on synthetic reasoning traces (as discussed in section 4.2 of the paper). The process includes data generation and model training.

## Getting Started

### 1. Data Generation
To generate the training data, navigate to the `data_generaition` directory and run the following script:

```bash
cd data_generaition
bash run_all_stages.sh
```

### 2. Training Environment Setup
To install the required dependencies and set up the environment, run:

```bash
pip install -r requirements.txt
```

Note: Do not use this environment for data generation. Use it for training only! The Llama-Factory training framework specialized for GPT-OSS-20B (with flash attention 3 and liger kernel) is adopted from [this commit](https://github.com/Imbernoulli/LLaMA-Factory/commit/66808664146f9831c95f48a5c8f67f5ed159d92a).



### 3. Training
To launch the SFT training, run the provided training script:

```bash
bash training/n1_11_n80k_l24k_5e-5.sh
```

You will find the final checkpoints stored in `training/LLaMA-Factory/outputs`

---

## Data Generation Pipeline Details

The data generation pipeline creates a specialized dataset aimed at mitigating contextual drag. The pipeline consists of five stages:

### Stage 0: Dataset Preprocessing
- **Goal**: Sample problems with reasonable difficulty from the BigMathRL dataset.
- **Process**: Problems are filtered based on their solve rate (e.g., Llama-8B solve rate ≤ 0.5) to ensure they are sufficiently challenging. A balanced subsample is created across different mathematical sources.
- **Output**: `bigmathrl_subset.ds`

### Stage 1: Initial Response Sampling
- **Goal**: Run inference on multiple models and verify the correctness of their responses.
- **Process**: Inference is performed on 5 models (GPT-OSS 20B, Qwen3-8B, Nemotron-7B, LlamaR1-8B, and QwenR1-7B). Each model generates multiple trajectories per problem. Correctness is verified using a `math-verify` backend.
- **Postprocessing**: Combines outputs from all models, flattens the dataset, and prepares it for the next stage.
- **Output**: `training_mix.ds`

### Stage 2: Stepwise Verification (GPT-OSS 20B)
- **Goal**: Obtain verdicts from GPT-OSS 20B on the responses generated in Stage 1.
- **Process**: GPT-OSS 20B is used to evaluate the responses from Stage 1. The postprocessing step checks the accuracy of these verdicts by comparing them against the ground truth correctness obtained in Stage 1.
- **Output**: `processed_sv_outputs_flattened.ds`

### Stage 3: Data Aggregation
- **Goal**: Gather data into balanced settings for training.
- **Process**: Data is aggregated into two primary settings:
    - **40k 1T**: 40,000 samples where the response in context is correct.
    - **40k 1F**: 40,000 samples where the response in context is incorrect.
- **Constraints**: Only samples where GPT-OSS 20B provided a correct verdict and the response in context has a verifiable correctness are included.

### Stage 4: Format Training Data (Synthesis)
- **Goal**: Synthesize the final training data with stitched reasoning traces and fallback behavior.
- **Process**: Reasoning traces are synthesized based on the correctness of the response in context:
    - **If Incorrect**: Stitch GPT-OSS 20B's accurate verdict from Stage 2 with a correct reasoning trace from Stage 1.
    - **If Correct**: Stitch GPT-OSS 20B's accurate verdict with a reuse template to recycle the existing correct answer.
- **Configurations**: Stage 4 behavior is controlled by JSON configuration files (located in `configs/`). Key parameters include:
    - `total_size`: Total number of training samples (e.g., 80,000).
    - `length_limit`: Maximum token length for a sample (e.g., 24,576).
    - `tokenizer`: The tokenizer used for length filtering (e.g., `openai/gpt-oss-20b`).
    - `remap_sv_index`: Boolean flag to enable remapping of indices in the reasoning traces (e.g., "The solution" -> "The first solution").
    - `training_sources`: Paths and weights for the aggregated datasets from Stage 3.
    - `templates`: Specifies which prompt, reuse, and new solution templates to use from `sft_templates_final.json`.
- **Output**: Final SFT training data in JSONL format.
