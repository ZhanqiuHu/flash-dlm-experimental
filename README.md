

# FlashDLM: dLLM Acceleration with Guided Parallel Decoding and KV Caching

[![arXiv](https://img.shields.io/badge/arXiv-2505.21467v2-b31b1b.svg)](https://arxiv.org/abs/2505.21467v2)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion

FlashDLM implements guided diffusion for accelerating diffusion language model inference, as described in our paper ["FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion"](https://arxiv.org/abs/2505.21467v2).

FlashDLM achieves an average of **12.14x** end-to-end speedup on Dream-7B-Instruct model across various tasks with no or minimal accuracy loss.

**Note: This is experimental research code under development.**

## Demo

![Flash-DLM Demo](demo/flashdlm-demo.gif)

## Key Features

- **Guided Diffusion (Parallel Decoding)**: Using lightweight autoregressive model to choose safe-to-unmask tokens in diffusion language model 
- **KV Caching**: Default guided diffusion uses sliding window caching, KV projections within the sliding window are recomputed.
- **Evaluation**: Built-in evaluation scripts for GSM8K and other benchmarks (coming soon)

## Installation

### Minimal Installation
```bash
# Create a new conda environment
conda create --name flash-dlm-test python=3.11
conda activate flash-dlm-test

# Install minimal requirements
pip install -r requirements_minimal.txt
```

## Usage

### Command-line API (no Gradio)

A minimal script to call the model directly without the Gradio UI is provided in `flash_dlm_api.py`.

Basic usage (uses a default prompt):

```bash
python flash_dlm_api.py
```

Custom prompt:

```bash
python flash_dlm_api.py "Explain photosynthesis in simple terms."
```

Options:

- `--max-new-tokens`: maximum generated tokens (default: 128)
- `--temperature`: sampling temperature for the draft model (default: 0.2)
- `--sampling-strategy`: token matching strategy for verification, one of `deterministic` or `topk_relative` (default: `deterministic`). When `topk_relative` is selected, the verifier uses top-2 with a relative threshold of 0.5.
- `--verbose`: print generation stats

Example with options:

```bash
python flash_dlm_api.py "Write a short poem about autumn" --sampling-strategy topk_relative --max-new-tokens 256 --verbose
```

Notes:
- Default models: Dream `Dream-org/Dream-Flash-Instruct-7B` and AR `Qwen/Qwen2.5-1.5B-Instruct`. Override via `--dream-model` and `--ar-model`.

### Example: Running GSM8K evaluation with Dream Flash model

```bash
python guided_diffusion/dream_eval/gsm8k_guided_evaluator.py --config test_configs/dream/gsm8k/guided_diffusion/<config-file>.yaml
```

## Citation

If you use this work, please cite our paper:

```bibtex
@article{hu2025accelerating,
  title={FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion},
  author={Hu, Zhanqiu and Meng, Jian and Akhauri, Yash and Abdelfattah, Mohamed S. and Seo, Jae-sun and Zhang, Zhiru and Gupta, Udit},
  journal={arXiv preprint arXiv-2505.21467v2},
  year={2025},
  url={https://arxiv.org/abs/2505.21467v2}
}
```
