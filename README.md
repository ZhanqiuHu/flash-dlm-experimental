# Flash-DLM

[![arXiv](https://img.shields.io/badge/arXiv-2505.21467v2-b31b1b.svg)](https://arxiv.org/abs/2505.21467v2)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion

Flash-DLM implements guided diffusion for accelerating diffusion language model inference, as described in our paper ["FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion"](https://arxiv.org/abs/2505.21467v2).

**Note: This is experimental research code under development.**

## Key Features

- **Guided Diffusion**: Using lightweight autoregressive model to choose safe-to-unmask tokens in diffusion language model 
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
  journal={arXiv preprint arXiv:2505.21467v2},
  year={2025},
  url={https://arxiv.org/abs/2505.21467v2}
}
```