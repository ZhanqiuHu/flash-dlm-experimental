# Flash-DLM

Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion

**Note: This is experimental research code under development.**

## Usage

### Example: Running GSM8K evaluation with Dream Flash model

```bash
python guided_diffusion/dream_eval/gsm8k_spec_evaluator.py --config test_configs/dream/gsm8k/guided_diffusion/100samples/dream_flash_qwen2.5_1.5b_instruct_guided_diffusion_100samples.yaml
```

## Citation

If you use this work, please cite our paper:

```bibtex
@article{hu2025accelerating,
  title={Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion},
  author={Hu, Zhanqiu and Meng, Jian and Akhauri, Yash and Abdelfattah, Mohamed S. and Seo, Jae-sun and Zhang, Zhiru and Gupta, Udit},
  journal={arXiv preprint arXiv:2505.21467},
  year={2025},
  url={https://arxiv.org/abs/2505.21467}
}
```