---
base_model: unsloth/qwen2.5-7b-instruct-bnb-4bit
library_name: transformers
model_name: uzabsa_qwen2.5-7b_20260222_001629
tags:
- generated_from_trainer
- unsloth
- sft
- trl
licence: license
---

# Model Card for uzabsa_qwen2.5-7b_20260222_001629

This model is a fine-tuned version of [unsloth/qwen2.5-7b-instruct-bnb-4bit](https://huggingface.co/unsloth/qwen2.5-7b-instruct-bnb-4bit).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/s-matlatipov-national-university-of-uzbekistan-named-aft/uzabsa-llm/runs/9dvndnmk) 


This model was trained with SFT.

### Framework versions

- TRL: 0.23.0
- Transformers: 4.57.2
- Pytorch: 2.6.0+cu124
- Datasets: 4.3.0
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```