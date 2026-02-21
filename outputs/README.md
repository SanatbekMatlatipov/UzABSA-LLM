# Output Directory

This directory stores training outputs including:

- Model checkpoints
- LoRA adapters
- Merged models
- Training logs
- Evaluation results

## Structure

After training, each run creates a timestamped subdirectory:

```
outputs/
├── uzabsa_qwen2.5-7b_20240301_120000/
│   ├── lora_adapters/          # LoRA adapter weights
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── merged_model/           # Full merged model
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer files...
│   ├── checkpoint-100/         # Training checkpoints
│   ├── checkpoint-200/
│   └── training_*.log          # Training logs
└── evaluation_results/
    └── eval_results_*.json     # Evaluation results
```

## Note

Large model files are excluded from git via `.gitignore`.
