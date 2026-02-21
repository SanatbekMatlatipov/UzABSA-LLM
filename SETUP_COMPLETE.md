# Environment Setup Complete ✅

## Current Status

- ✅ Python 3.13.9 in `.venv` environment
- ✅ PyTorch 2.10.0 installed
- ✅ All dependencies loaded
- ✅ GPU config module working
- ✅ Training scripts ready
- ✅ Documentation updated (venv → .venv)

---

## Quick Start (Copy & Paste)

### 1️⃣ Activate Environment (One-time per terminal session)

```bash
.\.venv\Scripts\Activate.ps1
```

### 2️⃣ Check GPU Setup

```bash
python -m src.gpu_config --check
python -m src.gpu_config --recommend
```

### 3️⃣ Quick Test (5 minutes)

```bash
# Prepare small dataset
python scripts/prepare_complete_dataset.py --max-examples 100 --output-dir ./data/test_processed

# Train for 50 steps
python scripts/train_unsloth.py `
    --model qwen2.5-7b `
    --dataset ./data/test_processed `
    --max-steps 50 `
    --output-dir ./outputs/test
```

### 4️⃣ Full Training (40 minutes, single GPU)

```bash
# Prepare full dataset
python scripts/prepare_complete_dataset.py --output-dir ./data/processed

# Train full model
python scripts/train_unsloth.py `
    --model qwen2.5-7b `
    --gpu-id 0 `
    --dataset ./data/processed `
    --batch-size 8 `
    --grad-accum 2 `
    --max-steps 1000 `
    --output-dir ./outputs/qwen2.5-7b-v1
```

---

## Complete Walkthrough Available

📖 See **[RUN_END_TO_END.md](RUN_END_TO_END.md)** for:
- Step-by-step guide (9 major steps)
- All command examples
- Expected outputs
- Troubleshooting
- Timeline estimates
- Research paper preparation

---

## Key Files Updated

| File | Change |
|------|--------|
| `README.md` | venv → .venv |
| `GPU_SETUP.md` | venv → .venv |
| `GUIDE.md` | Added quick setup + venv → .venv |
| `RUN_END_TO_END.md` | **NEW** - Complete guide |

---

## Architecture Overview

```
User Input (Uzbek Review)
        ↓
[System Prompt] + [Instruction] + [Review Text]
        ↓
Model (Qwen/Llama/DeepSeek/etc, 4-bit QLoRA)
        ↓
JSON Output {"aspects": [{"term": "...", "polarity": "..."}]}
        ↓
[Parse & Evaluate] → F1 Score
```

---

## Experiment Template

```bash
# Train multiple models with single command
foreach ($model in @("qwen2.5-7b", "llama3.1-8b", "deepseek-7b")) {
    python scripts/train_unsloth.py `
        --model $model `
        --gpu-id 0 `
        --dataset ./data/processed `
        --max-steps 1000 `
        --output-dir "./outputs/$model" `
        --run-name $model
}
```

---

## Research Paper Support

### Logs for Your Paper
- **RESEARCH_LOG.md** - 15 log entries with methodology, claims, risks
- **GPU_CONFIG_SUMMARY.md** - Technical implementation details
- **GUIDE.md** - Usage documentation for reproducibility

### Tables to Generate (after training)
1. Model comparison (P/R/F1 scores)
2. LoRA ablation (rank effects)
3. Prompt language comparison
4. Training efficiency (time/memory)

### Figures to Generate
1. System pipeline diagram
2. Training loss curves
3. Polarity distribution
4. Confusion matrix
5. F1 vs LoRA rank plot

---

## Performance Expectations

**Single GPU (RTX A6000, 46GB):**
- ~25 examples/sec training speed
- 6,175 examples × 3 epochs ≈ 35-40 minutes
- Peak VRAM: ~40-42 GB

**Dual GPU (2× A6000, 92GB):**
- ~50 examples/sec (2x faster)
- Same dataset ≈ 15-20 minutes
- DDP handles synchronization

**Expected F1 Scores (Qwen 2.5-7B):**
- ATE F1: 0.65-0.75
- Aspect-Polarity Pair F1: 0.60-0.70
- Sentiment Accuracy: 0.75-0.85

---

## Recommended Experiment Sequence

### Phase 1: Validation (2 hours)
1. ✅ Dataset exploration (completed)
2. Test train on 100 examples
3. Full train on 6,175 examples
4. Evaluate on validation set

### Phase 2: Model Comparison (4-6 hours)
1. Train 4 models at ~7B scale
2. Evaluate each
3. Create comparison table

### Phase 3: Ablations (4-6 hours)
1. LoRA rank: r ∈ {4, 8, 16, 32}
2. Prompt language: Uzbek vs English
3. Model scale: 3B vs 7B vs 14B

### Phase 4: Analysis & Paper (2-3 hours)
1. Generate tables and figures
2. Write results section
3. Update RESEARCH_LOG.md

**Total time: ~10-15 hours over 2-3 days**

---

## Next Steps

1. **Read** [RUN_END_TO_END.md](RUN_END_TO_END.md)
2. **Run** the quick test (5 min)
3. **Monitor** training with GPU check
4. **Document** in RESEARCH_LOG.md
5. **Push** to GitHub when ready

---

## Support

- 📖 Documentation: [GUIDE.md](GUIDE.md), [GPU_SETUP.md](GPU_SETUP.md)
- 🔍 Logs: Check `training_*.log` if errors occur
- 📊 Metrics: WandB dashboard (if `--no-wandb` not set)
- 🐛 Troubleshooting: See [RUN_END_TO_END.md](RUN_END_TO_END.md) § Troubleshooting

---

**Ready to run! 🚀**

Activate environment and start with:
```bash
.\.venv\Scripts\Activate.ps1
python scripts/prepare_complete_dataset.py --max-examples 100 --output-dir ./data/test_processed
```
