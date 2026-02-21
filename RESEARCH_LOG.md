# Research Log — UzABSA-LLM Project
# Fine-tuning Open-Source LLMs for Uzbek Aspect-Based Sentiment Analysis
# ======================================================================
# Author: Sanatbek Matlatipov
# Started: February 2026
# Status: In Progress
# ======================================================================


## LOG 001 — Problem Statement & Motivation
Date: Feb 2026

- ABSA for Uzbek language is under-explored; no published LLM fine-tuning work exists for this task in Uzbek.
- Existing multilingual models (mBERT, XLM-R) have limited Uzbek coverage.
- Open-source LLMs (Qwen, Llama, DeepSeek) show promise for low-resource languages via instruction tuning.
- Research question: **Can parameter-efficient fine-tuning (QLoRA) of open-source LLMs achieve competitive ABSA performance for Uzbek text?**
- Sub-questions:
  - Which base model performs best for Uzbek ABSA?
  - How does QLoRA compare to full fine-tuning for this low-resource scenario?
  - What is the impact of prompt language (Uzbek vs English instructions) on performance?


## LOG 002 — Dataset Description
Date: Feb 2026

### Primary Dataset (Annotated)
- Source: HuggingFace `Sanatbek/aspect-based-sentiment-analysis-uzbek`
- Size: **6,175 examples** (train split)
- Annotation scheme: **SemEVAL 2014 Task 4** format
- Fields per example:
  - `sentence_id`: unique identifier (format: "XXXX#Y")
  - `text`: Uzbek review sentence
  - `aspect_terms`: list of {term, polarity, from, to} — character-level spans
  - `aspect_categories`: list of {category, polarity}
- Polarity labels: `positive`, `negative`, `neutral`
- Statistics (measured):
  - Avg aspects per example: **2.53**
  - Avg text length: **6.42 words** per sentence
  - Polarity distribution: **~64% positive, ~16% negative, ~20% neutral**
- NOTE: Class imbalance exists — positive is dominant. Consider stratified splitting or weighted loss.

### Secondary Dataset (Raw/Unannotated)
- Source: sharh.commeta.uz
- Size: **5,058 raw reviews**
- Fields: review_text, object_name, rating_value, reviewer_name, review_date, source_url
- Statistics (measured):
  - Avg words: **13.55** per review
  - Avg chars: **96.26** per review
- Purpose: potential semi-supervised learning, data augmentation, domain analysis
- Cleaning pipeline: null removal → empty string removal → min length filter (10 chars) → deduplication
- NOTE: Not used for training in current experiments — reserved for future work.


## LOG 003 — Methodology: Task Formulation
Date: Feb 2026

- ABSA reformulated as **text generation** task (instruction tuning)
- Input: Uzbek review sentence + task instruction
- Output: structured JSON containing extracted aspects with polarities
- Template format: **ChatML** (`<|im_start|>system/user/assistant<|im_end|>`)
- This is a **joint extraction** approach — model extracts both aspect terms and their sentiment in a single pass

### Instruction Format (Uzbek prompt — used for training)
```
System: Siz o'zbek tilida matnlardan aspektlarni va ularning hissiyotlarini 
        aniqlash bo'yicha mutaxassissiz. [...]
User:   Quyidagi o'zbek tilidagi matndan aspektlarni, kategoriyalarni va 
        hissiyot polaritesini aniqlang:
        Matn: "{text}"
Output: {"aspects": [{"term": "...", "category": "...", "polarity": "..."}]}
```

- NOTE: JSON output format chosen for parseability — allows automatic metric computation
- NOTE: Uzbek-language system prompt used by default; English alternative available for ablation study


## LOG 004 — Model Selection
Date: Feb 2026

### Candidate Models (all 4-bit quantized via bitsandbytes)

| Model | Parameters | HuggingFace Path | Notes |
|-------|-----------|-----------------|-------|
| Qwen 2.5 7B | 7B | unsloth/Qwen2.5-7B-Instruct-bnb-4bit | Strong multilingual, good Uzbek |
| Qwen 2.5 14B | 14B | unsloth/Qwen2.5-14B-Instruct-bnb-4bit | Larger capacity |
| Qwen 2.5 32B | 32B | unsloth/Qwen2.5-32B-Instruct-bnb-4bit | Highest capacity |
| Llama 3 8B | 8B | unsloth/llama-3-8b-Instruct-bnb-4bit | Meta's flagship |
| Llama 3.1 8B | 8B | unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit | Updated Llama |
| Llama 3.2 3B | 3B | unsloth/Llama-3.2-3B-Instruct-bnb-4bit | Smallest, fastest |
| DeepSeek 7B | 7B | unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit | Distilled reasoning |
| DeepSeek 14B | 14B | unsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit | Larger reasoning |
| Mistral 7B | 7B | unsloth/mistral-7b-instruct-v0.3-bnb-4bit | Efficient architecture |
| Gemma 2 9B | 9B | unsloth/gemma-2-9b-it-bnb-4bit | Google's model |

- Selection rationale: all are instruction-tuned, available in 4-bit, and cover diverse architectures
- KEY POINT FOR PAPER: Compare at least 3-4 models (e.g., Qwen 7B, Llama 3.1 8B, DeepSeek 7B, Mistral 7B) at same parameter scale for fair comparison


## LOG 005 — Fine-tuning Method: QLoRA
Date: Feb 2026

### Quantization
- Method: **QLoRA** (Dettmers et al., 2023)
- Base precision: **4-bit NormalFloat** (NF4) via bitsandbytes
- Compute dtype: **BFloat16** (bf16=True, fp16=False)
- Double quantization: enabled (Unsloth default)

### LoRA Configuration
- Rank (r): **16**
- Alpha: **32** (scaling factor = alpha/r = 2.0)
- Dropout: **0.05**
- Bias: **none**
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
  - All attention projections + FFN projections
  - 7 target modules per transformer layer
- RSLoRA: disabled
- LoftQ: disabled

### Training Hyperparameters
- Optimizer: **AdamW 8-bit** (memory efficient)
- Learning rate: **2e-4**
- LR scheduler: **cosine** decay
- Warmup ratio: **0.1** (10% of total steps)
- Weight decay: **0.01**
- Max gradient norm: **1.0** (gradient clipping)
- Max sequence length: **2048 tokens**
- Effective batch size: **8** (per_device=2 × grad_accum=4)
  - Auto-adjusted based on GPU memory (up to 16 on A6000)
- Max steps: **1000** (or 3 epochs if max_steps=-1)
- Seed: **42**
- Packing: **disabled** (each example processed individually)
- Gradient checkpointing: **enabled** (Unsloth optimized variant)
- Trainer: **SFTTrainer** from TRL library

### Optimization Framework
- Library: **Unsloth** — claims 2x faster training, 80% less memory
- KEY POINT: Unsloth provides fused kernels and optimized gradient checkpointing
- NOTE: Report actual training time and peak memory as empirical evidence

### Data Split
- Train/Validation: **90/10** split (stratified shuffle, seed=42)
- ~5,558 train / ~617 validation (estimated from 6,175 total)
- NOTE: No separate test set — consider k-fold or holding out a test set


## LOG 006 — Evaluation Metrics
Date: Feb 2026

### Aspect Term Extraction (ATE)
- **Exact match**: predicted term must exactly equal gold term (case-insensitive, stripped)
- **Partial match**: substring or superstring matching (relaxed criterion)
- Micro-averaged Precision, Recall, F1:
  - P = TP / (TP + FP)
  - R = TP / (TP + FN)
  - F1 = 2PR / (P + R)
- Aggregation: across ALL examples (micro), not per-example average (macro)

### Aspect Sentiment Classification (ASC)
- Evaluated on **matched** aspect pairs only — (term, polarity) tuple must match
- Same micro P/R/F1 formulation
- Additionally:
  - Sentiment accuracy (sklearn accuracy_score)
  - Sentiment macro-F1 (sklearn, zero_division=0)

### Joint ABSA Metric
- Aspect-polarity pair F1 captures joint extraction + classification
- KEY POINT: This is the primary metric — captures end-to-end performance

### Output Parsing
- Model output parsed via JSON extraction (regex: `r'\{[\s\S]*\}'`)
- Fallback: regex-based term/polarity extraction for malformed JSON
- KEY METRIC TO REPORT: **JSON parse success rate** — indicates model's instruction-following ability
- Inference: greedy decoding (temperature=0.1, do_sample=False), max_new_tokens=512


## LOG 007 — Hardware & Infrastructure
Date: Feb 2026

### GPU Setup
- **4x NVIDIA RTX A6000** (48GB VRAM each, 192GB total)
  - Ampere architecture (SM 8.6)
  - CUDA 12.8
- Planning to use **1-2 GPUs** per experiment
- Single A6000 config: batch_size=8, grad_accum=2 → effective batch=16
- Dual A6000 config: batch_size=8, grad_accum=2, DDP → effective batch=32

### Software Stack
- Python 3.10+
- PyTorch ≥ 2.1.0
- Transformers ≥ 4.45.0
- Unsloth (latest from GitHub main)
- TRL (for SFTTrainer)
- PEFT (for LoRA)
- bitsandbytes (for 4-bit quantization)
- accelerate ≥ 0.27.0

### Estimated Training Time
- ~6,175 examples × 3 epochs = ~18,525 training steps (at batch_size=1)
- With effective batch=8: ~2,316 steps per epoch → ~6,948 total
- Single A6000 @ ~25 examples/sec: ~12 min/epoch → ~37 min total
- NOTE: Record actual time per model for paper


## LOG 008 — Experiment Plan
Date: Feb 2026

### Experiment 1 — Model Comparison (Primary)
- Goal: Compare base models on Uzbek ABSA
- Models: Qwen 2.5-7B, Llama 3.1-8B, DeepSeek-7B, Mistral-7B (all at ~7B scale)
- Fixed: LoRA r=16, alpha=32, lr=2e-4, 3 epochs, same data split
- Metric: Aspect-polarity pair F1

### Experiment 2 — LoRA Rank Ablation
- Goal: Effect of LoRA rank on performance
- Values: r ∈ {4, 8, 16, 32, 64}
- Fixed: Best model from Exp 1, alpha=2r

### Experiment 3 — Prompt Language Ablation
- Goal: Uzbek vs English system prompts
- Compare: Uzbek instruction prompt vs English instruction prompt
- Fixed: Best model, best LoRA rank

### Experiment 4 — Model Scale (if time permits)
- Goal: Effect of model size
- Compare: Qwen 2.5 at 7B, 14B, 32B (or Llama 3B vs 8B)

### Experiment 5 — Zero-shot / Few-shot Baseline
- Goal: Establish baseline without fine-tuning
- Test base models (no LoRA) on same test set
- Compare: zero-shot vs few-shot (3-5 examples in prompt) vs fine-tuned


## LOG 009 — Key Claims to Support in Paper
Date: Feb 2026

1. **First** comprehensive evaluation of open-source LLMs fine-tuned for Uzbek ABSA
   - Evidence: literature review showing no prior work
   
2. QLoRA enables **efficient** fine-tuning on consumer/prosumer GPUs
   - Evidence: training time, peak VRAM usage, comparison to full fine-tuning cost
   
3. Joint aspect extraction + sentiment classification via **instruction tuning** is effective
   - Evidence: F1 scores on aspect-polarity pairs
   
4. **Model comparison** across architectures reveals best choice for Uzbek
   - Evidence: controlled experiments at same scale (Exp 1)
   
5. **Structured JSON output** from generative models is reliable
   - Evidence: JSON parse success rate > X%
   
6. **Uzbek-language prompts** affect model performance
   - Evidence: Exp 3 ablation results


## LOG 010 — Paper Structure (Planned)
Date: Feb 2026

1. **Introduction**: Low-resource ABSA, motivation for Uzbek, LLM approach
2. **Related Work**: ABSA methods, LLM fine-tuning, Uzbek NLP, QLoRA
3. **Dataset**: Description, annotation scheme (SemEVAL 2014), statistics, class distribution
4. **Methodology**:
   - Task formulation (generation-based ABSA)
   - Instruction format (ChatML, Uzbek prompts)
   - QLoRA configuration
   - Model selection
5. **Experimental Setup**: Hardware, hyperparameters, metrics, baselines
6. **Results**: Model comparison table, ablations, error analysis
7. **Discussion**: Findings, limitations, Uzbek-specific challenges
8. **Conclusion & Future Work**: Semi-supervised extension with raw data, larger datasets

### Tables to Prepare
- Table 1: Dataset statistics (size, polarity distribution, avg aspects)
- Table 2: Model comparison results (P, R, F1 for ATE and ASC)
- Table 3: LoRA rank ablation results
- Table 4: Prompt language comparison
- Table 5: Training efficiency (time, memory, parameters)

### Figures to Prepare
- Fig 1: System architecture / pipeline diagram
- Fig 2: Training loss curves (per model)
- Fig 3: Polarity distribution in dataset
- Fig 4: Confusion matrix for sentiment classification
- Fig 5: F1 vs LoRA rank plot


## LOG 011 — Important References to Cite
Date: Feb 2026

- Pontiki et al. (2014) — SemEVAL 2014 Task 4 (ABSA task definition)
- Dettmers et al. (2023) — QLoRA: Efficient Finetuning of Quantized LLMs
- Hu et al. (2022) — LoRA: Low-Rank Adaptation of LLMs
- Touvron et al. (2023) — Llama 2 (and Llama 3 tech report if available)
- Yang et al. (2024) — Qwen2.5 Technical Report
- Jiang et al. (2023) — Mistral 7B
- DeepSeek-AI (2024) — DeepSeek-R1 technical report
- Zhang et al. (2024) — Instruction tuning survey
- Unsloth — cite GitHub repo + any available paper
- Relevant Uzbek NLP papers (search needed)


## LOG 012 — Risks & Limitations (Note for Paper)
Date: Feb 2026

- Dataset size (6,175) is small by LLM standards — risk of overfitting
  - Mitigation: early stopping, validation monitoring, low epochs
- No separate test set currently — using val for evaluation
  - TODO: Create proper train/val/test split (80/10/10)
- Class imbalance (64% positive) may bias predictions
  - TODO: Report per-class F1, not just micro-average
- Uzbek tokenizer coverage unknown for each model
  - TODO: Measure tokenizer fertility (tokens/word ratio) per model
  - KEY INSIGHT: Higher fertility = less efficient encoding = potential performance impact
- Single-domain data (reviews) — generalization unclear
- JSON parsing failures reduce effective evaluation set
  - Must report parse failure rate


## LOG 013 — TODO Before Running Experiments
Date: Feb 2026

- [ ] Create proper 80/10/10 train/val/test split
- [ ] Measure tokenizer fertility for each model on Uzbek text
- [ ] Run zero-shot baselines (no fine-tuning) for comparison
- [ ] Prepare evaluation script to output publication-ready tables
- [ ] Run full data preparation on all 6,175 examples
- [ ] Execute Experiment 1 (model comparison)
- [ ] Record: training time, peak VRAM, loss curves, final metrics
- [ ] Analyze errors: what types of aspects/sentiments does model miss?
- [ ] Check if any model handles Cyrillic Uzbek (Ўзбек) vs Latin (O'zbek)


## LOG 014 — Tokenizer Fertility Analysis (TODO)
Date: Feb 2026

- This is critical for understanding model behavior on Uzbek
- Measure: avg tokens per word for each model's tokenizer
- Lower is better (model "understands" the language better)
- Example test sentences to prepare:
  ```
  "Bu restoranning ovqatlari juda mazali, lekin narxlari qimmat."
  "Xizmat ko'rsatish sifati past, kutish vaqti juda uzoq."
  "Mahsulot sifati a'lo darajada, yetkazib berish tez."
  ```
- Compare across: Qwen, Llama, DeepSeek, Mistral, Gemma tokenizers


## LOG 015 — Reproducibility Checklist
Date: Feb 2026

For paper submission, ensure:
- [ ] All hyperparameters documented (LOG 005 ✓)
- [ ] Random seed fixed (42 ✓)
- [ ] Dataset version/hash recorded
- [ ] Model versions (exact HuggingFace paths ✓)
- [ ] Hardware specified (LOG 007 ✓)
- [ ] Software versions (requirements.txt ✓)
- [ ] Training logs saved (WandB ✓)
- [ ] Code released on GitHub (repo exists ✓)
- [ ] Dataset publicly available (HuggingFace ✓)
- [ ] Statistical significance tests (multiple seeds or bootstrap)


# ======================================================================
# END OF CURRENT LOGS — Update as experiments progress
# ======================================================================
# Next: Run experiments → LOG 016+ with actual results
# ======================================================================
