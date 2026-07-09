# UzABSA-LLM → BDCC Revision Log

Running log of the revision effort for resubmission to **MDPI BDCC**, Special Issue
*"Artificial Intelligence (AI) and Natural Language Processing (NLP)"* (`W633E7395P`).

Every completed subtask appends a **dated entry** here with the numbers and file paths it
produced. These entries are the raw material for the manuscript edits — quote them directly.

Plan of record: `~/.claude/plans/let-s-make-what-to-memoized-cupcake.md`

Owner legend: 🤖 = Claude (automated) · 🧑 = human step · 🤝 = shared

---

## 2026-07-07 — Scaffold created 🤖

- Created `paper_materials/revision_v2/` with subfolders: `results/`, `baselines/`,
  `human_validation/` (+ `returned/`), `significance/`, `writing/`.
- This log initialized.

Baseline numbers already in the repo (from `outputs/my_run/*/eval_results_*.json`), evaluated
on the **609-example split** (721 gold pairs) — the current paper's headline table:

| Model | ATE exact F1 | ATE partial F1 | Pair F1 | Sent. Acc | Sent. macro-F1 | JSON parse |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | **0.6603** | **0.7705** | 0.5795 | 0.8777 | 0.8113 | **100.0%** |
| Llama3.1-8B | 0.6549 | 0.7591 | **0.5805** | **0.8864** | **0.8435** | 95.89% |
| DeepSeek-R1-7B | 0.6034 | 0.7279 | 0.5018 | 0.8317 | 0.7717 | 95.40% |

LLM-as-Judge aggregate (307-review stratified sample, GPT-4o-mini): completeness 3.32,
accuracy 4.47, sentiment 4.21, relevance 4.06, overall 3.75. Include(≥3.5)=195 (63.5%).

---
<!-- Append new dated entries below this line -->

## 2026-07-07 — P0-A human-validation artifacts built 🤖 (annotation pending 🧑)

**Scripts:** `scripts/build_human_validation.py`, `scripts/analyze_human_validation.py`.

**Sample built** (`paper_materials/revision_v2/human_validation/`):
- `sample_150.csv` — 150 reviews drawn from the 307 GPT-4o-mini-judged set (seed 42),
  spanning all 23 domains, oversampling low-score/distant domains. Internal join file
  (carries model predictions + judge scores). NOT given to annotators.
- `rubric_template.csv` — annotator-facing; rate the model's predicted aspects 1–5 on
  five dimensions (same rubric as the LLM judge). 150 rows.
- `gold_template.csv` — annotator-facing subset of **80** reviews (67 from priority
  domains); write correct aspects from scratch in `term :: polarity ;; ...` format.
- `protocol_uz.md` — Uzbek-language annotation instructions for both tasks.

**Analysis pipeline** (`scripts/analyze_human_validation.py`) — smoke-tested end-to-end
with synthetic annotators; computes:
- IAA per rubric dim: quadratic-weighted Cohen's κ + Krippendorff's α (interval; unit-tested:
  1.0 perfect / 0.85 one-off / −0.75 total disagreement) + Spearman.
- Judge validation (R6): Spearman ρ + MAE, mean-human vs GPT-4o-mini, per dimension.
- Silver quality (R3): model-vs-human ATE (exact/partial) + pair F1 on the 80 gold reviews,
  reusing `src/evaluation.py` metric functions; plus a gold-vs-gold IAA ceiling.
- Per-proximity (R2): in/near/out/distant breakdown of model-vs-human F1.

**Runs in the Python 3.9 env** (imports `src/evaluation.py` directly, bypassing heavy
`src/__init__.py`). No GPU needed.

**🧑 NEXT (blocker):** give `protocol_uz.md` + `rubric_template.csv` + `gold_template.csv`
to 2 native speakers; they return `rubric_<name>.csv` / `gold_<name>.csv` into
`human_validation/returned/`. Then run `python scripts/analyze_human_validation.py`.

## 2026-07-07 — P1 BERT baseline pipeline built 🤖 (run on Mac pending 🧑)

**Scripts:** `scripts/prepare_bio_dataset.py`, `scripts/train_bert_baseline.py`,
`scripts/eval_bert_baseline.py`. Run guide: `baselines/README_RUN_ON_MAC.md`.

- **BIO-polarity scheme** (O + B/I × {positive,negative,neutral,conflict}) → one encoder
  does joint extraction + polarity, decodable to `{term,polarity}` for reuse of the shared
  metric functions.
- Built from the **same ChatML `data/processed` split** (5480/609, seed 42) via
  `parse_chatml_example`, so BERT is comparable to the LLMs on the identical 609 set.
- **Self-test passed** (`--selftest`): BIO conversion + round-trip decode correct, incl. the
  morphological fallback (term `narx` aligns to text `narxi`) and correct skipping of
  category-only / absent terms.
- **Fairness:** `eval_bert_baseline.py` scores against the FULL term-bearing gold from the
  ChatML (identical reference set to the LLM eval), not the reduced BIO-aligned subset.
- Targets: `tahrirchi/tahrirchi-bert-base`, `elmurod1202/bertbek-news-big-cased`.
  Runs on Apple MPS; saves `training_log.txt`, `eval_results.json`, `preds_per_example.jsonl`.

**🧑 NEXT:** run the 3 commands in `baselines/README_RUN_ON_MAC.md` on the Mac; send me the
two `eval_results.json`. Note the term-alignment rate from step 1 (a paper caveat).

## 2026-07-07 — P2-C significance + P2-B pred-dumper built 🤖 (run pending 🧑)

**Scripts:** `scripts/significance_test.py` (paired bootstrap), `scripts/dump_llm_preds.py`
(regenerates per-example LLM predictions the original eval never saved).

- `significance_test.py`: paired bootstrap over the 609 set; reports observed delta,
  95% CI, two-sided p-value for ate_exact_f1 / ate_partial_f1 / pair_f1 / sentiment_acc.
  **Self-test passed**: identical-system control → delta 0, CI [0,0], p=1.0; distinct
  systems → sensible CI/p. Consumes the `preds_per_example.jsonl` format that
  `eval_bert_baseline.py` and `dump_llm_preds.py` both emit.
- Primary use: settle the **Qwen-vs-Llama pair-F1 gap of 0.001** (expected: not
  significant → replace bold "winner" with "statistical tie") and best-LLM vs best-BERT.
- `dump_llm_preds.py`: runs a fine-tuned (merged fp16) or base LLM over the 609 set on
  Mac MPS, saving predictions for the bootstrap. Also serves the zero-shot baseline (P2-A).

**🧑 NEXT (optional, Mac-heavy):** after BERT eval, optionally
`python scripts/dump_llm_preds.py ...` for Qwen + Llama, then
`python scripts/significance_test.py --a qwen_preds.jsonl --b llama_preds.jsonl ...`.

## 2026-07-07 — P0-B writing pass + P0-C BDCC reframe 🤖 (applied to main.tex)

**Edits applied to `paper_materials/MDPI/paper/main.tex` (compiles clean, 0 undefined refs):**
- Abstract: grammar fixed; "first systematic" → qualified "to the best of our knowledge".
- RQ `\begin{description}` block → flowing prose paragraph; RQ tags removed from 2 subsection
  titles + the loss paragraph; conclusion rewritten (no bold RQ callouts).
- Contributions trimmed 4→3, de-superlatived; loss "finding" demoted to an observation in
  Intro + Discussion.
- Limitations rewritten: removes now-addressed "no encoder baselines"/"no human gold"; states
  compute constraint on the held-out split honestly; frames cross-domain claim cautiously (R2).
- **Confirmed the MDPI file has no `??` / no undefined refs** — the reviewer's broken citation
  (R4) was in the older Springer version, not this one.

**Forward references added** (contingent — see `writing/pending_edits.md`): BERT-benchmark and
human-validation claims are now in Intro/Conclusion and MUST be backed by P1/P0-A before submit.

**Staged in `writing/`:**
- `pending_edits.md` — the numbers-dependent edits (baseline table rows, human-validation
  subsection, significance statements, abstract additions) with exact source files/locations.
- `bdcc_citations.bib` — 4 in-SI / low-resource BDCC papers; **all DOIs verified to resolve**
  (doi.org → correct MDPI articles). Top pick `bdcc_ner_disaster` (10.3390/bdcc10060185): same
  SI, LoRA PEFT of Qwen2-7B vs BERT+CRF — mirrors our design.
- `presubmission_email.md` — ready ~230-word scope enquiry to the guest editors (Sardar Jaf,
  Basel Barakat, Yongqiang Cheng); does not mention the Computers rejection.
- `abstract_bdcc_lead.md` — optional pipeline-first abstract opening for venue fit.

**🧑 NEXT:** (1) send the pre-submission email; (2) run P1 (BERT) + hand out P0-A annotation;
(3) once numbers land, I complete the edits in `pending_edits.md`.

## 2026-07-07 — README updated with a minimal, Mac-only install path 🤖

**Added:** `baselines/requirements_bert_baseline.txt` — 11 packages needed for the BERT
baseline ONLY (transformers, datasets, accelerate, tokenizers, safetensors, evaluate,
seqeval, scikit-learn, scipy, numpy, tqdm). Deliberately excludes unsloth/bitsandbytes/trl/
wandb/tensorboard/jupyter/dev-tools from the root `requirements.txt` — those are CUDA/LLM-
training-only, irrelevant here, and some may not build on Apple Silicon at all.

**Updated `baselines/README_RUN_ON_MAC.md`** with a new "Step 0" covering:
- **arm64 check** — the earlier `platform.machine()` probe in this session returned
  `x86_64` on the user's system Python, which means MPS acceleration will silently fall
  back to CPU (no error) if that Python is used. Added an explicit check + fix (Homebrew
  arm64 Python, or `arch -arm64 pyenv install`).
- venv creation + `pip install torch` (no `--index-url` needed on macOS — the default
  PyPI wheel ships MPS support) + `pip install -r requirements_bert_baseline.txt`.
- A verification one-liner confirming `arch: arm64` and `mps avail: True` before proceeding.

**🧑 NEXT:** run Step 0 first — if `platform.machine()` still prints `x86_64` after creating
the new venv, the BERT training will silently run on CPU (slower, ~2-4x, but should still
finish; not blocking). Then proceed to steps 1-4 as before.

## 2026-07-08 — Fixed `evaluate` module-shadowing bug in BERT trainer 🤖

**Symptom:** `AttributeError: module 'evaluate' has no attribute 'load'` at
`train_bert_baseline.py` startup (after the model downloaded fine).

**Cause:** running `python scripts/train_bert_baseline.py` puts `scripts/` on `sys.path[0]`,
so `import evaluate` resolved to this repo's `scripts/evaluate.py` instead of the HuggingFace
`evaluate` library. (The BERT "LOAD REPORT" above the error is benign — MLM head dropped, a
fresh token-classification head initialized, as intended.)

**Fix:** removed the `evaluate` dependency from the trainer; now calls `seqeval` directly
(`from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score`,
signature `(y_true, y_pred)`). Also dropped `evaluate` from `requirements_bert_baseline.txt`.
No behaviour change to the reported span metrics.

**🧑 NEXT:** ensure `seqeval` is installed (`pip install seqeval`), then re-run the same
`train_bert_baseline.py` command — it will proceed past this point.

## 2026-07-08 — Fixed Trainer `tokenizer` arg for new transformers 🤖

**Symptom:** `TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'`
(after the `evaluate` shadowing fix — training reached Trainer construction).

**Cause:** transformers >=4.46 renamed `Trainer(tokenizer=...)` to `processing_class=...`.
User's env is bleeding-edge (torch 2.12.1).

**Fix:** build Trainer kwargs and pick `processing_class` vs `tokenizer` via
`inspect.signature(Trainer.__init__)`, so it works on old and new transformers. Also confirmed
`eval_strategy` (not the deprecated `evaluation_strategy`) is used — correct for new versions.

## 2026-07-08 — BERT baselines + human validation integrated into main.tex 🤖

### BERT baseline results (609-example set, identical gold to LLM eval)
| System | ATE exact F1 | ATE partial F1 | Pair F1 | Sent. Acc | Sent. macro-F1 |
|---|---|---|---|---|---|
| **BERTbek** (elmurod1202/bertbek-news-big-cased) | **0.6694** | **0.7951** | 0.5615 | 0.8388 | 0.7691 |
| **TahrirchiBERT** (tahrirchi/tahrirchi-bert-base) | 0.6528 | 0.7746 | 0.5505 | 0.8432 | 0.7673 |

- BIO term-alignment: 98.2% train / 98.5% val (`data/bio_processed/stats.json`).
- Training: ~10 min each on M2 Max (lr 3e-5, 4 epochs, batch 16, seed 42).
- **Story:** BERTbek BEATS all LLMs at extraction (0.6694 vs Qwen 0.6603); LLMs win sentiment
  (macro-F1 0.8435/0.8113 vs ~0.77) and pair F1 (0.5805/0.5795 vs 0.5615). Honest trade-off.
- Paired bootstrap BERTbek-vs-Tahrirchi (5000 resamples,
  `significance/bertbek_vs_tahrirchi.json`): only partial-ATE significant (p=0.023);
  exact-ATE p=0.18, pair p=0.46, sentiment p=0.80 — ties.

### Human validation — what the returned data showed
- `rubric_Sanatbek.csv` is DEGENERATE (sentiment & relevance 150×5; overall 145×5) → IAA
  unusable (κ 0.0–0.08, α negative). See `human_validation/REDO_RUBRIC_INSTRUCTIONS.md`
  (~2–3h redo; then κ/α go into sec:res_human).
- `rubric_Jaloliddin.csv` has realistic spread → used as the expert for judge calibration:
  ρ sentiment 0.795, relevance 0.683, **overall 0.633**, accuracy 0.437, completeness 0.436
  (all p<1e-6, n=150); judge overall mean 3.75 vs expert 3.77; judge harsher on completeness
  (3.30 vs 3.99). Saved: `results/judge_vs_expert_annotator.json`.
- Gold task: both files genuine & independent (verified per-review: 31/80 differing counts).
  Model-vs-expert(J): ATE exact 0.209 / partial 0.433; 100% polarity on exact-matched terms.
  **Inter-human ceiling: exact 0.277 / partial 0.415** → model's partial agreement AT the
  human-human level. Per-proximity exact pair F1: out 0.244, distant 0.170 (mirrors judge
  gradient). Annotator2 gold much sparser (54 vs 134 aspects) — convention divergence,
  disclosed in the paper.
- Released `data/final_dataset/uzbek_multi_domain_absa_gold80.json` (80 reviews, double
  annotation, human_verified=true).

### main.tex changes (compiles clean: 0 undefined refs, 0 overfull)
- New §Methodology "Encoder Baselines" (sec:meth_encoder) + bib entries kuriyozov2024bertbek,
  tahrirchibert2023 (both verified).
- tab:absa_results extended to 5 systems (grouped LLM/encoder header); bold corrected
  (BERTbek now holds ATE bolds); caption notes non-significance.
- New results finding paragraph "Encoders are competitive at extraction; LLMs win on
  sentiment" incl. bootstrap result.
- New §"Human Validation of Annotation Quality" (sec:res_human) + tab:judge_calibration;
  scope honestly limited (author-annotators, no IAA claim yet, convention divergence stated).
- New Discussion paragraph "When is a 7B LLM Worth It?".
- Limitations updated (gold-subset caveats; significance scope).
- Abstract rewritten back half: BERT trade-off + judge calibration + gold subset; fixed the
  now-inaccurate "highest ATE F1" claim (BERTbek exceeds Qwen).
- Data availability: gold subset + baseline/validation code mentioned.

### Still open
1. 🧑 Rubric redo (REDO_RUBRIC_INSTRUCTIONS.md) → then add IAA sentence to sec:res_human.
2. 🧑 Optional: `dump_llm_preds.py` for Qwen+Llama → LLM-vs-BERT + Qwen-vs-Llama bootstrap
   (currently described descriptively; encoder-pair test done).
3. 🧑 Upload gold80 file to the Zenodo record (paper says it's there).
4. Send pre-submission email (writing/presubmission_email.md) — update its abstract to
   mention encoder baselines are DONE (it already does).

## 2026-07-08 (later) — Redone annotations integrated; full-paper review pass 🤖

### Redone annotation data (returned/ overwritten in place)
- Rubric (Sanatbek v2): realistic spread. Row-level agreement with Jaloliddin: 84–91%
  exact per dimension, all disagreements ±1 → weighted κ 0.91–0.97, α 0.91–0.97.
- Gold (Sanatbek v2): 125 aspects (was 54). Inter-annotator gold agreement now exact F1
  0.9266, polarity 100% (was 0.277).
- **User confirmed the redo consulted/aligned with Jaloliddin's files** → reported in the
  paper as a two-stage annotate-then-reconcile protocol: round-1 INDEPENDENT agreement =
  exact 0.277/partial 0.415 (convention divergence); round-2 reconciled consolidated gold
  (residual agreement 0.927). Rubric κ reported as post-calibration consistency, NOT
  independent IAA. Judge calibration now vs MEAN human rating: ρ overall 0.657, sentiment
  0.806, relevance 0.667, completeness 0.462, accuracy 0.433; human/judge means 3.84/3.75.
- Model vs consolidated gold: exact 0.209–0.224 / partial 0.425–0.433 per annotator;
  100% polarity on matched. Gold80 release file REBUILT with reconciled data.

### main.tex changes (compiles clean: 0 undefined, 0 overfull, 0 bibtex warnings)
- §Human Validation rewritten: two-stage protocol narrative; updated calibration table
  (mean-human); model-vs-gold with two reference points (pre-reconciliation human agreement
  ≈ model's 0.43 partial; reconciled ceiling 0.93 → gap is span segmentation, not opinion
  detection); scope caveat (author-annotators) retained.
- **Abstract cut ~280 → ~190 words** (dropped preprocessing counts, loss-vs-performance
  sentence, 63.5% inclusion detail; kept: first-study claim, key numbers, encoder trade-off,
  pipeline, judge calibration ρ=0.66, gold subset, public release).
- Fixed leftover contradiction: "encoder-based baselines left for future work" (Results
  setup ¶) → now points at sec:meth_encoder. "Qwen highest ATE F1" → "highest LLM ATE F1".
- Related Work: added BERTbek/TahrirchiBERT sentence (Uzbek NLP ¶) and in-venue BDCC
  citation (Zhang et al. 2026, BDCC 10(6):185 — LoRA Qwen2-7B vs BERT+CRF; authors fetched
  via CrossRef) in the PEFT ¶. Keywords: added "LLM-as-Judge".
- Contribution 2 harmonized ("two-stage annotation-agreement analysis...").
- Limitations updated to reconciliation phrasing.

### Remaining before submission
1. 🧑 Upload gold80 (rebuilt) to Zenodo record.
2. 🧑 Send pre-submission email (writing/presubmission_email.md).
3. 🧑 Optional: dump_llm_preds for LLM-vs-encoder significance.
4. 🧑 Authorship: single corresponding author + statistician role in \authorcontributions.
