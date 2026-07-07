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
