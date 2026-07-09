# Pending manuscript edits — status after 2026-07-08 integration

Most of the original list is **DONE and applied** to `paper_materials/MDPI/paper/main.tex`
(compiles clean: 0 undefined refs). What remains:

## Still pending

1. **IAA numbers in §Human Validation** — blocked on the rubric redo
   (`human_validation/REDO_RUBRIC_INSTRUCTIONS.md`, ~2–3h for Sanatbek). After the redo:
   re-run `python scripts/analyze_human_validation.py`, then add one sentence to
   `sec:res_human` reporting weighted Cohen's κ and Krippendorff's α per dimension.
   Until then the paper honestly claims only a single-expert calibration (no IAA claim).

2. **LLM significance tests (optional strengthening)** — run on the Mac:
   `python scripts/dump_llm_preds.py --model <merged Qwen> --out .../qwen_preds.jsonl`
   (and Llama), then `scripts/significance_test.py` for Qwen-vs-Llama and Qwen-vs-BERTbek.
   Current text phrases LLM-vs-encoder gaps descriptively and reports only the
   encoder-pair bootstrap (done: only partial-ATE significant, p=0.023).

3. **Zenodo upload** — the paper's Data Availability now mentions the gold-verified subset;
   upload `data/final_dataset/uzbek_multi_domain_absa_gold80.json` to the Zenodo record
   (new version) so the claim is true at submission time.

4. **BDCC citations** — verified entries live in `writing/bdcc_citations.bib`; still optional
   to add 1–2 `\cite{}`s in Related Work (top pick: `bdcc_ner_disaster`, same SI, LoRA on
   Qwen2-7B vs BERT+CRF). Low-effort editorial-fit signal.

5. **Pre-submission email** — `writing/presubmission_email.md` is ready to send (its abstract
   already reflects encoder baselines + human validation).

6. **Authorship decision** — single corresponding author (you); statistician co-author's
   concrete role = bootstrap significance + agreement statistics. Update
   `\authorcontributions{}` before submission.

## Done (applied 2026-07-07/08) — for the record
- P0-B writing pass (grammar, RQ de-templatizing, claim qualification, loss demotion,
  limitations rewrite); confirmed no `??`/undefined refs (reviewer's broken cite was in the
  old Springer version).
- Encoder baselines: §Methodology subsection, 5-system results table with corrected bolds,
  results finding paragraph, Discussion "When is a 7B LLM Worth It?", verified bib entries.
- Human validation: §Results subsection + judge-calibration table (expert = annotator with
  valid ratings), model-vs-gold + inter-human ceiling numbers, per-proximity gradient,
  gold80 release file built, Data Availability updated.
- Abstract rewritten (fixed now-inaccurate "highest ATE F1" claim; added BERT trade-off,
  judge calibration, gold subset).
