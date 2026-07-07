# Pending manuscript edits — do these once experiment numbers arrive

The safe, standalone writing fixes (P0-B) are **already applied** to
`paper_materials/MDPI/paper/main.tex` and the file compiles clean. The edits below are
**contingent on results** and must be completed before submission.

## ⚠️ Forward references already placed in main.tex (must be backed by real results)

I added three forward-looking claims. They are TRUE only after the corresponding
experiment runs. If an experiment is dropped, revert the matching claim.

| Claim now in text | Location | Backed by |
|---|---|---|
| "benchmarked against fine-tuned Uzbek BERT encoders" | Intro contribution 1 ([main.tex:187]) + Conclusion | **P1 BERT baselines** must run |
| "a human-validation study (IAA, judge calibration, gold-verified subset)" | Intro contribution 2 + Conclusion + Limitations | **P0-A** annotators must return files |
| "quality degrades measurably with domain distance" | Conclusion + Limitations | Already supported by judge scores; strengthened by P0-A per-proximity F1 |

## Numbers-dependent edits (fill when results land)

1. **Baseline rows in the results table** (`tab:absa_results`, [main.tex:~430]).
   Add two rows (Tahrirchi-BERT, BERTbek) with ATE exact/partial F1, pair F1, sentiment
   acc/macro-F1 from `baselines/*/eval_results.json`. Add 1–2 sentences in
   §Results stating whether the LLM formulation beats the encoders (answers R1).
   Also report the BIO **term-alignment rate** from `data/bio_processed/stats.json` as a
   footnote caveat (BERT could only be trained on alignable gold terms).

2. **New subsection "Human Validation of the Silver Dataset"** (after §4.2
   Multi-Domain Quality Assessment). Pull from `results/human_validation_report.json`:
   - IAA: weighted Cohen's κ + Krippendorff's α per rubric dimension (+ gold aspect/polarity κ).
   - Judge calibration (R6): Spearman ρ + MAE, human vs GPT-4o-mini, per dimension. State
     whether the judge is trustworthy (ρ target > 0.5–0.7).
   - Silver quality (R3): model-vs-human ATE/pair F1 on the 80 gold reviews; the gold-vs-gold
     ceiling; the per-proximity breakdown (in/near/out/distant) → substantiates R2 framing.
   - Mention release of the gold-verified subset (`*_gold_verified.json`, flip
     `human_verified: true` on those records in `data/final_dataset/`).

3. **Significance statements** (§Results, near the Qwen-vs-Llama comparison,
   [main.tex:~458]). Replace bold "winner" phrasing with the paired-bootstrap result from
   `significance/qwen_vs_llama.json` — expected: pair-F1 delta not significant → report as
   a statistical tie with 95% CI. Do the same for best-LLM vs best-BERT.

4. **Abstract additions** ([main.tex:124]). After the human-validation + BERT numbers exist,
   add one sentence each: (a) "Fine-tuned Uzbek BERT baselines reach ATE F1 of X, vs Y for
   the best LLM"; (b) "A native-speaker validation of N reviews yields Cohen's κ = … and a
   judge–human Spearman ρ = …, and releases a gold-verified subset." Keep it tight.

5. **BDCC citations.** Verified DOIs in `writing/bdcc_citations.bib`. Add `bdcc_ner_disaster`
   (same SI; LoRA on Qwen2-7B vs BERT+CRF — cite as a close peer in Related Work / PEFT),
   `bdcc_collectivia` (low-resource multilingual), optionally `bdcc_katakana`. Append these
   entries to `mybibliography.bib` and add `\cite{}`s, then recompile.

6. **Optional BDCC reframing of the abstract lead.** For a *Big Data and Cognitive Computing*
   venue, consider leading the abstract with the scalable annotation-pipeline / data-creation
   contribution, with the model comparison as the enabling study. A drop-in alternative
   opening is in `writing/abstract_bdcc_lead.md`. Judgement call — current lead is acceptable.

## Applied already (P0-B, no numbers needed) — for the record
- Abstract grammar fixed ("models (Qwen…", "each evaluated…"); "first systematic" qualified
  with "to the best of our knowledge".
- RQ description-list → flowing prose; RQ tags removed from two subsection titles and the
  loss paragraph; conclusion rewritten to match (no bold RQ callouts).
- Contribution list trimmed 4→3 + de-superlatived; loss "finding" demoted to an observation
  in both Intro and Discussion.
- Limitations paragraph rewritten (removes now-addressed "no encoder baselines"/"no human
  gold" claims; states compute constraint on the test split honestly).
- Confirmed MDPI `main.tex` compiles with **zero undefined references / no `??`** (the
  reviewer's broken citation was in the older Springer version, not this file).

## Authorship optics (from P0-B.7) — your decision
- Consider a single corresponding author (you). Give the statistician co-author a concrete,
  visible role: the **paired-bootstrap significance analysis** (P2-C) and IAA statistics.
  Update `\authorcontributions{}` accordingly.
