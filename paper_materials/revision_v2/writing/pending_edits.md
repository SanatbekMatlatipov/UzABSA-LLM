# Pending manuscript edits — status after 2026-08-29 rerun integration

Target journal: **MDPI BDCC**, template updated to the 27 July 2026 class.
`main.tex` now reports the **v2 (deduplicated-split, fixed-prompt) results**
throughout, with zero-shot controls, a split-effect analysis, all-pairs paired
bootstraps, and a vector training-curves figure. See REVISION_LOG 2026-08-29.

## Still pending before submission

1. **Compile the PDF locally** — no LaTeX toolchain on this machine. Run the
   full cycle (pdflatex → bibtex → pdflatex ×2) and check:
   - the new tikz figure (fig:loss_curves) renders on one line (3 panels);
   - tab:zeroshot and tab:split_effect fit the text width;
   - no overfull boxes / undefined refs (structural checks already pass).

2. **One p-value sentinel** — `PVAL_QB_CLAUSE` in the encoder finding awaits
   the qwen_vs_tahrirchi / qwen_vs_bertbek bootstrap results (running). Grep
   for `PVAL_` before compiling; it must be gone.

3. **Zero-shot caption claim** — tab:zeroshot asserts all FT–ZS differences
   significant at p<1e-4; verify against {qwen,llama,deepseek}_ft_vs_zs.json
   once the tests finish (Qwen's needs the regenerated ZS preds file).

4. **Zenodo v2 upload** — bundle ready at `revision_v2/zenodo_release/`
   (regenerated 2026-08-29 with corrected README). Upload as New Version of
   record 18790639 before submission so the Data Availability claim is true.

5. **HuggingFace model update** — `scripts/push_to_hub.py` is repointed at the
   v2 models with correction notices; run `--all --dry-run`, then `--all`
   (~45 GB upload). The currently published weights are the defective v1 ones.

6. **Confirm the special issue** `O9A9UWB542` is open and NLP-scoped in
   susy.mdpi.com (MDPI serves 403 to automated access; must be checked by a
   human).

7. **Author consents** — Matlatipov/Djalilov/Aripov author-order change and
   Djalilov's ORCID/affiliation; Aripov's email now filled in
   (mirsaid.aripov@nuu.uz) — confirm it is the address he wants.

8. **Presubmission email** (`writing/presubmission_email.md`) predates both the
   author change and the v2 results; refresh before sending.

## Done 2026-08-29 (see REVISION_LOG for detail)

- Full v2 rerun: 3 LLMs + 2 encoders retrained on the corrected split; all
  evals, zero-shot controls (×3), seed replicate (Qwen/43), bootstrap CIs,
  all-pairs significance tests.
- Manuscript: abstract, contributions, setup, tab:absa_results,
  tab:train_efficiency, findings, loss-vs-task, two new subsections
  (sec:res_zeroshot, sec:res_split), discussion, conclusion, limitations,
  pipeline provenance footnote, vector fig:loss_curves.
- MDPI template upgraded 13-Mar → 27-Jul-2026 (matched cls+journalnames pair);
  `pdftex` option dropped from \documentclass per new template.
- Zenodo bundle rebuilt with data-derived README (no false `conflict` claim).
- push_to_hub.py: repointed at v2, cards generated from artifacts, Windows
  cp1252 crash fixed.

## Superseded / closed

- ~~IAA numbers~~ (in paper, tab:iaa)
- ~~LLM significance tests blocked by disk~~ (run on the 4×A6000 machine)
- ~~61/609 leakage disclosure~~ (leak eliminated; now analyzed in
  sec:res_split instead of disclosed as a limitation)
- ~~zero-shot baseline missing~~ (sec:res_zeroshot)
- ~~TODO-ARIPOV-EMAIL placeholder~~ (filled)
