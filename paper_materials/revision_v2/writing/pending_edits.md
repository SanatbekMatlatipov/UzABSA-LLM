# Pending manuscript edits — status 2026-08-29 (post-rerun, post-merge)

Target journal: **MDPI BDCC**, template = 27 July 2026 class. `main.tex` reports
the **v2 (deduplicated-split, fixed-prompt) results** throughout, with zero-shot
controls, split-effect analysis, all-pairs paired bootstraps, and a vector
training-curves figure. PDF compiles clean (verified in Overleaf 2026-08-29).

## Special issue — CONFIRMED OPEN

Target: **`9N6QB6G1RO` — "Natural Language Processing Applications in Big Data"**
(Big Data and Cognitive Computing, ISSN 2504-2289).
**Deadline for manuscript submissions: 22 October 2026.** Confirmed by the author
on the live MDPI page 2026-08-29.

(For the record: automated search reported a stale 31-Dec-2024 deadline for this
SI, and MDPI serves 403 to automated fetches, so the page could not be read
directly. The deadline was extended; the search index had not caught up. Trust
the live page over search results here.)

Scope fit is good and needs no manuscript change: the SI is about NLP applied to
big-data settings, and the Introduction already frames the work as "a problem
generic to large-scale text analytics" — turning a heterogeneous stream of
user-generated reviews into a quality-controlled analytical resource — which is
exactly the SI's angle. The 5,038-review multi-domain corpus and the
model-assisted annotation pipeline are the big-data contribution.

## Remaining before submission

1. **HuggingFace upload (~44 GB)** — step-by-step in
   `writing/huggingface_upload_guide.md`. Verified ready: token `depparse` has
   `repo.write`, `--all --dry-run` clean, cards carry v2 metrics + correction
   notice. The currently published weights are the **defective v1** ones, so this
   should happen before the Data Availability statement is relied on.

2. **Select SI `9N6QB6G1RO`** in the SuSy submission form (confirmed open,
   deadline 22 October 2026).

## Open items from the 2026-08-31 external review (see REVISION_LOG.md)

Ordered by value per unit of effort. Items 1–3 need no GPU.

1. 🧑 **Resolve the licensing contradiction — blocking.** Zenodo publishes the
   corpus as **CC BY 4.0** (permits commercial reuse), while the Acknowledgments
   state permission was granted "exclusively for academic research purposes".
   These conflict. Either obtain written permission from Sardor Berdiyev
   (Commeta) explicitly covering open redistribution under CC BY, **or** switch
   the Zenodo record to a non-commercial/research-only license and reword the
   manuscript. Do not submit with both statements standing.

2. 🤖/🧑 **Judge the remaining 4,731 reviews — <$1, no GPU.** The strongest
   available answer to "most of the silver corpus was never quality-tiered".
   `python scripts/llm_judge.py --annotations data/annotated/reviews_annotated.json
   --provider openai --model gpt-4o-mini --sample-size 5038 --output-dir data/judged_full`
   Measured cost basis: 307 reviews ≈ $0.045, so the full corpus ≈ $0.75.
   Then re-run `assemble_dataset.py`, refresh Table 7–10 / Figure 4, and the
   "unjudged" hedging in §Layer 3 can be dropped entirely.

3. 🧑 **Third-annotator adjudication of the 80-review subset.** Would let
   "reconciled double-annotated" become genuine adjudicated gold. Ideally also
   widen beyond 80 reviews and report bootstrap CIs on the human-vs-model F1.

4. 🧑 **HF model card + repo license sync.** Card shows a different JSON schema,
   Uzbek (not English) polarity labels, term–category–sentiment triples (the
   paper says the layers are independent), and sampling rather than greedy
   decoding; its project link 404s. Repo license is GPL-3.0 while source headers
   say MIT and HF says Apache-2.0 — and the **Llama branch cannot be relicensed
   as Apache-2.0** under the Llama community licence. Pick one licence per
   artifact and make the card match §Methodology.

5. 🖥️ **GPU-bound, honestly disclosed rather than fixed:**
   - Regenerate the 5,038 annotations with the corrected Qwen v2 checkpoint
     (~6 GPU-h at the measured 4.3 s/review). The released silver corpus still
     comes from the pre-fix annotator. Currently footnoted in §pipeline and
     Limitations — a reviewer may still press on it.
   - True three-way train/dev/test split + ≥3 seeds per system. The 609-example
     partition is a dev set reused for headline numbers; stated in Limitations.

## Done — 2026-08-29

- **PDF compiles** (Overleaf, user-verified). Structural checks also pass:
  0 unbalanced braces/envs, 0 undefined refs, 0 uncited-key errors.
- **All p-value sentinels resolved.** Every significance claim in the paper is
  backed by a committed JSON in `significance_v2/`:
  - Qwen-Llama: tied on all four metrics (p = 0.79 / 1.00 / 0.68 / 0.27)
  - DeepSeek below Qwen (pair F1 p=0.009) and Llama (pair F1 p=0.002)
  - Qwen vs encoders, extraction: ties (vs Tahrirchi p=0.61; vs BERTbek p=0.29)
  - Qwen pair-F1/sentiment over BERTbek significant (p=0.020/0.021), not over
    Tahrirchi (p=0.13/0.14); Llama sentiment over Tahrirchi significant (p=0.014)
  - Encoder-encoder: no metric significant (0.46 / 0.13 / 0.25 / 0.39)
- **Zero-shot caption claim verified.** All three FT-vs-ZS comparisons are
  significant at p<10^-4 (Qwen dPair +0.435, Llama +0.451, DeepSeek +0.449).
  The regenerated Qwen ZS predictions reproduce both published tables to four
  decimals (ATEex 0.3222 / 0.7077; pair 0.2100 / 0.6448) — i.e. the table numbers
  and the significance tests come from the same predictions, independently
  re-derived.
- **Zenodo v2 uploaded** (2026-08-28) — Data Availability statement is true.
- **Author details final**: Matlatipov (1st, corresponding), Djalilov (2nd),
  Aripov (3rd); `mirsaid.aripov@nuu.uz` confirmed correct.
- **Pre-submission email refreshed** by the author.
- MDPI template upgraded 13-Mar to 27-Jul-2026 (matched cls + journalnames pair);
  `pdftex` dropped from `\documentclass` per the new template.
- `push_to_hub.py` repointed at v2 models; cards generated from artifacts;
  Windows cp1252 crash fixed.

## Closed / superseded

- ~~IAA numbers~~ -> in paper (`tab:iaa`)
- ~~LLM significance tests blocked by disk~~ -> run on the 4xA6000 machine
- ~~61/609 leakage disclosure~~ -> leak eliminated; now a *result*
  (`sec:res_split`) rather than a limitation
- ~~zero-shot baseline missing~~ -> `sec:res_zeroshot` + `tab:zeroshot`
- ~~TODO-ARIPOV-EMAIL~~ -> filled and confirmed
- ~~W&B screenshot figure~~ -> pgfplots vector figure from the released logs
