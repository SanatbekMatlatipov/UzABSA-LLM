# Pending manuscript edits — status 2026-08-29 (post-rerun, post-merge)

Target journal: **MDPI BDCC**, template = 27 July 2026 class. `main.tex` reports
the **v2 (deduplicated-split, fixed-prompt) results** throughout, with zero-shot
controls, split-effect analysis, all-pairs paired bootstraps, and a vector
training-curves figure. PDF compiles clean (verified in Overleaf 2026-08-29).

## Blocking — needs a decision

**The target special issue appears CLOSED.** `9N6QB6G1RO` =
*"Natural Language Processing Applications in Big Data"* (guest editors Xingyi
Song, Ye Jiang, Yunfei Long), and two independent web searches report its
manuscript deadline as **31 December 2024**. MDPI serves 403 to automated
fetches, so this could not be confirmed on the page itself — **verify by logging
into susy.mdpi.com and checking whether it still accepts submissions.**

If it is closed, the best-matching open alternative found:

| SI | Title | Deadline | Why it fits |
|---|---|---|---|
| `42jxfu49ss` | Advances in NLP and Text Mining: **2nd Edition** | **31 Dec 2026** | Keywords explicitly include *low-resource NLP*, *large language models*, *text mining* — a direct match for this paper |
| `JEISHYZ92J` | Advances in NLP and Text Mining (1st ed.) | check | same scope, earlier edition |

Note the previously targeted `W633E7395P` ("AI and NLP") closed 20 July 2026, and
`O9A9UWB542` (targeted before that) was never confirmed. The manuscript itself is
SI-agnostic — no LaTeX change is required to switch; the SI is chosen in the SuSy
submission form. Only the Introduction's scope framing would benefit from tuning
if the chosen SI is narrower than generic NLP.

## Remaining before submission

1. **HuggingFace upload (~44 GB)** — step-by-step in
   `writing/huggingface_upload_guide.md`. Verified ready: token `depparse` has
   `repo.write`, `--all --dry-run` clean, cards carry v2 metrics + correction
   notice. The currently published weights are the **defective v1** ones, so this
   should happen before the Data Availability statement is relied on.

2. **Confirm the SI** (above) and select it in SuSy.

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
