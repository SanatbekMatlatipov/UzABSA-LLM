# UzABSA-LLM v3: the stronger version, in one place

This folder holds everything for a stronger version of the paper: the protocol, one runner, and every checkpoint record, log, prediction and judge output the runner produces. No number in it is estimated or typed in by hand. `SUMMARY.md` and `registry.json` are rebuilt from the prediction files, and `MANIFEST.json` stores the SHA-256 of each file, so any number can be checked against the file it came from.

## Already done (30 October 2026, no GPU)

**Three-way split.** `data/` holds train 4,932 / dev 548 / test 609 reviews. The test split is the paper's held-out set, byte for byte, so every v3 number is comparable with Table 2. The dev split is carved from train with the same duplicate-aware grouping, and no review text appears in two splits.

**Ensemble of the paper's five systems.** A majority vote keeps a term when at least 3 of the 5 fine-tuned systems predict it, and gives it the polarity most of them chose. It needs no training and has no tuned parameter. On the 609 test reviews:

| System | Exact ATE P / R / F1 | Partial F1 (1:1) | Pair F1 | Sent. acc. |
|---|---|---|---|---|
| Qwen 2.5-7B (best single system, exact F1) | 0.714 / 0.702 / 0.708 | 0.804 | 0.645 | 0.911 |
| Llama 3.1-8B (best single system, pair F1) | 0.719 / 0.689 / 0.704 | 0.802 | 0.651 | 0.925 |
| Majority of the 3 LLMs | 0.775 / 0.678 / 0.724 | 0.800 | 0.667 | 0.922 |
| **Majority of all 5** | **0.800** / **0.707** / **0.751** | **0.821** | **0.690** | 0.919 |

- **Gains:** the five-system vote beats every member on exact ATE F1 (+0.043 over Qwen) and pair F1 (+0.039 over Llama, +0.045 over Qwen).
- **Significance:** all these gains hold under the paired bootstrap after Holm correction (p ≤ 0.034).
- **The encoders cause the gain.** The LLM-only vote gains just +0.016 to +0.022, which is not significant. Decoders and encoders make different errors, and the vote mostly removes spurious terms. Precision reaches 0.800, against 0.729 for the most precise single system (TahrirchiBERT), and recall (0.707) matches the best single system (BERTbek, 0.704).
- **Other combinations tried:** two more combinations were scored during exploration. Qwen, Llama and both encoders (at least 3 of 4) reached exact 0.738 and pair 0.684. Qwen's two seeds plus Llama (at least 2 of 3) reached 0.736 and 0.677. Both also beat every single system. The v3 seeds will test the five-system rule on runs it has never seen.
- **Files:** `results/ensembles/`.

**Judge statistics checked.** `scripts/judge_panel.py` reproduces the published judge's agreement with the two human raters exactly: Spearman 0.657, tier κ 0.468, linear κ 0.546, 73.3% tier agreement and 85.4% Include precision.

**Judge panel calibrated (real API runs, 30 October 2026).** GPT-4o-mini and GPT-4.1 each scored the 150 human-rated reviews under both rubrics; the panel score is their mean. Results are compared with the mean of the two human raters:

| Judge | Spearman ρ, overall (95% CI) | Tier κ (95% CI) | Tier agreement | Include precision | Completeness ρ |
|---|---|---|---|---|---|
| GPT-4o-mini, published run (the paper) | 0.657 (0.55–0.75) | 0.468 (0.33–0.60) | 73.3% | 85.4% | 0.462 |
| GPT-4o-mini, v1 rubric, rerun | 0.659 | 0.509 | 75.3% | 87.4% | 0.468 |
| GPT-4.1, v1 rubric | 0.741 | 0.412 | 70.0% | 87.6% | 0.657 |
| **Panel, v1 rubric** | **0.760 (0.68–0.84)** | **0.601 (0.47–0.73)** | **81.3%** | **87.7%** | **0.636** |
| Panel, checklist rubric | 0.661 | 0.138 | 69.3% | 71.6% | 0.499 |

- **Panel vs. published judge:** in this calibration run, the v1 panel improves on the published judge in the paired bootstrap, ρ by +0.11 (p < 0.001) and κ by +0.13 (p = 0.026).
- **Numbers the paper uses:** the released tiers come from the full-corpus run, which also scored these 150 reviews. On them the panel reaches:
  - ρ 0.781, a significant gain (p < 0.001);
  - κ 0.562, a gain that is *not* significant (p = 0.08);
  - 79.3% tier agreement and 86.7% Include precision.

  The run above is therefore a repeat measurement; the two runs put 95.3% of the reviews in the same tier. `scripts/judge_paper_numbers.py` writes every judge number in the paper to `results/judge_paper_numbers.json`.
- **Largest gain:** completeness, the dimension the paper names as the judge's weak point.
- **The checklist rubric failed.** Written as the primary rubric, it made GPT-4o-mini lenient: mean overall score 4.78 against 3.84 for the humans, and 141 of 150 reviews rated Include. The two judges then barely agreed with each other (Krippendorff's α 0.02 on the overall score).
- **What scores the corpus:** the v1 rubric, the paper's own, scored by the two-judge panel. Because the rubric was chosen with the calibration ratings, report its κ as slightly optimistic.
- **Cost:** about $0.30 per rubric for the 150 reviews.
- **Files:** `judge/calibration_v1/` and `judge/calibration_checklist/`, each with every raw response under `raw/`.

**All 5,038 released reviews judged (30–31 October 2026).** Both judges scored every review with the v1 rubric, with no parse failures. Cost: $10.01 ($0.63 for GPT-4o-mini, $9.38 for GPT-4.1).

- **Tiers:** Include 3,608 (71.6%), Flag 972 (19.3%), Exclude 458 (9.1%). No review is left unjudged.
- **Same 307 reviews:** on the reviews the published judge scored, the panel puts 71.3% in Include, against 63.5% for the published judge. The two agree closely (rank correlation 0.87, same tier for 86%). The higher Include rate therefore comes from the better-calibrated judge, not from the sample, and the 307-review sample was representative.
- **Domains:** restaurants, the training domain, reach 80.9% Include (n = 1,626).
  - In 18 of the 20 other domain groups most reviews reach Include, highest in travel (79.5%). Among the larger domains, healthcare and grocery stores (72.6%) and e-commerce (72.0%) follow.
  - The two exceptions are entertainment (46.2%, n = 26) and investment (exactly 50%, n = 22).
  - Every estimate now rests on all of a domain's reviews: the smallest group, three pooled domains, has 31 reviews, against 3–12 judged reviews per domain before.
- **The two judges agree** on the overall score across all 5,038 reviews: Spearman 0.74, Krippendorff's α 0.75.
- **Files:** `judge/release_v1/` holds the scores, the summary and every raw response with its token usage; `final_dataset_release/` holds the tiered corpus.

**Uzbek-Cyrillic reviews transliterated (31 October 2026, Mac).** The corpus has 4,681 reviews in Latin script, 205 in Russian and 152 in Uzbek Cyrillic (language check in `src/transliterate.py`). The released Uzbek-Cyrillic reviews reach Include only 50.0% of the time, against 72.3% for all others and 91.2% for Russian.

The corrected Qwen checkpoint (HF `qwen2.5-7b` @ `8636bd1`, bf16 on MPS) annotated each of them twice:
- as written;
- transliterated to Latin, with the terms mapped back to the Cyrillic text (240 of 242 mapped exactly).

The same v1 panel judged both:

| Uzbek-Cyrillic reviews (n = 152) | Include | Flag | Exclude | Mean overall |
|---|---|---|---|---|
| Released annotation (old checkpoint) | 50.0% | 27.6% | 22.4% | 3.29 |
| Corrected Qwen, Cyrillic input | 54.6% | 23.0% | 22.4% | 3.31 |
| **Corrected Qwen, Latin transliteration** | **68.4%** | 13.2% | 18.4% | **3.53** |

- **Effect:** 31 reviews move into Include and 10 out of it (McNemar p = 0.0015). Mean overall +0.22, 95% CI [0.08, 0.37], Wilcoxon p = 0.009. Every rubric dimension improves.
- **Cost:** $0.61, plus 28 min on an M2 Max.
- **Pipeline:** the v3 `annotate` stage uses the converter.
- **Files:** `translit/` (`report.json`, `raw.jsonl`, both judge runs). The result is in Sect. 5.5 of the paper.

**Test-set gold re-checked (31 October 2026).** The protocol is in `test_recheck/INSTRUCTIONS.md`. The author and Jaloliddin Rajabov each annotated the 263 reviews (575 candidate terms) independently, blind to which terms were gold. κ was 0.94 for aspect decisions and 0.91 for polarity, and they resolved the 37 disagreements together.

Outcomes (`summary.json`, `outcomes_by_reason`):
- **Gold terms that at most one system predicts (152):** 68% were wrong; 52% are not aspects at all.
- **Non-gold terms that 3 or more systems predict (130):** 51% are real aspects.
- **Polarity disputes (29):** 55% were changed to the systems' polarity.
- **Control reviews:** 97% of their gold terms were kept.

In total, 281 gold annotations changed in 169 reviews.

| System | Exact ATE F1, original → re-checked | Pair F1, original → re-checked |
|---|---|---|
| Qwen 2.5-7B | 0.708 → **0.770** | 0.645 → **0.702** |
| Llama 3.1-8B | 0.704 → 0.752 | 0.651 → 0.700 |
| TahrirchiBERT | 0.701 → 0.756 | 0.623 → 0.687 |
| BERTbek | 0.692 → 0.740 | 0.608 → 0.666 |
| DeepSeek-R1-7B | 0.678 → 0.721 | 0.603 → 0.654 |

- **Significance:** Qwen vs. Llama remains a tie (p = 0.21 and 0.95; `test_recheck/significance/`).
- **Do not score the majority vote on the re-checked gold.** The reviews were selected with the same five-system votes.

## What each stage fixes in the paper

| The paper today | Stage | What the paper can say afterwards | Cost |
|---|---|---|---|
| Limitation (i): checkpoints chosen on the held-out set | `split`, `train-llm`, `train-enc` | Checkpoints and epochs chosen on dev, so the 609 reviews become a true test set | About 9 A6000-hours for 9 LLM runs (about 5 h on two GPUs); encoders about 1 h on the Mac |
| Limitation (ii): seed variance only for Qwen | the same runs | Mean ± sd over 3 seeds for all five systems | Included above |
| The ensemble is a new result | `ensemble` | Seed-matched votes with their spread over 3 seeds, plus one vote over all 15 runs | Minutes on a CPU |
| Limitation (v): silver labels from the pre-fix annotator | `annotate` | Corpus re-annotated by the v3 run with the best dev score | About 6 GPU-hours |
| Only 307 of 5,038 reviews judged; 14 domains have 3–12 judged reviews | `judge-release`, then `judge-v3` after re-annotation | **Done for the released corpus:** every review judged and tiered (3,608 Include, 972 Flag, 458 Exclude) | $10.01, actual |
| Judge–human tier κ = 0.47 with a single small judge | `judge-calibrate` | **Done:** on the released tiers the two-judge panel reaches ρ = 0.78 (p < 0.001 against the published judge) and κ = 0.56 (p = 0.08); in the paper since 31 October 2026 | $0.62, actual |
| Numbers are hard to trace to files | `collect` | Registry, manifest and per-job logs | Seconds |

## How to run

Run from the repository root. Every stage skips jobs whose output already exists, so you can run a command again after a crash. `--force` reruns finished jobs and `--dry-run` prints the commands without running them.

| Where | Command | What it does |
|---|---|---|
| GPU machine | `python experiments/v3/run_v3.py gpu` | Builds the split; trains 3 models × 3 seeds, at most one job per GPU listed in `config.json`; scores dev and test; re-annotates the corpus with the best dev run. Disk: about 15 GB per merged 7–8B model, about 140 GB in total. |
| Mac (MPS) or GPU machine | `python experiments/v3/run_v3.py encoders` | Builds the BIO split; trains 2 encoders × 3 seeds; scores dev and test. |
| Any machine with API keys | `python scripts/judge_panel.py calibrate --smoke --out /tmp/smoke`, then `python experiments/v3/run_v3.py judge` | The smoke test makes one call per judge to check keys and model names. Then: calibration against the human ratings, judging of all 5,038 released annotations, and their tiers. Done on 30–31 October 2026; reruns skip finished work. |
| After the GPU stages | `python experiments/v3/run_v3.py judge-v3 assemble report` | Judges the re-annotated corpus, builds its tiers, runs ensembles and significance tests, and rebuilds `SUMMARY.md`. |

- **Judges:** a judge runs when `enabled` is true in `judges.json` and its key variable (`OPENAI_API_KEY`, `GEMINI_API_KEY`, …) is set.
- **Resuming judges:** every response is logged with its token usage, so an interrupted run resumes without paying twice.
- **Logs:** each job writes `logs/<stage>/<job>.log`, and `logs/runner.log` records each job's command, host, git commit, duration and exit code.

## Rules that keep these results defensible

- **Seeds:** report every seed as mean ± sd, never only the best seed.
- **Selection:** choose checkpoints, the annotator and any setting on dev or on the human calibration ratings. The 609 test reviews are only scored.
- **Ensemble rule:** simple majority over the members, fixed before any v3 run existed. Report the LLM-only vote and all members next to it; `SUMMARY.md` does this.
- **Judge rubric:** the checklist rubric was written from the human annotation protocol (`paper_materials/revision_v2/human_validation/protocol_uz.md`) before any calibration result existed, and it lost to the v1 rubric. Report both rubrics and every judge, not only the best one.
- **Judge selection:** the corpus is scored with the v1 panel because it agreed best with the 150 human ratings. Say so in the paper: the panel's κ of 0.60 is measured on the same ratings that chose it, so it is slightly optimistic.

## Layout

```
experiments/v3/
  config.json       models, seeds, GPUs, training arguments
  judges.json       judge panel: endpoints, key variables, prices
  run_v3.py         the runner
  data/             train / dev / test (test = the paper's 609 reviews)
  data_bio/         BIO version of the split for the encoders
  runs/<system>_s<seed>/   checkpoints, training logs, dev and test predictions
  annotated/        re-annotated corpus
  judge/            calibration_<rubric>/, release_<rubric>/, v3_<rubric>/;
                    raw/ in each holds every judge response with its token usage
  final_dataset_release/   Include / Flag / Exclude tiers of the released corpus
  final_dataset_v3/        the same for the re-annotated corpus
  results/          ensembles/, significance/
  translit/         Uzbek-Cyrillic reviews annotated as written and transliterated, their judge
                    scores and report.json (scripts/cyrillic_experiment.py)
  test_recheck/     re-check of the test-set gold: INSTRUCTIONS.md, GUIDELINES_uz.md, the annotator
                    workbooks and, once filled, the cleaned test set (scripts/test_recheck.py)
  logs/             one log per job, plus runner.log
  registry.json, SUMMARY.md, MANIFEST.json
```

Weights stay out of git (see `.gitignore`); publish them on the Hub instead. The new code follows the rest of the pipeline in `scripts/`:

- **New scripts:** `make_dev_split.py`, `ensemble_preds.py`, `judge_panel.py`, `collect_v3.py`, `cyrillic_experiment.py` and `test_recheck.py`; new module `src/transliterate.py`.
- **Changed scripts:** `evaluate.py` gained `--split`; `annotate_reviews.py` gained `--source-name` and `--transliterate-cyrillic`, which the `annotate` stage passes; `src/inference.py`'s `extract_aspects` gained `transliterate=`.
