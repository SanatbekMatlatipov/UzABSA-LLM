# UzABSA-LLM v3: results index

Built 2026-10-31T01:27:15 from git `c5ea987308` (uncommitted changes). Every number is recomputed from the prediction files listed in `registry.json`; `MANIFEST.json` holds their SHA-256 hashes.

## Paper systems (v2, 609 held-out reviews)

| System | Exact ATE F1 | Partial F1 (1:1) | Pair F1 | Sent. acc. | Macro-F1 | Checkpoint |
|---|---|---|---|---|---|---|
| Qwen 2.5-7B | 0.708 | 0.804 | 0.645 | 0.911 | 0.862 | hf://Sanatbek/UzABSA-LLM@qwen2.5-7b |
| Qwen 2.5-7B (seed 43) | 0.710 | 0.817 | 0.655 | 0.922 | 0.881 | not kept |
| Llama 3.1-8B | 0.704 | 0.802 | 0.651 | 0.925 | 0.887 | hf://Sanatbek/UzABSA-LLM@llama3.1-8b (**stale**: last updated 2026-02-25) |
| DeepSeek-R1-7B | 0.678 | 0.789 | 0.603 | 0.890 | 0.831 | hf://Sanatbek/UzABSA-LLM@deepseek-r1-7b (**stale**: last updated 2026-02-25) |
| BERTbek | 0.692 | 0.799 | 0.608 | 0.879 | 0.813 | not kept |
| TahrirchiBERT | 0.701 | 0.786 | 0.623 | 0.889 | 0.814 | not kept |

## Ensembles (majority vote, no training, no tuned parameter)

| Ensemble | Exact ATE F1 | Partial F1 (1:1) | Pair F1 | Sent. acc. | vs. best member (pair F1) |
|---|---|---|---|---|---|
| Majority of 3 LLMs (v2 systems) | 0.724 | 0.800 | 0.667 | 0.922 | +0.016 vs Llama 3.1-8B (p=0.151, Holm 1.0) |
| Majority of 5 (v2 systems) | 0.751 | 0.821 | 0.690 | 0.919 | +0.039 vs Llama 3.1-8B (p=0.0034, Holm 0.034) |

## Judge calibration: calibration_checklist (n=150 human-rated reviews)

| Judge | Spearman (overall) | Tier κ | Linear κ | Tier agreement | Include precision |
|---|---|---|---|---|---|
| gpt-4o-mini [checklist] | 0.412 | 0.195 | 0.244 | 0.713 | 0.716 |
| gpt-4.1 [checklist] | 0.621 | 0.372 | 0.433 | 0.700 | 0.819 |
| gpt-4o-mini (published, v1) | 0.657 | 0.468 | 0.546 | 0.733 | 0.854 |
| panel mean [checklist] | 0.661 | 0.138 | 0.222 | 0.693 | 0.716 |

## Judge calibration: calibration_v1 (n=150 human-rated reviews)

| Judge | Spearman (overall) | Tier κ | Linear κ | Tier agreement | Include precision |
|---|---|---|---|---|---|
| gpt-4o-mini [v1] | 0.659 | 0.509 | 0.573 | 0.753 | 0.874 |
| gpt-4.1 [v1] | 0.741 | 0.412 | 0.539 | 0.700 | 0.876 |
| gpt-4o-mini (published, v1) | 0.657 | 0.468 | 0.546 | 0.733 | 0.854 |
| panel mean [v1] | 0.760 | 0.601 | 0.663 | 0.813 | 0.877 |

## Corpus judging: release_v1

5038 of 5038 reviews scored; Include rate 71.6%; tiers {'include': 3608, 'exclude': 458, 'flag': 972}; cost (USD) gpt-4o-mini: 0.6347, gpt-4.1: 9.3807.

## Still missing

- Llama 3.1-8B: the Hub branch `llama3.1-8b` was last updated 2026-02-25, before the run behind the paper's numbers; push that run (scripts/push_to_hub.py)
- DeepSeek-R1-7B: the Hub branch `deepseek-r1-7b` was last updated 2026-02-25, before the run behind the paper's numbers; push that run (scripts/push_to_hub.py)
- LLM runs with test predictions: 0/9 (run_v3.py gpu / encoders)
- encoder runs with test predictions: 0/6 (run_v3.py gpu / encoders)
- re-annotation and judging of the corpus (run_v3.py annotate judge-v3 assemble)
