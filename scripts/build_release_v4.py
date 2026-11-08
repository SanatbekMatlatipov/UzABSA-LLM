#!/usr/bin/env python3
"""Assemble version 4 of the Zenodo release: the multi-domain review corpus, its quality metadata,
the human validation files and the benchmark files derived from UzABSA.

Every file is copied or re-shaped from an existing pipeline output; nothing is
re-annotated or re-scored. Sources:
  corpus     experiments/v3/final_dataset_release/   (judge panel, all 5,038 reviews)
  judges     experiments/v3/judge/release_v1/raw/     (one line per review and judge)
  human      paper_materials/revision_v2/human_validation/ (150 rated, 80 re-annotated)
  benchmark  data/processed (5,480 / 609 split), experiments/v3/data (dev split),
             experiments/v3/test_recheck/, the prediction files listed in
             experiments/v3/registry.json, and the UzABSA files on Hugging Face
             (Sanatbek/aspect-based-sentiment-analysis-uzbek, for sentence IDs)
  taxonomy   data/raw/absa_subcategories.json, data/raw/business_categories.json

Usage (from the repository root):
    python scripts/build_release_v4.py                      # downloads the UzABSA jsonl from the Hub
    python scripts/build_release_v4.py --uzabsa-jsonl <path to aspect-based-sentiment-analysis-uzbek.jsonl>
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import importlib.util

import datasets

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("uzabsa_evaluation", ROOT / "src/evaluation.py")
_evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluation)
parse_eval = _evaluation.parse_chatml_example
V3 = ROOT / "experiments/v3"
HV = ROOT / "paper_materials/revision_v2/human_validation"
DIMS = ["completeness", "accuracy", "sentiment", "relevance", "overall"]

SYSTEMS = [  # (column name, file)
    ("qwen2.5-7b_ft", "paper_materials/revision_v2/results_v2/qwen2.5-7b_preds.jsonl"),
    ("llama3.1-8b_ft", "paper_materials/revision_v2/results_v2/llama3.1-8b_preds.jsonl"),
    ("deepseek-r1-7b_ft", "paper_materials/revision_v2/results_v2/deepseek-7b_preds.jsonl"),
    ("bertbek_ft", "paper_materials/revision_v2/baselines_v2/bertbek/preds_per_example.jsonl"),
    ("tahrirchibert_ft", "paper_materials/revision_v2/baselines_v2/tahrirchi-bert/preds_per_example.jsonl"),
    ("vote5_ft", "experiments/v3/results/ensembles/majority_of_5_v2_systems_preds.jsonl"),
    ("qwen2.5-7b_zs", "paper_materials/revision_v2/results_v2/qwen2.5-7b_zeroshot_preds.jsonl"),
    ("llama3.1-8b_zs", "paper_materials/revision_v2/results_v2/llama3.1-8b_zeroshot_preds.jsonl"),
    ("deepseek-r1-7b_zs", "paper_materials/revision_v2/results_v2/deepseek-7b_zeroshot_preds.jsonl"),
]


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path, rows, cols):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def chatml_parts(t):
    text = re.search(r'Matn:\s*"(.*?)"<\|im_end\|>', t, re.S).group(1)
    target = json.loads(re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", t, re.S).group(1))["aspects"]
    return text, target


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def corpus(out):
    src = V3 / "final_dataset_release"
    for name in ["uzbek_multi_domain_absa_full.json", "uzbek_multi_domain_absa.jsonl",
                 "uzbek_multi_domain_absa_approved.json", "reviews_flagged_for_review.json",
                 "dataset_stats.json"]:
        shutil.copy2(src / name, out / name)
    full = json.loads((src / "uzbek_multi_domain_absa_full.json").read_text(encoding="utf-8"))
    cols = ["review_id", "business_category", "business_name", "user_rating", "aspect_index",
            "aspect_term", "aspect_category", "polarity"] + [f"judge_{d}" for d in DIMS] + \
           ["quality_tier", "human_verified", "annotation_source", "review_text"]
    rows = []
    for r in full:
        base = {k: r.get(k, "") for k in ["review_id", "business_category", "business_name", "user_rating",
                                          "quality_tier", "human_verified", "annotation_source"]}
        base.update({f"judge_{d}": r[f"judge_{d}"] for d in DIMS})
        base["review_text"] = norm(r["text"])
        for i, a in enumerate(r["aspects"] or []):
            rows.append({**base, "aspect_index": i, "aspect_term": a.get("term") or "",
                         "aspect_category": a.get("category") or "", "polarity": a.get("polarity") or ""})
    write_csv(out / "uzbek_multi_domain_absa.csv", rows, cols)
    # small stratified preview: the first reviews of every domain, so all 23 domains show
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["business_category"]].append(r)
    per = max(1, 300 // len(by_dom))
    preview = [x for d in sorted(by_dom) for x in by_dom[d][:per]][:300]
    write_csv(out / "00_PREVIEW_uzbek_absa_sample.csv", preview, cols)
    return full, len(rows)


def judges(out, full):
    rows = []
    for judge in ["gpt-4o-mini", "gpt-4.1"]:
        for r in jl(V3 / f"judge/release_v1/raw/{judge}__v1.jsonl"):
            rows.append({"review_id": r["review_id"], "judge": judge, "served_model": r.get("served_model", ""),
                         **{d: r["scores"][d] for d in DIMS},
                         "explanation": norm((r.get("evidence") or {}).get("explanation", ""))})
    write_csv(out / "judge_panel_scores.csv", rows, ["review_id", "judge", "served_model"] + DIMS + ["explanation"])
    # the corpus fields must be the mean of the two judges
    by = defaultdict(dict)
    for r in rows:
        by[r["review_id"]][r["judge"]] = r
    bad = sum(1 for r in full for d in DIMS
              if abs((by[r["review_id"]]["gpt-4o-mini"][d] + by[r["review_id"]]["gpt-4.1"][d]) / 2 - r[f"judge_{d}"]) > 1e-9)
    assert bad == 0, f"{bad} panel scores differ from the judge mean"
    return len(rows), Counter(r["served_model"] for r in rows)


def human(out, full):
    panel = {r["review_id"]: r for r in full}
    sample = {r["review_id"]: r for r in csv.DictReader(open(HV / "sample_150.csv", encoding="utf-8-sig"))}
    a = {r["review_id"]: r for r in csv.DictReader(open(HV / "returned/rubric_Jaloliddin.csv", encoding="utf-8-sig"))}
    b = {r["review_id"]: r for r in csv.DictReader(open(HV / "returned/rubric_Sanatbek.csv", encoding="utf-8-sig"))}
    cols = ["review_id", "business_category", "in_gold80", "review_text", "model_aspects"]
    cols += [f"annotator1_{d}" for d in DIMS] + ["annotator1_notes"]
    cols += [f"annotator2_{d}" for d in DIMS] + ["annotator2_notes"]
    cols += [f"single_judge_{d}" for d in DIMS] + [f"panel_{d}" for d in DIMS]
    rows = []
    for rid, s in sample.items():
        row = {"review_id": rid, "business_category": s["business_category"],
               "in_gold80": s["in_gold_subset"] == "yes", "review_text": norm(s["text"]),
               "model_aspects": s["model_aspects_readable"]}
        for tag, src in [("annotator1", a), ("annotator2", b)]:
            for d in DIMS:
                row[f"{tag}_{d}"] = int(src[rid][f"{d}_1_5"])
            row[f"{tag}_notes"] = norm(src[rid].get("notes", "") or "")
        for d in DIMS:
            row[f"single_judge_{d}"] = int(float(s[f"judge_{d}"]))
            row[f"panel_{d}"] = panel[rid][f"judge_{d}"]
        rows.append(row)
    write_csv(out / "human_ratings_150.csv", rows, cols)
    shutil.copy2(ROOT / "paper_materials/revision_v2/zenodo_release/uzbek_multi_domain_absa_gold80.json",
                 out / "uzbek_multi_domain_absa_gold80.json")
    shutil.copy2(HV / "protocol_uz.md", out / "human_validation_protocol_uz.md")
    return rows


def benchmark(out, uzabsa_jsonl):
    v2 = datasets.load_from_disk(str(ROOT / "data/processed"))
    v3 = datasets.load_from_disk(str(V3 / "data"))
    dev_of = {}
    for sp, tag in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        for t in v3[sp]["text"]:
            dev_of.setdefault(t, set()).add(tag)
    assert all(len(v) == 1 for v in dev_of.values()), "a ChatML text sits in two dev-split partitions"
    ids = defaultdict(list)
    for s in jl(uzabsa_jsonl):
        ids[norm(s["text"])].append(s["sentence_id"])
    rows, missing = [], 0
    for sp, tag in [("train", "train"), ("validation", "test")]:
        for i, t in enumerate(v2[sp]["text"]):
            text, target = chatml_parts(t)
            sid = ids.get(norm(text), [])
            missing += not sid
            rows.append({"split": tag, "index": i, "split_with_dev": next(iter(dev_of[t])),
                         "text": text,
                         "aspect_terms": [{"term": x["term"], "polarity": x["polarity"]} for x in target if "term" in x],
                         "aspect_categories": [{"category": x["category"], "polarity": x["polarity"]}
                                               for x in target if "category" in x],
                         "uzabsa_sentence_ids": sid})
    write_jsonl(out / "uzabsa_benchmark_split.jsonl", rows)

    test = [r for r in rows if r["split"] == "test"]
    clean = jl(V3 / "test_recheck/test_clean.jsonl")
    # The released predictions and the re-check workbooks were built from the text that
    # src/evaluation.parse_chatml_example returns. Its non-greedy pattern stops at the first
    # inner double quote, so a few reviews reached the systems shortened. Record both strings.
    inputs = [parse_eval(t)[0] for t in v2["validation"]["text"]]
    assert [c["text"] for c in clean] == inputs
    for r, x in zip(test, inputs):
        r["_input"] = x
    recheck = [{"index": r["index"], "text": r["text"], "input_text": r["_input"], "rechecked": c["rechecked"],
                "gold_terms_original": r["aspect_terms"], "gold_terms_rechecked": c["gold_aspects"]}
               for r, c in zip(test, clean)]
    write_jsonl(out / "uzabsa_test_rechecked.jsonl", recheck)
    for src, dst in [("changes.csv", "uzabsa_test_recheck_changes.csv"),
                     ("agreement.json", "uzabsa_test_recheck_agreement.json"),
                     ("summary.json", "uzabsa_test_recheck_summary.json"),
                     ("GUIDELINES_uz.md", "uzabsa_test_recheck_guidelines_uz.md")]:
        shutil.copy2(V3 / "test_recheck" / src, out / dst)

    preds = {name: jl(ROOT / path) for name, path in SYSTEMS}
    merged = []
    for r in test:
        i = r["index"]
        row = {"index": i, "text": r["text"], "input_text": r["_input"], "gold_terms_original": r["aspect_terms"],
               "gold_terms_rechecked": clean[i]["gold_aspects"], "predictions": {}, "parse_success": {}}
        for name, p in preds.items():
            assert p[i]["text"] == r["_input"] and p[i]["gold_aspects"] == r["aspect_terms"], (name, i)
            row["predictions"][name] = [{k: v for k, v in a.items() if k in ("term", "polarity", "votes")}
                                        for a in p[i]["pred_aspects"]]
            if "parse_success" in p[i] and not name.startswith(("bertbek", "tahrirchi", "vote")):
                row["parse_success"][name] = p[i]["parse_success"]
        merged.append(row)
    write_jsonl(out / "uzabsa_test_predictions.jsonl", merged)
    for r in test:
        del r["_input"]
    return rows, missing


def taxonomy(out):
    shutil.copy2(ROOT / "data/raw/absa_subcategories.json", out / "absa_subcategories.json")
    shutil.copy2(ROOT / "data/raw/business_categories.json", out / "business_categories.json")


README = """# UzABSA multi-domain: Uzbek business reviews with aspect-based sentiment annotations (version 4)

{n_reviews:,} Uzbek-language reviews of {n_businesses} businesses in {n_domains} domains, posted on the Uzbek review
platform Sharh between 7 March and 20 June 2025, with {n_aspects:,} aspect annotations (term, category, polarity).
Every review carries the 1-5 scores and written rationales of two language-model judges and a quality tier.
Native-speaker ratings of 150 reviews and human annotations of 80 reviews are included, together with benchmark files
derived from the UzABSA restaurant corpus (Hugging Face, doi:10.57967/hf/10264).

## Files

| File | Records | Content |
|---|---|---|
| `00_PREVIEW_uzbek_absa_sample.csv` | {n_preview} rows | First rows of every domain of the flat table (preview) |
| `uzbek_multi_domain_absa_full.json` | {n_reviews:,} reviews | All reviews with aspects, panel scores and quality tier |
| `uzbek_multi_domain_absa.jsonl` | {n_reviews:,} reviews | Same records, one per line |
| `uzbek_multi_domain_absa.csv` | {n_csv:,} rows | One row per aspect; review-level fields repeated |
| `uzbek_multi_domain_absa_approved.json` | {n_include:,} reviews | Include tier (panel overall score >= 3.5) |
| `reviews_flagged_for_review.json` | {n_flag:,} reviews | Flag tier (panel overall score 2.5 or 3.0) |
| `dataset_stats.json` | - | Counts by tier, domain, polarity and category; mean panel scores |
| `judge_panel_scores.csv` | {n_judge:,} rows | Scores of each judge (GPT-4o-mini, GPT-4.1) on five criteria, with rationale |
| `human_ratings_150.csv` | 150 reviews | 1-5 ratings and notes of two annotators; single-judge and panel scores |
| `uzbek_multi_domain_absa_gold80.json` | 80 reviews | Reconciled aspect-polarity annotations of two annotators and the model's aspects |
| `human_validation_protocol_uz.md` | - | Annotator instructions (Uzbek) |
| `uzabsa_benchmark_split.jsonl` | 6,089 examples | UzABSA examples, review-grouped train/test split (5,480/609) and train/dev/test split (4,932/548/609) |
| `uzabsa_test_rechecked.jsonl` | 609 examples | Test examples with original and re-checked gold terms |
| `uzabsa_test_recheck_changes.csv` | 281 rows | Individual label changes of the re-check |
| `uzabsa_test_recheck_agreement.json` | - | Agreement of the two re-check annotators |
| `uzabsa_test_recheck_summary.json` | - | Change counts, outcome per candidate type, system scores on both label versions |
| `uzabsa_test_recheck_guidelines_uz.md` | - | Re-check guidelines (Uzbek) |
| `uzabsa_test_predictions.jsonl` | 609 examples | Predictions of nine fine-tuned and zero-shot systems with both gold versions |
| `absa_subcategories.json` | 18 domains | 119 domain-specific sub-aspects |
| `business_categories.json` | 630 businesses | Business name and assigned domain |

## Main fields of a review record

`review_id`, `text`, `business_name`, `business_category` (one of 23 Uzbek domain labels), `user_rating` (1-5),
`aspects` (each with `term`, `category` from `ovqat` food, `xizmat` service, `muhit` ambience, `narx` price,
`boshqalar` other, and `polarity` positive/negative/neutral), `num_aspects`, `annotation_source`,
`judge_completeness`, `judge_accuracy`, `judge_sentiment`, `judge_relevance`, `judge_overall` (mean of the two
judges), `quality_tier` (`include`, `flag_for_review` or `exclude`).

## How the data were produced

Reviewer names, posting dates, like counts and links were removed and 20 exact duplicates dropped. A
Qwen2.5-7B-Instruct model fine-tuned with QLoRA on the UzABSA corpus annotated each review (greedy decoding). GPT-4o-mini
and GPT-4.1 scored every annotation on completeness, accuracy, sentiment, relevance and overall quality (1-5);
the mean overall score sets the tier: include >= 3.5, flag 2.5-3.0, exclude <= 2.0. Two native speakers rated 150
annotations on the same criteria (quadratic-weighted kappa 0.91-0.97) and annotated 80 reviews from scratch; the
panel's overall score correlates with the mean human rating at Spearman rho = 0.78. In the benchmark files, two
annotators re-checked 575 candidate terms in 263 UzABSA test reviews (kappa 0.94), changing 281 labels.

The labels outside `uzbek_multi_domain_absa_gold80.json` are model-generated (silver standard); use `quality_tier`
or the judge scores to select data.

## Loading

```python
import pandas as pd
df = pd.read_csv("uzbek_multi_domain_absa.csv")
include = df[df.quality_tier == "include"]
```

## Source, permission and licence

Reviews come from the public review platform Sharh (https://sharh.commeta.uz) and are used with the permission of
the platform owner, Commeta, for academic research. Released under CC BY-NC 4.0.
Code: https://github.com/SanatbekMatlatipov/UzABSA-LLM - Models: https://huggingface.co/Sanatbek/UzABSA-LLM
"""


def write_readme(out, full, n_csv):
    stats = json.loads((out / "dataset_stats.json").read_text(encoding="utf-8"))
    tiers = Counter(r["quality_tier"] for r in full)
    with open(out / "00_PREVIEW_uzbek_absa_sample.csv", encoding="utf-8-sig") as f:
        n_preview = sum(1 for _ in csv.reader(f)) - 1
    text = README.format(n_reviews=len(full), n_businesses=len({r["business_name"] for r in full}),
                         n_domains=stats["domains"], n_aspects=stats["total_aspects"], n_preview=n_preview,
                         n_csv=n_csv, n_include=tiers["include"], n_flag=tiers["flag_for_review"],
                         n_judge=2 * len(full))
    (out / "01_README.md").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uzabsa-jsonl", default=None,
                    help="local copy of the UzABSA jsonl; downloaded from the Hub (revision ff033e1) if omitted")
    ap.add_argument("--out", default=str(ROOT / "paper_materials/data-in-brief/zenodo_v4"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if args.uzabsa_jsonl is None:
        from huggingface_hub import hf_hub_download
        args.uzabsa_jsonl = hf_hub_download("Sanatbek/aspect-based-sentiment-analysis-uzbek",
                                            "aspect-based-sentiment-analysis-uzbek.jsonl",
                                            repo_type="dataset", revision="ff033e1")
    full, n_csv = corpus(out)
    n_judge, served = judges(out, full)
    hum = human(out, full)
    bench, missing = benchmark(out, args.uzabsa_jsonl)
    taxonomy(out)
    write_readme(out, full, n_csv)

    report = {"csv_rows": n_csv, "judge_rows": n_judge, "served_models": dict(served),
              "human_rows": len(hum), "benchmark_rows": len(bench),
              "benchmark_split": dict(Counter(r["split"] for r in bench)),
              "benchmark_split_with_dev": dict(Counter(r["split_with_dev"] for r in bench)),
              "benchmark_without_uzabsa_id": missing}
    files = []
    for p in sorted(out.iterdir()):
        files.append({"file": p.name, "bytes": p.stat().st_size,
                      "md5": hashlib.md5(p.read_bytes()).hexdigest()})
    report["files"] = files
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
