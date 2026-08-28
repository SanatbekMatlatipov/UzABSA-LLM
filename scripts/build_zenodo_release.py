#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Zenodo Release Bundler
# =============================================================================
"""
Builds the exact set of files to upload to the Zenodo record, fixing the
"Cannot preview file" problem on the record landing page.

Why this exists
---------------
Zenodo (InvenioRDM) previews only the FIRST previewable file in alphanumeric
order, unless a file is explicitly ticked in the "Preview" column. Its JSON/text
previewer refuses files above ~1 MB, which is why the 3.3 MB
`uzbek_multi_domain_absa_full.json` renders as "Cannot preview file". Its CSV
previewer, by contrast, renders an interactive sortable table.

So the release bundle leads with a small CSV whose name sorts first, giving the
landing page a readable table on first open, and keeps the full JSON/JSONL
alongside it for actual use.

Output (to --out, default paper_materials/revision_v2/zenodo_release/):
    00_PREVIEW_uzbek_absa_sample.csv   small, always-previewable table (leads the page)
    01_README.md                       human-readable record description
    uzbek_multi_domain_absa.csv        full flattened dataset, one row per aspect
    uzbek_multi_domain_absa_full.json  \
    uzbek_multi_domain_absa_silver.json > copied verbatim from data/final_dataset/
    uzbek_multi_domain_absa.jsonl      /
    uzbek_multi_domain_absa_gold80.json
    dataset_stats.json

Usage:
    python scripts/build_zenodo_release.py
    python scripts/build_zenodo_release.py --preview-rows 300

Author: UzABSA Team
License: MIT
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

# Files copied through untouched, in the order they should appear in the record.
PASSTHROUGH = [
    "uzbek_multi_domain_absa.jsonl",
    "uzbek_multi_domain_absa_full.json",
    "uzbek_multi_domain_absa_silver.json",
    "uzbek_multi_domain_absa_approved.json",
    "uzbek_multi_domain_absa_gold80.json",
    "dataset_stats.json",
]

CSV_COLUMNS = [
    "review_id", "business_category", "business_name", "user_rating",
    "aspect_term", "aspect_category", "polarity",
    "quality_tier", "judge_overall", "human_verified", "annotation_source",
    "review_text",
]


def flatten(records):
    """One row per aspect, carrying its review's context."""
    for r in records:
        aspects = r.get("aspects") or []
        base = {
            "review_id": r.get("review_id", ""),
            "business_category": r.get("business_category", ""),
            "business_name": r.get("business_name", ""),
            "user_rating": r.get("user_rating", ""),
            "quality_tier": r.get("quality_tier", ""),
            "judge_overall": r.get("judge_overall", ""),
            "human_verified": r.get("human_verified", ""),
            "annotation_source": r.get("annotation_source", ""),
            "review_text": (r.get("text") or "").replace("\n", " ").strip(),
        }
        if not aspects:
            yield {**base, "aspect_term": "", "aspect_category": "", "polarity": ""}
            continue
        for a in aspects:
            yield {
                **base,
                "aspect_term": a.get("term", ""),
                "aspect_category": a.get("category", ""),
                "polarity": a.get("polarity", ""),
            }


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_preview(all_rows, n_rows):
    """A stratified slice: keeps every business category represented, so the
    landing-page table shows the multi-domain nature rather than the first
    domain alphabetically."""
    by_cat = {}
    for r in all_rows:
        by_cat.setdefault(r["business_category"], []).append(r)
    cats = sorted(by_cat)
    per_cat = max(1, n_rows // max(1, len(cats)))
    out = []
    for c in cats:
        out.extend(by_cat[c][:per_cat])
    return out[:n_rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/final_dataset")
    ap.add_argument("--out", default="./paper_materials/revision_v2/zenodo_release")
    ap.add_argument("--preview-rows", type=int, default=300,
                    help="rows in the small always-previewable CSV")
    ap.add_argument("--version", default="v2",
                    help="version label written into the README")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    full = json.loads((data / "uzbek_multi_domain_absa_full.json").read_text(encoding="utf-8"))
    stats = json.loads((data / "dataset_stats.json").read_text(encoding="utf-8"))

    rows = list(flatten(full))

    # Full flattened CSV
    full_csv = out / "uzbek_multi_domain_absa.csv"
    write_csv(full_csv, rows)

    # Small preview CSV — named to sort first so Zenodo picks it by default
    prev_csv = out / "00_PREVIEW_uzbek_absa_sample.csv"
    write_csv(prev_csv, build_preview(rows, args.preview_rows))

    for name in PASSTHROUGH:
        src = data / name
        if src.exists():
            shutil.copy2(src, out / name)

    n_reviews = len(full)
    n_aspects = sum(len(r.get("aspects") or []) for r in full)
    n_gold = 0
    gold_path = data / "uzbek_multi_domain_absa_gold80.json"
    if gold_path.exists():
        n_gold = len(json.loads(gold_path.read_text(encoding="utf-8")))
    cats = sorted({r.get("business_category", "") for r in full if r.get("business_category")})

    readme = out / "01_README.md"
    readme.write_text(f"""# UzABSA Multi-Domain: an Uzbek Aspect-Based Sentiment Analysis dataset ({args.version})

{n_reviews:,} Uzbek-language business reviews spanning {len(cats)} domains, annotated with
{n_aspects:,} aspect-sentiment pairs, each carrying explicit quality metadata.
A {n_gold}-review subset is verified against native-speaker gold annotations.

## Start here

| File | What it is |
|---|---|
| `00_PREVIEW_uzbek_absa_sample.csv` | Small stratified sample. This is what previews on this page. |
| `uzbek_multi_domain_absa.csv` | **Full dataset as a flat table** — one row per aspect. Easiest for spreadsheets/pandas. |
| `uzbek_multi_domain_absa_full.json` | Full dataset, nested (aspects grouped per review) + all judge scores. |
| `uzbek_multi_domain_absa_silver.json` | Silver-standard subset: judge-included plus unjudged reviews. |
| `uzbek_multi_domain_absa_approved.json` | Judge-approved only (overall score >= 3.5). |
| `uzbek_multi_domain_absa_gold80.json` | {n_gold} reviews re-annotated from scratch by two native speakers. |
| `uzbek_multi_domain_absa.jsonl` | JSONL, for `datasets.load_dataset("json", ...)`. |
| `dataset_stats.json` | Corpus statistics. |

## Columns in the CSV files

`review_id`, `business_category`, `business_name`, `user_rating`,
`aspect_term`, `aspect_category`, `polarity`, `quality_tier`, `judge_overall`,
`human_verified`, `annotation_source`, `review_text`

- **aspect_category** is one of `ovqat` (food), `muhit` (ambiance), `xizmat` (service),
  `narx` (price), `boshqalar` (other).
- **polarity** is one of `positive`, `negative`, `neutral`, `conflict`.
- **quality_tier** is `include` (judge overall >= 3.5), `flag` (2.5-3.49),
  `exclude` (< 2.5), or `unjudged` (not in the judged sample).

## How it was built

Reviews were annotated by a QLoRA-fine-tuned Qwen 2.5-7B model, then a stratified
sample of 307 reviews was scored by GPT-4o-mini on a five-dimension rubric
(completeness, accuracy, sentiment, relevance, overall). Those scores drive the
`quality_tier` field. The judge itself was calibrated against two native-speaker
annotators on 150 reviews, and {n_gold} reviews were re-annotated from scratch to
provide human gold references.

**These are silver-standard annotations.** Everything outside
`uzbek_multi_domain_absa_gold80.json` is model-generated, with quality estimated
rather than verified. Use `quality_tier` to filter.

## Loading it

```python
import pandas as pd
df = pd.read_csv("uzbek_multi_domain_absa.csv")
high_quality = df[df.quality_tier.isin(["include", "unjudged"])]
```

## Domains

{", ".join(cats)}

## Source and permissions

Reviews come from the public Uzbek business review platform
[sharh.commeta.uz](https://sharh.commeta.uz/en), used for academic research with the
explicit permission of the platform owner.

## Citation

Accompanies the UzABSA-LLM study of parameter-efficient fine-tuning for Uzbek ABSA.
Code: <https://github.com/SanatbekMatlatipov/UzABSA-LLM> ·
Models: <https://huggingface.co/Sanatbek/UzABSA-LLM>

License: CC BY 4.0
""", encoding="utf-8")

    print(f"Zenodo release bundle -> {out}\n")
    total = 0
    for p in sorted(out.iterdir()):
        kb = p.stat().st_size / 1024
        total += kb
        flag = ""
        if p.suffix in {".json", ".jsonl", ".md"} and kb > 1024:
            flag = "  <-- over ~1MB: will NOT preview on Zenodo (fine, not the lead file)"
        print(f"  {p.name:42s} {kb:9.1f} KB{flag}")
    print(f"\n  {'TOTAL':42s} {total/1024:9.1f} MB")
    print(f"\nReviews: {n_reviews:,} | aspects: {n_aspects:,} | domains: {len(cats)} | gold: {n_gold}")
    print("\nUpload ALL of the above to Zenodo as a New Version.")
    print("On the upload form, tick the 'Preview' checkbox next to")
    print("00_PREVIEW_uzbek_absa_sample.csv so it renders as the landing-page table.")


if __name__ == "__main__":
    main()
