#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Human-Validation Sample & Template Builder  (P0-A)
# =============================================================================
"""
Builds the human-validation study materials from the 307 already-judged reviews
(``data/judged/judge_results.json``), so that native-speaker annotation can begin
immediately and, crucially, OVERLAP the GPT-4o-mini-judged items (required to
correlate human vs. judge scores on the same reviews).

Outputs (all under paper_materials/revision_v2/human_validation/):
  - sample_150.csv        Full metadata + model predictions + judge scores.
                          Internal join file for analysis (NOT given to annotators).
  - rubric_template.csv   Annotator-facing. One row per sampled review with the
                          review text + the MODEL's predicted aspects (readable) and
                          five empty 1-5 rating columns (same rubric as the LLM judge).
  - gold_template.csv     Annotator-facing subset (~80). Blank ``gold_aspects`` column
                          for writing the correct aspects from scratch.

The sample is deterministic (seed 42): every domain is represented, small domains
(n<=5) are taken in full, and low-scoring / distant domains are oversampled so the
study stresses exactly the domains the reviewers doubt.

Usage:
    python scripts/build_human_validation.py \
        --judged data/judged/judge_results.json \
        --out-dir paper_materials/revision_v2/human_validation \
        --rubric-size 150 --gold-size 80 --seed 42

Author: UzABSA Team
License: MIT
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# Domains whose judged mean-overall is < 3.5 (below the inclusion threshold) OR that
# are conceptually distant from the restaurant training domain. These are oversampled
# because they are where the cross-domain / silver-quality claims are weakest (R2, R3, R6).
PRIORITY_DOMAINS = {
    "Sug'urta",                 # insurance   (overall 2.67)
    "Investitsiya/Trading",     # investment  (overall 2.67)
    "Ta'lim",                   # education   (overall 3.25)
    "Sport/Fitnes",             # gyms        (overall 3.25)
    "Kitob/Nashriyot",          # books       (overall 3.00)
    "Davlat xizmatlari",        # government  (overall 3.33)
    "Telekommunikatsiya",       # telecom     (overall 3.32)
    "To'lov tizimlari",         # payments    (overall 3.54)
    "Bank/Moliya",              # banking     (overall 3.70)
}

SCORE_DIMS = ["completeness", "accuracy", "sentiment", "relevance", "overall"]


def aspects_to_readable(aspects):
    """Render model-predicted aspects as a compact, annotator-legible string."""
    parts = []
    for a in aspects or []:
        term = (a.get("term") or "").strip()
        if not term:
            continue
        cat = (a.get("category") or "").strip()
        pol = (a.get("polarity") or "").strip()
        parts.append(f"{term} [{cat}] -> {pol}")
    return " ;; ".join(parts) if parts else "(model predicted no aspects)"


def stratified_sample(records, target, seed, priority_domains):
    """Deterministically pick `target` records spread across all domains.

    Rules: (1) every domain appears; (2) domains with <=5 records are taken in full;
    (3) remaining budget is allocated proportionally to domain size but with a x1.6
    weight on priority (low-score / distant) domains; (4) reproducible under `seed`.
    """
    rng = random.Random(seed)
    by_dom = defaultdict(list)
    for r in records:
        by_dom[r["business_category"]].append(r)
    for dom in by_dom:
        by_dom[dom].sort(key=lambda r: r["review_id"])  # stable order before shuffle
        rng.shuffle(by_dom[dom])

    domains = sorted(by_dom.keys())
    chosen = {}

    # Pass 1: guarantee small domains in full, and at least 3 from every domain.
    for dom in domains:
        recs = by_dom[dom]
        take = len(recs) if len(recs) <= 5 else min(3, len(recs))
        chosen[dom] = recs[:take]

    # Pass 2: distribute the remaining budget by weighted size.
    used = sum(len(v) for v in chosen.values())
    remaining = max(0, target - used)
    weights = {}
    for dom in domains:
        left = len(by_dom[dom]) - len(chosen[dom])
        if left <= 0:
            weights[dom] = 0.0
            continue
        w = left * (1.6 if dom in priority_domains else 1.0)
        weights[dom] = w
    total_w = sum(weights.values())
    if total_w > 0 and remaining > 0:
        # Largest-remainder allocation for determinism.
        raw = {d: remaining * (w / total_w) for d, w in weights.items()}
        alloc = {d: int(v) for d, v in raw.items()}
        leftover = remaining - sum(alloc.values())
        # hand out leftover slots to the largest fractional parts
        frac_order = sorted(domains, key=lambda d: (raw[d] - alloc[d]), reverse=True)
        for d in frac_order[:leftover]:
            alloc[d] += 1
        for dom in domains:
            extra = min(alloc.get(dom, 0), len(by_dom[dom]) - len(chosen[dom]))
            if extra > 0:
                start = len(chosen[dom])
                chosen[dom].extend(by_dom[dom][start:start + extra])

    flat = [r for dom in domains for r in chosen[dom]]
    flat.sort(key=lambda r: r["review_id"])
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default="data/judged/judge_results.json")
    ap.add_argument("--out-dir", default="paper_materials/revision_v2/human_validation")
    ap.add_argument("--rubric-size", type=int, default=150)
    ap.add_argument("--gold-size", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = json.load(open(args.judged, encoding="utf-8"))
    print(f"Loaded {len(records)} judged records")

    rubric = stratified_sample(records, args.rubric_size, args.seed, PRIORITY_DOMAINS)
    print(f"Rubric sample: {len(rubric)} reviews across "
          f"{len({r['business_category'] for r in rubric})} domains")

    # Gold subset: a deterministic sub-sample of the rubric set, itself oversampling
    # priority domains (gold matters most where the model is weakest).
    rng = random.Random(args.seed + 1)
    prio = [r for r in rubric if r["business_category"] in PRIORITY_DOMAINS]
    rest = [r for r in rubric if r["business_category"] not in PRIORITY_DOMAINS]
    rng.shuffle(prio); rng.shuffle(rest)
    gold = (prio + rest)[: args.gold_size]
    gold.sort(key=lambda r: r["review_id"])
    gold_ids = {r["review_id"] for r in gold}
    print(f"Gold subset: {len(gold)} reviews "
          f"({sum(1 for r in gold if r['business_category'] in PRIORITY_DOMAINS)} from priority domains)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- sample_150.csv : internal metadata + join keys (NOT for annotators) ---
    with open(out / "sample_150.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "business_category", "in_gold_subset", "text",
                    "model_aspects_readable", "model_aspects_json"]
                   + [f"judge_{d}" for d in SCORE_DIMS])
        for r in rubric:
            js = r.get("judge_scores", {})
            w.writerow([
                r["review_id"], r["business_category"],
                "yes" if r["review_id"] in gold_ids else "no",
                r["text"], aspects_to_readable(r.get("aspects")),
                json.dumps(r.get("aspects", []), ensure_ascii=False),
            ] + [js.get(d, "") for d in SCORE_DIMS])

    # --- rubric_template.csv : annotator rates the MODEL predictions 1-5 ---
    with open(out / "rubric_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "business_category", "review_text",
                    "model_predicted_aspects",
                    "completeness_1_5", "accuracy_1_5", "sentiment_1_5",
                    "relevance_1_5", "overall_1_5", "notes"])
        for r in rubric:
            w.writerow([r["review_id"], r["business_category"], r["text"],
                        aspects_to_readable(r.get("aspects")),
                        "", "", "", "", "", ""])

    # --- gold_template.csv : annotator writes correct aspects from scratch ---
    with open(out / "gold_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "business_category", "review_text",
                    "gold_aspects", "notes"])
        # two worked examples first so annotators see the exact format
        w.writerow(["EXAMPLE_1", "Restoran/Ovqatlanish",
                    "Ovqatlari mazali edi lekin narxi qimmat.",
                    "ovqat :: positive ;; narx :: negative", "example row - delete"])
        w.writerow(["EXAMPLE_2", "Bank/Moliya",
                    "Ilova sekin ishlaydi, xizmat yaxshi.",
                    "ilova :: negative ;; xizmat :: positive", "example row - delete"])
        for r in gold:
            w.writerow([r["review_id"], r["business_category"], r["text"], "", ""])

    print(f"\nWrote:\n  {out/'sample_150.csv'}\n  {out/'rubric_template.csv'}"
          f"\n  {out/'gold_template.csv'}")


if __name__ == "__main__":
    main()
