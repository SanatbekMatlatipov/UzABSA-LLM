#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Human-Validation Analysis  (P0-A)
# =============================================================================
"""
Turns returned annotator files into the manuscript's human-validation numbers:

  1. Inter-annotator agreement (IAA) — quadratic-weighted Cohen's kappa and
     Krippendorff's alpha (interval metric) per rubric dimension; aspect/polarity
     agreement on the gold task.
  2. LLM-as-Judge validation (R6) — Spearman rho + MAE between mean human rubric
     scores and GPT-4o-mini scores, per dimension, on the same reviews.
  3. Silver-quality validation (R3) — model-vs-human ATE (exact/partial) and pair
     F1 on the gold subset, reusing src.evaluation metric functions.
  4. Per-domain-proximity breakdown (R2).

Inputs (from paper_materials/revision_v2/human_validation/):
  - sample_150.csv                       (built by build_human_validation.py; has judge_* + model_aspects_json)
  - returned/rubric_<name>.csv           one per annotator (1-5 ratings of model preds)
  - returned/gold_<name>.csv             one per annotator (gold_aspects free-text)

Output:
  - paper_materials/revision_v2/results/human_validation_report.json
  - printed summary (also paste into REVISION_LOG.md)

Smoke test (no real annotations yet):
  python scripts/analyze_human_validation.py --smoketest
    -> synthesizes two fake annotators, runs the full pipeline, writes to a temp dir,
       and confirms every metric computes. Use to verify the pipeline before real data.

Real run:
  python scripts/analyze_human_validation.py

Author: UzABSA Team
License: MIT
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Import the metric functions directly from src/evaluation.py WITHOUT triggering the
# heavy src/__init__.py (which imports datasets/huggingface_hub). The metric functions
# only need numpy/sklearn, so this keeps the analysis runnable in a lightweight env.
import importlib.util as _ilu  # noqa: E402
_eval_path = Path(__file__).parent.parent / "src" / "evaluation.py"
_spec = _ilu.spec_from_file_location("uzabsa_evaluation", _eval_path)
_evaluation = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_evaluation)
compute_ate_metrics = _evaluation.compute_ate_metrics
compute_aspect_polarity_metrics = _evaluation.compute_aspect_polarity_metrics

RUBRIC_DIMS = ["completeness", "accuracy", "sentiment", "relevance", "overall"]
VALID_POLARITIES = {"positive", "negative", "neutral", "conflict"}
# Domain proximity groups for the R2 breakdown.
PROXIMITY = {
    "in": {"Restoran/Ovqatlanish"},
    "near": {"Sayohat/Turizm", "Oziq-ovqat do'konlari", "Gul/Sovg'a", "Bozor/BSC",
             "Yetkazib berish", "Go'zallik"},
    "distant": {"Sug'urta", "Investitsiya/Trading", "Ta'lim", "Sport/Fitnes",
                "Kitob/Nashriyot", "Davlat xizmatlari"},
}


def proximity_of(domain):
    for grp, doms in PROXIMITY.items():
        if domain in doms:
            return grp
    return "out"  # everything else = out-of-domain


# ------------------------- parsing helpers -------------------------

def parse_gold_aspects(cell):
    """Parse a `term :: polarity ;; ...` free-text cell into aspect dicts. Lenient."""
    cell = (cell or "").strip()
    if not cell:
        return []
    aspects = []
    # accept ' ;; ' or ';;' or single ';' as pair separators
    for chunk in re.split(r"\s*;;\s*|\s*;\s*", cell):
        chunk = chunk.strip()
        if not chunk:
            continue
        # accept '::' or ':' between term and polarity
        m = re.split(r"\s*::\s*|\s*:\s*", chunk, maxsplit=1)
        term = m[0].strip()
        pol = (m[1].strip().lower() if len(m) > 1 else "neutral")
        if pol not in VALID_POLARITIES:
            pol = "neutral"
        if term:
            aspects.append({"term": term, "polarity": pol})
    return aspects


def read_rubric_csv(path):
    """Return {review_id: {dim: int}} skipping blank/unparseable ratings."""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rid = (row.get("review_id") or "").strip()
            if not rid or rid.startswith("EXAMPLE"):
                continue
            scores = {}
            for dim in RUBRIC_DIMS:
                v = (row.get(f"{dim}_1_5") or "").strip()
                if v:
                    try:
                        iv = int(float(v))
                        if 1 <= iv <= 5:
                            scores[dim] = iv
                    except ValueError:
                        pass
            if scores:
                out[rid] = scores
    return out


def read_gold_csv(path):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rid = (row.get("review_id") or "").strip()
            if not rid or rid.startswith("EXAMPLE"):
                continue
            out[rid] = parse_gold_aspects(row.get("gold_aspects"))
    return out


# ------------------------- agreement metrics -------------------------

def cohen_weighted_kappa(a, b):
    """Quadratic-weighted Cohen's kappa for two lists of 1-5 ratings."""
    try:
        from sklearn.metrics import cohen_kappa_score
        return round(float(cohen_kappa_score(a, b, weights="quadratic")), 4)
    except Exception:
        return None


def krippendorff_alpha(matrix, metric="interval"):
    """Krippendorff's alpha via the coincidence-matrix method. Self-contained.

    `matrix`: list of units, each a list of coder ratings (ints); None = missing.
    Uses the interval difference metric delta^2(c,k) = (c-k)^2 (appropriate for a
    1-5 Likert scale). Returns 1.0 on perfect agreement, ~0 at chance, <0 below chance.
    """
    # coincidence counts o[(c,k)] built from within-unit ordered pairs, each weighted
    # by 1/(m_u - 1) where m_u = number of ratings on that unit.
    o = defaultdict(float)
    for unit in matrix:
        ratings = [v for v in unit if v is not None]
        m = len(ratings)
        if m < 2:
            continue
        w = 1.0 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(ratings[i], ratings[j])] += w
    if not o:
        return None
    values = sorted({c for pair in o for c in pair})
    n_c = {c: sum(o[(c, k)] for k in values) for c in values}  # marginals
    n = sum(n_c.values())
    if n < 2:
        return None

    def d2(a, b):
        return (a - b) ** 2  # interval metric

    Do = sum(o[(c, k)] * d2(c, k) for c in values for k in values) / n
    De = sum(n_c[c] * n_c[k] * d2(c, k) for c in values for k in values) / (n * (n - 1))
    if De == 0:
        return None
    return round(1 - Do / De, 4)


def spearman(x, y):
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return round(float(r), 4), round(float(p), 4)
    except Exception:
        return None, None


# ------------------------- core analysis -------------------------

def load_sample(sample_csv):
    meta = {}
    with open(sample_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rid = row["review_id"]
            meta[rid] = {
                "domain": row["business_category"],
                "in_gold": row["in_gold_subset"] == "yes",
                "model_aspects": json.loads(row["model_aspects_json"] or "[]"),
                "judge": {d: (int(row[f"judge_{d}"]) if row.get(f"judge_{d}", "").strip() else None)
                          for d in RUBRIC_DIMS},
            }
    return meta


def analyze(sample_csv, returned_dir, out_json):
    meta = load_sample(sample_csv)
    returned = Path(returned_dir)
    rubric_files = sorted(returned.glob("rubric_*.csv"))
    gold_files = sorted(returned.glob("gold_*.csv"))

    rubrics = {p.stem.replace("rubric_", ""): read_rubric_csv(p) for p in rubric_files}
    golds = {p.stem.replace("gold_", ""): read_gold_csv(p) for p in gold_files}
    report = {"annotators_rubric": list(rubrics), "annotators_gold": list(golds),
              "n_reviews_meta": len(meta)}

    # ---- 1. IAA on rubric (needs >=2 annotators) ----
    iaa = {}
    if len(rubrics) >= 2:
        names = list(rubrics)[:2]
        a, b = rubrics[names[0]], rubrics[names[1]]
        common = sorted(set(a) & set(b))
        for dim in RUBRIC_DIMS:
            xa = [a[r][dim] for r in common if dim in a[r] and dim in b[r]]
            xb = [b[r][dim] for r in common if dim in a[r] and dim in b[r]]
            if len(xa) >= 2:
                matrix = [[va, vb] for va, vb in zip(xa, xb)]
                iaa[dim] = {
                    "n": len(xa),
                    "weighted_cohen_kappa": cohen_weighted_kappa(xa, xb),
                    "krippendorff_alpha": krippendorff_alpha(matrix),
                    "spearman_rho": spearman(xa, xb)[0],
                }
    report["iaa_rubric"] = iaa or "insufficient annotators (need 2 rubric files)"

    # ---- 2. Judge validation: mean human rubric vs GPT-4o-mini ----
    judge_val = {}
    if rubrics:
        # mean human score per review per dim
        human_mean = defaultdict(dict)
        all_rids = set().union(*[set(r) for r in rubrics.values()])
        for rid in all_rids:
            for dim in RUBRIC_DIMS:
                vals = [rubrics[n][rid][dim] for n in rubrics
                        if rid in rubrics[n] and dim in rubrics[n][rid]]
                if vals:
                    human_mean[rid][dim] = sum(vals) / len(vals)
        for dim in RUBRIC_DIMS:
            hv, jv = [], []
            for rid, hm in human_mean.items():
                j = meta.get(rid, {}).get("judge", {}).get(dim)
                if dim in hm and j is not None:
                    hv.append(hm[dim]); jv.append(j)
            if len(hv) >= 3:
                rho, p = spearman(hv, jv)
                mae = round(sum(abs(a - b) for a, b in zip(hv, jv)) / len(hv), 4)
                judge_val[dim] = {"n": len(hv), "spearman_rho": rho, "p": p, "mae": mae}
    report["judge_validation"] = judge_val or "no rubric data"

    # ---- 3. Silver-quality: model vs human gold (ATE + pair F1) ----
    silver = {}
    for name, gold in golds.items():
        rids = [r for r in gold if r in meta]
        preds = [[a for a in meta[r]["model_aspects"] if a.get("term")] for r in rids]
        refs = [[a for a in gold[r]] for r in rids]
        pred_terms = [[a["term"] for a in p] for p in preds]
        ref_terms = [[a["term"] for a in rr] for rr in refs]
        silver[name] = {
            "n_reviews": len(rids),
            "ate_exact": compute_ate_metrics(pred_terms, ref_terms),
            "ate_partial": compute_ate_metrics(pred_terms, ref_terms, partial_match=True),
            "pair": compute_aspect_polarity_metrics(preds, refs),
        }
    report["silver_quality_model_vs_human"] = silver or "no gold data"

    # gold-vs-gold ceiling (how much annotators agree on aspects)
    if len(golds) >= 2:
        names = list(golds)[:2]
        ga, gb = golds[names[0]], golds[names[1]]
        common = [r for r in ga if r in gb]
        pa = [[a for a in ga[r]] for r in common]
        pb = [[a for a in gb[r]] for r in common]
        ta = [[a["term"] for a in x] for x in pa]
        tb = [[a["term"] for a in x] for x in pb]
        report["gold_iaa_ceiling"] = {
            "n_reviews": len(common),
            "ate_exact_a_vs_b": compute_ate_metrics(ta, tb),
            "pair_a_vs_b": compute_aspect_polarity_metrics(pa, pb),
        }

    # ---- 4. Per-proximity model-vs-human (uses first gold annotator) ----
    if golds:
        name = list(golds)[0]
        gold = golds[name]
        by_prox = defaultdict(lambda: {"preds": [], "refs": []})
        for r in gold:
            if r not in meta:
                continue
            grp = proximity_of(meta[r]["domain"])
            by_prox[grp]["preds"].append([a for a in meta[r]["model_aspects"] if a.get("term")])
            by_prox[grp]["refs"].append(list(gold[r]))
        prox_report = {}
        for grp, d in by_prox.items():
            pt = [[a["term"] for a in p] for p in d["preds"]]
            rt = [[a["term"] for a in r] for r in d["refs"]]
            prox_report[grp] = {"n": len(d["preds"]),
                                "ate_exact": compute_ate_metrics(pt, rt),
                                "pair": compute_aspect_polarity_metrics(d["preds"], d["refs"])}
        report["per_proximity_model_vs_human"] = prox_report

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out_json}")
    return report


# ------------------------- smoke test -------------------------

def make_smoketest(base):
    """Synthesize two fake annotators from sample_150.csv to verify the pipeline."""
    base = Path(base)
    ret = base / "returned"
    ret.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    rows = list(csv.DictReader(open(base / "sample_150.csv", encoding="utf-8-sig")))
    for name in ("smoke1", "smoke2"):
        with open(ret / f"rubric_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["review_id", "business_category", "review_text", "model_predicted_aspects",
                        "completeness_1_5", "accuracy_1_5", "sentiment_1_5", "relevance_1_5",
                        "overall_1_5", "notes"])
            for r in rows:
                # fake ratings loosely correlated with the judge scores
                def jitter(dim):
                    base_v = int(r[f"judge_{dim}"]) if r.get(f"judge_{dim}", "").strip() else 3
                    return max(1, min(5, base_v + rng.choice([-1, 0, 0, 1])))
                w.writerow([r["review_id"], r["business_category"], r["text"],
                            r["model_aspects_readable"]]
                           + [jitter(d) for d in RUBRIC_DIMS] + [""])
        with open(ret / f"gold_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["review_id", "business_category", "review_text", "gold_aspects", "notes"])
            for r in rows:
                if r["in_gold_subset"] != "yes":
                    continue
                # fake gold = model aspects with occasional polarity flip / drop
                asp = json.loads(r["model_aspects_json"] or "[]")
                parts = []
                for a in asp:
                    if not a.get("term"):
                        continue
                    if rng.random() < 0.2:
                        continue
                    pol = a.get("polarity", "neutral")
                    if rng.random() < 0.15:
                        pol = rng.choice(list(VALID_POLARITIES))
                    parts.append(f"{a['term']} :: {pol}")
                w.writerow([r["review_id"], r["business_category"], r["text"],
                            " ;; ".join(parts), ""])
    print(f"Smoke-test annotator files written to {ret}")


def main():
    ap = argparse.ArgumentParser()
    hv = "paper_materials/revision_v2/human_validation"
    ap.add_argument("--sample", default=f"{hv}/sample_150.csv")
    ap.add_argument("--returned", default=f"{hv}/returned")
    ap.add_argument("--out", default="paper_materials/revision_v2/results/human_validation_report.json")
    ap.add_argument("--smoketest", action="store_true",
                    help="synthesize fake annotators and run end-to-end")
    args = ap.parse_args()

    if args.smoketest:
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        shutil.copy(args.sample, tmp / "sample_150.csv")
        make_smoketest(tmp)
        analyze(tmp / "sample_150.csv", tmp / "returned",
                tmp / "human_validation_report.json")
        shutil.rmtree(tmp)
        print("\nSMOKE TEST PASSED — pipeline runs end-to-end.")
    else:
        analyze(args.sample, args.returned, args.out)


if __name__ == "__main__":
    main()
