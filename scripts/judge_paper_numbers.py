#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: the paper's judge-panel numbers (Table 4, Figs. 4-5, text)
# =============================================================================
"""
Computes every number the paper reports about the judge panel from the
panel's own outputs, so the figures and text are regenerated, not typed in:

  calibration  the 150 human-rated reviews, scored inside the corpus run whose
               tiers are released: tier agreement, kappa, the tier confusion
               matrix, Include precision/recall, per-dimension rho / MAE /
               means, and paired bootstrap tests against the published
               single-judge run. The separate calibration run is reported as
               a repeat measurement.
  corpus       tier counts and per-dimension means over every review
  domains      Fig. 4: the nine largest domains, the rest pooled, and all
               reviews, each with its mean overall score, 95% bootstrap CI and
               tier shares; plus every domain's own Include rate

Usage:
    python scripts/judge_paper_numbers.py \
        --corpus experiments/v3/judge/release_v1/judge_results.json \
        --repeat experiments/v3/judge/calibration_v1/raw \
        --out experiments/v3/results/judge_paper_numbers.json

Author: UzABSA Team
License: MIT
"""

import argparse
import importlib.util as ilu
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
_jp = ilu.spec_from_file_location("uzabsa_judge_panel", ROOT / "scripts" / "judge_panel.py")
_panel = ilu.module_from_spec(_jp); _jp.loader.exec_module(_panel)
DIMS, TIERS, tier = _panel.DIMS, ["include", "flag", "exclude"], _panel.tier

ENGLISH = {"Restoran/Ovqatlanish": "Restaurants", "Boshqa": "Other businesses",
           "Elektron tijorat": "E-commerce", "Tibbiyot/Sog'liqni saqlash": "Healthcare",
           "Oziq-ovqat do'konlari": "Grocery stores", "Bank/Moliya": "Banking",
           "To'lov tizimlari": "Payment systems", "Telekommunikatsiya": "Telecommunications",
           "Ta'lim": "Education"}


def calibration(scores, human, published):
    ids = [i for i in human if i in scores and i in published]
    out = _panel.agreement(scores, human, ids)
    jt = [tier(scores[i]["overall"]) for i in ids]
    ht = [tier(human[i]["overall"]) for i in ids]
    out["tiers"]["confusion_rows_judge_cols_human"] = [
        [sum(j == a and h == b for j, h in zip(jt, ht)) for b in TIERS] for a in TIERS]
    out["tiers"]["include_vs_exclude"] = sum({j, h} == {"include", "exclude"} for j, h in zip(jt, ht))
    return ids, out


def domains(rows, top=9, n_boot=10000, seed=42):
    by = defaultdict(list)
    for r in rows:
        by[r["business_category"]].append(r["judge_scores"])
    order = sorted(by, key=lambda d: -len(by[d]))
    groups = {ENGLISH.get(d, d): by[d] for d in order[:top]}
    groups[f"{len(order) - top} smaller domains"] = [s for d in order[top:] for s in by[d]]
    groups["All reviews"] = [s for d in order for s in by[d]]
    rng = np.random.default_rng(seed)

    def stats(scores):
        x = np.array([s["overall"] for s in scores])
        boots = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
        t = Counter(tier(v) for v in x)
        return {"n": len(x), "mean": round(float(x.mean()), 2),
                "ci95": [round(float(np.percentile(boots, q)), 2) for q in (2.5, 97.5)],
                "tier_pct": {k: round(100 * t[k] / len(x), 1) for k in TIERS},
                "accuracy_mean": round(float(np.mean([s["accuracy"] for s in scores])), 2)}
    per_domain = {d: {"n": len(by[d]),
                      "include_pct": round(100 * np.mean([s["overall"] >= 3.5 for s in by[d]]), 1),
                      "accuracy_mean": round(float(np.mean([s["accuracy"] for s in by[d]])), 2)}
                  for d in order}
    return {"groups": {g: stats(s) for g, s in groups.items()}, "per_domain": per_domain,
            "pooled_domains": order[top:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path,
                    default=ROOT / "experiments/v3/judge/release_v1/judge_results.json")
    ap.add_argument("--repeat", type=Path, default=ROOT / "experiments/v3/judge/calibration_v1/raw",
                    help="raw logs of the separate calibration run (repeat measurement)")
    ap.add_argument("--human-dir", type=Path, default=_panel.HV_DIR)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/v3/results/judge_paper_numbers.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    rows = json.load(open(args.corpus, encoding="utf-8"))
    corpus = {r["review_id"]: r["judge_scores"] for r in rows}
    human = _panel.load_human(args.human_dir)
    published = _panel.imported_scores({"file": "data/judged/judge_results.json"})

    ids, cal = calibration(corpus, human, published)
    systems = {"panel": corpus, "published single judge": published}
    cal["bootstrap"] = _panel.bootstrap(systems, human, ids, "published single judge", args.n_boot)
    _, cal_published = calibration(published, human, published)

    repeat = {}
    per_judge = {p.stem.split("__")[0]: {i: r["scores"] for i, r in _panel.load_raw(p).items()}
                 for p in sorted(args.repeat.glob("*.jsonl"))}
    if per_judge:
        _, rep = calibration(_panel.panel_mean(per_judge, ids), human, published)
        repeat = {"spearman_overall": rep["overall"]["spearman"], **{k: rep["tiers"][k] for k in
                  ("kappa", "kappa_linear", "agreement", "include_precision")}}
        # how often the two runs put a review in the same tier
        rep_scores = _panel.panel_mean(per_judge, ids)
        repeat["same_tier_as_corpus_run"] = round(float(np.mean(
            [tier(rep_scores[i]["overall"]) == tier(corpus[i]["overall"]) for i in ids])), 3)

    tiers = Counter(tier(s["overall"]) for s in corpus.values())
    report = {
        "source": {k: str(p.resolve().relative_to(ROOT.resolve()))
                   for k, p in (("corpus", args.corpus), ("repeat", args.repeat))},
        "calibration": {"n": len(ids), "panel_in_corpus_run": cal, "published_single_judge": cal_published,
                        "panel_repeat_run": repeat},
        "corpus": {"n": len(corpus), "tiers": {k: tiers[k] for k in TIERS},
                   "tier_pct": {k: round(100 * tiers[k] / len(corpus), 1) for k in TIERS},
                   "means": {d: round(float(np.mean([s[d] for s in corpus.values()])), 2) for d in DIMS}},
        "domains": domains(rows, n_boot=args.n_boot),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    t = cal["tiers"]
    b = cal["bootstrap"]["panel"]
    print(f"calibration n={len(ids)}: agreement {t['agreement']}  kappa {t['kappa']} "
          f"(CI {b['kappa']['ci95']}, vs published {b['kappa']['vs_published single judge']})  "
          f"linear {t['kappa_linear']}  incl. precision {t['include_precision']} recall {t['include_recall']}  "
          f"incl-vs-excl {t['include_vs_exclude']}")
    print(f"  rho overall {cal['overall']['spearman']} (CI {b['spearman_overall']['ci95']}, "
          f"vs published {b['spearman_overall']['vs_published single judge']})")
    print("  per dim:", {d: (cal[d]["spearman"], cal[d]["mae"], cal[d]["judge_mean"], cal[d]["human_mean"]) for d in DIMS})
    print("  confusion (rows judge I/F/E, cols human I/F/E):", t["confusion_rows_judge_cols_human"])
    print("  repeat run:", repeat)
    print("corpus:", report["corpus"])
    for g, s in report["domains"]["groups"].items():
        print(f"  {g:22s} n={s['n']:5d} mean={s['mean']} ci={s['ci95']} tiers={s['tier_pct']} acc={s['accuracy_mean']}")
    print("pooled:", {d: report["domains"]["per_domain"][d] for d in report["domains"]["pooled_domains"]})
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
