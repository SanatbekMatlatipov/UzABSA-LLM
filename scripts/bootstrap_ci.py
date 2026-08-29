#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Per-system bootstrap confidence intervals  (companion to P2-C)
# =============================================================================
"""
Percentile bootstrap 95% CIs for a single system's metrics on the evaluation split.

``significance_test.py`` answers "is A better than B?"; this answers "how precise is
any one of these numbers?", which is what a reader needs to judge a 0.02 gap on a
609-example set. Same metric implementations, same JSONL input format, so the CIs
are directly comparable to the point estimates in the results table.

Usage:
    python scripts/bootstrap_ci.py \
        --preds paper_materials/revision_v2/results_v2/qwen2.5-7b_preds.jsonl \
        --name "Qwen 2.5-7B" \
        --out paper_materials/revision_v2/significance_v2/qwen_ci.json

    python scripts/bootstrap_ci.py --selftest

Author: UzABSA Team
License: MIT
"""

import argparse
import importlib.util as ilu
import json
import random
from pathlib import Path

_st = ilu.spec_from_file_location(
    "uzabsa_sigtest", Path(__file__).parent / "significance_test.py")
_sig = ilu.module_from_spec(_st); _st.loader.exec_module(_sig)
metrics_on_indices = _sig.metrics_on_indices
load_jsonl = _sig.load_jsonl

KEYS = ["ate_exact_f1", "ate_partial_f1", "pair_f1", "sentiment_accuracy"]


def bootstrap_ci(preds, refs, n_boot=10000, seed=42):
    """Percentile bootstrap over reviews. Returns point estimate + 95% CI per metric."""
    n = len(preds)
    obs = metrics_on_indices(preds, refs, list(range(n)))
    rng = random.Random(seed)
    boot = {k: [] for k in KEYS}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        m = metrics_on_indices(preds, refs, idx)
        for k in KEYS:
            boot[k].append(m[k])
    out = {}
    for k in KEYS:
        v = sorted(boot[k])
        lo, hi = v[int(0.025 * n_boot)], v[int(0.975 * n_boot) - 1]
        out[k] = {
            "estimate": round(obs[k], 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "half_width": round((hi - lo) / 2, 4),
        }
    return {"n_examples": n, "n_boot": n_boot, "per_metric": out}


def _selftest():
    rng = random.Random(0)
    preds, refs = [], []
    for _ in range(200):
        gold = [{"term": f"t{rng.randrange(5)}", "polarity": rng.choice(["positive", "negative"])}]
        refs.append(gold)
        preds.append(gold if rng.random() < 0.8 else [])
    res = bootstrap_ci(preds, refs, n_boot=2000)
    pm = res["per_metric"]["pair_f1"]
    assert pm["ci95"][0] <= pm["estimate"] <= pm["ci95"][1], pm
    print("pair_f1:", pm)
    # a perfect system must have a degenerate CI at 1.0
    res2 = bootstrap_ci(refs, refs, n_boot=500)
    pm2 = res2["per_metric"]["pair_f1"]
    assert pm2["estimate"] == 1.0 and pm2["ci95"] == [1.0, 1.0], pm2
    print("perfect-system control:", pm2)
    print("\nSELFTEST PASSED - bootstrap CIs behave correctly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds")
    ap.add_argument("--name", default="system")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return

    rows = load_jsonl(args.preds)
    preds = [r["pred_aspects"] for r in rows]
    refs = [r["gold_aspects"] for r in rows]
    res = bootstrap_ci(preds, refs, n_boot=args.n_boot, seed=args.seed)
    res["system"] = args.name
    res["preds_file"] = args.preds

    print(f"\n{args.name}  (n={res['n_examples']}, B={res['n_boot']})")
    for k in KEYS:
        m = res["per_metric"][k]
        print(f"  {k:20s} {m['estimate']:.4f}  95% CI [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}]"
              f"  +/-{m['half_width']:.4f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
