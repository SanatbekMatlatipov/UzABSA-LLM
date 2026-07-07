#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Paired Bootstrap Significance Testing  (P2-C)
# =============================================================================
"""
Paired bootstrap significance tests over the 609-example evaluation set, so the
paper can replace bold "winner" claims with statistically honest statements — most
importantly the Qwen-vs-Llama pair-F1 gap of 0.001, which is almost certainly a tie.

Given two systems' per-example predictions (+ shared gold), it resamples reviews
with replacement B times, recomputes each metric and the paired delta, and reports
the observed delta, a 95% CI, and a two-sided bootstrap p-value for delta == 0.

Input format: JSONL, one object per review, aligned by line order across systems:
    {"text": ..., "pred_aspects": [{"term","polarity"}...], "gold_aspects": [...]}
`eval_bert_baseline.py` already emits exactly this. For the LLMs, regenerate the
same file with the (to-be-added) --save-preds path in the LLM eval (P2-B), or via
`scripts/dump_llm_preds.py`.

Metrics compared: ate_exact_f1, ate_partial_f1, pair_f1, sentiment_accuracy.

Usage:
    python scripts/significance_test.py \
        --a paper_materials/revision_v2/results/qwen_preds.jsonl --a-name Qwen \
        --b paper_materials/revision_v2/results/llama_preds.jsonl --b-name Llama \
        --out paper_materials/revision_v2/significance/qwen_vs_llama.json

    python scripts/significance_test.py --selftest

Author: UzABSA Team
License: MIT
"""

import argparse
import importlib.util as ilu
import json
import random
from pathlib import Path

_ev = ilu.spec_from_file_location(
    "uzabsa_evaluation", Path(__file__).parent.parent / "src" / "evaluation.py")
_evaluation = ilu.module_from_spec(_ev); _ev.loader.exec_module(_evaluation)
compute_ate_metrics = _evaluation.compute_ate_metrics
compute_aspect_polarity_metrics = _evaluation.compute_aspect_polarity_metrics


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def metrics_on_indices(preds, refs, idx):
    p = [preds[i] for i in idx]
    r = [refs[i] for i in idx]
    pt = [[a["term"] for a in x if a.get("term")] for x in p]
    rt = [[a["term"] for a in x if a.get("term")] for x in r]
    pw = [[a for a in x if a.get("term")] for x in p]
    rw = [[a for a in x if a.get("term")] for x in r]
    ate_e = compute_ate_metrics(pt, rt)
    ate_p = compute_ate_metrics(pt, rt, partial_match=True)
    pair = compute_aspect_polarity_metrics(pw, rw)
    return {
        "ate_exact_f1": ate_e["f1"],
        "ate_partial_f1": ate_p["f1"],
        "pair_f1": pair["pair_f1"],
        "sentiment_accuracy": pair["sentiment_accuracy"],
    }


def paired_bootstrap(a, b, n_boot=10000, seed=42):
    """a, b: dicts with 'preds' and 'refs' (aligned, same gold). Returns per-metric stats."""
    assert len(a["preds"]) == len(b["preds"]), "systems must have equal #examples"
    n = len(a["preds"])
    keys = ["ate_exact_f1", "ate_partial_f1", "pair_f1", "sentiment_accuracy"]
    obs_a = metrics_on_indices(a["preds"], a["refs"], list(range(n)))
    obs_b = metrics_on_indices(b["preds"], b["refs"], list(range(n)))
    obs_delta = {k: round(obs_a[k] - obs_b[k], 4) for k in keys}

    rng = random.Random(seed)
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ma = metrics_on_indices(a["preds"], a["refs"], idx)
        mb = metrics_on_indices(b["preds"], b["refs"], idx)
        for k in keys:
            boot[k].append(ma[k] - mb[k])

    out = {}
    for k in keys:
        deltas = sorted(boot[k])
        lo = deltas[int(0.025 * n_boot)]
        hi = deltas[int(0.975 * n_boot) - 1]
        # two-sided p: fraction of bootstrap deltas on the opposite side of 0 from obs, x2
        n_le0 = sum(1 for d in deltas if d <= 0)
        n_ge0 = sum(1 for d in deltas if d >= 0)
        p = min(1.0, 2.0 * min(n_le0, n_ge0) / n_boot)
        out[k] = {
            "obs_delta_A_minus_B": obs_delta[k],
            "ci95": [round(lo, 4), round(hi, 4)],
            "p_value": round(p, 4),
            "significant_at_0.05": bool(p < 0.05),
        }
    return {"A": obs_a, "B": obs_b, "n_examples": n, "n_boot": n_boot, "per_metric": out}


def _selftest():
    # Build two aligned systems: A slightly better on pair, both same gold.
    rng = random.Random(0)
    preds_a, preds_b, refs = [], [], []
    for _ in range(120):
        gold = [{"term": f"t{rng.randrange(5)}", "polarity": rng.choice(["positive", "negative"])}]
        refs.append(gold)
        # A copies gold ~85% of the time; B ~80%
        preds_a.append(gold if rng.random() < 0.85 else [])
        preds_b.append(gold if rng.random() < 0.80 else [])
    res = paired_bootstrap({"preds": preds_a, "refs": refs},
                           {"preds": preds_b, "refs": refs}, n_boot=2000)
    print(json.dumps(res["per_metric"]["pair_f1"], indent=2))
    # identical systems -> delta 0, CI contains 0, p high
    res2 = paired_bootstrap({"preds": preds_a, "refs": refs},
                            {"preds": preds_a, "refs": refs}, n_boot=2000)
    pm = res2["per_metric"]["pair_f1"]
    assert pm["obs_delta_A_minus_B"] == 0.0, pm
    assert pm["ci95"][0] <= 0 <= pm["ci95"][1], pm
    print("identical-system control:", pm)
    print("\nSELFTEST PASSED — paired bootstrap behaves correctly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--a-name", default="A"); ap.add_argument("--b-name", default="B")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="paper_materials/revision_v2/significance/result.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return

    ra, rb = load_jsonl(args.a), load_jsonl(args.b)
    a = {"preds": [x["pred_aspects"] for x in ra], "refs": [x["gold_aspects"] for x in ra]}
    b = {"preds": [x["pred_aspects"] for x in rb], "refs": [x["gold_aspects"] for x in rb]}
    res = paired_bootstrap(a, b, args.n_boot, args.seed)
    res["A_name"], res["B_name"] = args.a_name, args.b_name
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
