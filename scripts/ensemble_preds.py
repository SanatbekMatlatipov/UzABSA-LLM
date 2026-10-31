#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Majority-vote ensembles over saved per-example predictions
# =============================================================================
"""
Combines several systems' per-example predictions on the same reviews by
majority vote. Nothing is trained and nothing is tuned:

  * each system casts at most one vote per aspect term in a review, after the
    canonical normalisation of src/evaluation.py (lower-case, strip);
  * a term is kept when at least --min-votes systems predict it;
  * its polarity is the plurality of the polarities the voting systems gave
    it, ties going to the system listed first.

Writes the ensemble's predictions in the JSONL format significance_test.py
consumes, plus a metrics file with the canonical scores, the one-to-one
matching variants of robustness_analysis.py, paired bootstrap tests of the
ensemble against every member (Holm-adjusted over the file), and how many
gold pairs each member finds that no other member does.

Usage:
    python scripts/ensemble_preds.py --name "Majority of 5" --min-votes 3 \
        --systems Qwen=paper_materials/revision_v2/results_v2/qwen2.5-7b_preds.jsonl \
                  Llama=paper_materials/revision_v2/results_v2/llama3.1-8b_preds.jsonl \
        --out-dir experiments/v3/results/ensembles

Author: UzABSA Team
License: MIT
"""

import argparse
import importlib.util as ilu
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ra = ilu.spec_from_file_location(
    "uzabsa_robustness", Path(__file__).parent / "robustness_analysis.py")
_robust = ilu.module_from_spec(_ra); _ra.loader.exec_module(_robust)
load_jsonl, norm, canonical = _robust.load_jsonl, _robust.norm, _robust.canonical
per_example_counts, prf, paired_bootstrap = (
    _robust.per_example_counts, _robust.prf, _robust.paired_bootstrap)

VARIANTS = ["exact_1to1", "partial_1to1", "stem_1to1", "pair_exact_1to1", "pair_stem_1to1"]


def vote(rows_by_system, min_votes):
    """rows_by_system: list of aligned row lists, in tie-break order."""
    out = []
    for i in range(len(rows_by_system[0])):
        votes, pols, surface = Counter(), defaultdict(Counter), {}
        first_pol = defaultdict(list)          # polarities in system order, for ties
        for rows in rows_by_system:
            mine = defaultdict(list)
            for a in rows[i]["pred_aspects"]:
                if a.get("term"):
                    mine[norm(a["term"])].append(norm(a.get("polarity", "")))
                    surface.setdefault(norm(a["term"]), a["term"])
            for t, ps in mine.items():
                votes[t] += 1
                for p in dict.fromkeys(ps):
                    pols[t][p] += 1
                    first_pol[t].append(p)
        kept = []
        for t, v in votes.items():
            if v >= min_votes:
                top = max(pols[t].values())
                pol = next(p for p in first_pol[t] if pols[t][p] == top)
                kept.append({"term": surface[t], "polarity": pol, "votes": v})
        out.append(kept)
    return out


def canonical_counts(rows):
    """Per-review counts behind the canonical set-based scores, for the bootstrap.

    Accuracy is passed as an (correct, total, total) triple, whose F1 equals it.
    """
    ex, pair, sent = ([], [], []), ([], [], []), ([], [], [])
    for r in rows:
        pp = {(norm(a["term"]), norm(a.get("polarity", ""))) for a in r["pred_aspects"] if a.get("term")}
        gp = {(norm(a["term"]), norm(a.get("polarity", ""))) for a in r["gold_aspects"] if a.get("term")}
        pt, gt = {t for t, _ in pp}, {t for t, _ in gp}
        for acc, vals in ((ex, (len(pt & gt), len(pt), len(gt))),
                          (pair, (len(pp & gp), len(pp), len(gp)))):
            for lst, v in zip(acc, vals):
                lst.append(v)
        # same rule as compute_aspect_polarity_metrics: first polarity seen per term
        both = pt & gt
        ok = sum(next(p for t, p in pp if t == term) == next(p for t, p in gp if t == term)
                 for term in both)
        for lst, v in zip(sent, (ok, len(both), len(both))):
            lst.append(v)
    return {k: tuple(np.array(x) for x in c)
            for k, c in (("ate_exact_f1", ex), ("pair_f1", pair), ("sentiment_accuracy", sent))}


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = round(running, 4)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--systems", nargs="+", required=True, metavar="NAME=PATH",
                    help="member predictions; the first listed wins polarity ties")
    ap.add_argument("--min-votes", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="experiments/v3/results/ensembles")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    members = dict(s.split("=", 1) for s in args.systems)
    rows = {name: load_jsonl(path) for name, path in members.items()}
    ref = next(iter(rows.values()))
    for name, r in rows.items():
        assert [x["text"] for x in r] == [x["text"] for x in ref], f"{name}: reviews differ"
        assert [x["gold_aspects"] for x in r] == [x["gold_aspects"] for x in ref], f"{name}: gold differs"

    preds = vote(list(rows.values()), args.min_votes)
    ens = [{"idx": i, "text": x["text"], "pred_aspects": p, "gold_aspects": x["gold_aspects"],
            "parse_success": True} for i, (x, p) in enumerate(zip(ref, preds))]
    slug = re.sub(r"[^a-z0-9]+", "_", args.name.lower()).strip("_")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{slug}_preds.jsonl", "w", encoding="utf-8") as f:
        for e in ens:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    every = {args.name: ens, **rows}
    scores = {}
    for name, r in every.items():
        entry = {"canonical": canonical(r)}
        for k in VARIANTS:
            p, rc, f1 = prf(*(x.sum() for x in per_example_counts(r, k)))
            entry[k] = {"p": round(p, 4), "r": round(rc, 4), "f1": round(f1, 4)}
        scores[name] = entry

    # bootstrap: ensemble vs each member; canonical exact/pair/accuracy + 1-to-1 partial
    def counts(r):
        c = canonical_counts(r)
        c["ate_partial_1to1_f1"] = per_example_counts(r, "partial_1to1")
        c["pair_stem_1to1_f1"] = per_example_counts(r, "pair_stem_1to1")
        return c
    ce = counts(ens)
    tests = {}
    for name, r in rows.items():
        cm = counts(r)
        tests[name] = {k: paired_bootstrap(ce[k], cm[k], args.n_boot) for k in ce}
    flat = [(m, k) for m in tests for k in tests[m]]
    for (m, k), a in zip(flat, holm([tests[m][k]["p_value"] for m, k in flat])):
        tests[m][k]["p_holm"] = a

    # gold pairs each member finds and no other member does
    def hits(r, i):
        gp = {(norm(a["term"]), norm(a.get("polarity", ""))) for a in r[i]["gold_aspects"] if a.get("term")}
        return gp & {(norm(a["term"]), norm(a.get("polarity", ""))) for a in r[i]["pred_aspects"] if a.get("term")}
    unique, union, n_gold = Counter(), 0, 0
    for i in range(len(ref)):
        found = {name: hits(r, i) for name, r in rows.items()}
        all_found = set().union(*found.values())
        union += len(all_found)
        n_gold += len({(norm(a["term"]), norm(a.get("polarity", "")))
                       for a in ref[i]["gold_aspects"] if a.get("term")})
        for name, h in found.items():
            others = set().union(*(v for k, v in found.items() if k != name))
            unique[name] += len(h - others)

    report = {"name": args.name, "rule": f"term kept if >= {args.min_votes} of {len(rows)} "
              f"systems predict it; polarity by plurality, ties to the first listed system",
              "members": members, "n_reviews": len(ref), "n_boot": args.n_boot,
              "scores": scores, "ensemble_vs_member": tests,
              "complementarity": {"gold_pairs": n_gold, "found_by_any_member": union,
                                  "oracle_pair_recall": round(union / n_gold, 4),
                                  "found_only_by": dict(unique)}}
    (out_dir / f"{slug}_metrics.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    keys = ["ate_exact_f1", "ate_partial_1to1_f1", "pair_f1", "sentiment_accuracy"]
    print(f"{'system':24s} exact  part1:1  pair   acc")
    for name, s in scores.items():
        c = s["canonical"]
        print(f"{name:24s} {c['ate_exact_f1']:.4f} {s['partial_1to1']['f1']:.4f} "
              f"{c['pair_f1']:.4f} {c['sentiment_accuracy']:.4f}")
    for m, t in tests.items():
        print(f"vs {m:21s} " + "  ".join(
            f"{k.split('_')[0] if k != 'ate_partial_1to1_f1' else 'part'}"
            f"{t[k]['delta']:+.3f} (p={t[k]['p_value']}, holm={t[k]['p_holm']})" for k in keys))
    print(f"Saved -> {out_dir / (slug + '_metrics.json')}")


if __name__ == "__main__":
    main()
