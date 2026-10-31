#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Metric-Robustness and Annotation-Validity Analyses
# =============================================================================
"""
Offline analyses over the saved per-example predictions (no model inference),
written to answer reviewer concerns about the evaluation protocol:

  1. Matching variants for aspect-term extraction: the canonical set-based
     scores of src/evaluation.py, plus stricter or morphology-aware variants
     that keep repeated terms (multisets) and enforce one-to-one alignment via
     maximum bipartite matching: exact, substring ("partial"), stem-level
     (light Uzbek suffix stripping) and a soft character-level ROUGE-L score.
  2. Format vs. extraction: LLM scores restricted to reviews whose output
     parsed as JSON, so parse failures cannot masquerade as extraction errors.
  3. Paired bootstrap tests for the headline contrasts under the new variants.
  4. LLM-judge vs. native-speaker agreement at the level of quality *tiers*.
  5. Per-domain judge scores with bootstrap CIs (small domains pooled).

Usage:
    python scripts/robustness_analysis.py \
        --out paper_materials/revision_v2/robustness/robustness.json

Author: UzABSA Team
License: MIT
"""

import argparse
import csv
import importlib.util as ilu
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).parent.parent
_ev = ilu.spec_from_file_location("uzabsa_evaluation", ROOT / "src" / "evaluation.py")
_evaluation = ilu.module_from_spec(_ev); _ev.loader.exec_module(_evaluation)
compute_ate_metrics = _evaluation.compute_ate_metrics
compute_aspect_polarity_metrics = _evaluation.compute_aspect_polarity_metrics

R2 = ROOT / "paper_materials" / "revision_v2"
SYSTEMS = {
    "Qwen 2.5-7B": R2 / "results_v2" / "qwen2.5-7b_preds.jsonl",
    "Llama 3.1-8B": R2 / "results_v2" / "llama3.1-8b_preds.jsonl",
    "DeepSeek-R1-7B": R2 / "results_v2" / "deepseek-7b_preds.jsonl",
    "BERTbek": R2 / "baselines_v2" / "bertbek" / "preds_per_example.jsonl",
    "TahrirchiBERT": R2 / "baselines_v2" / "tahrirchi-bert" / "preds_per_example.jsonl",
    "Qwen 2.5-7B (seed 43)": R2 / "results_v2" / "qwen2.5-7b_seed43_preds.jsonl",
    "Qwen 2.5-7B (ZS)": R2 / "results_v2" / "qwen2.5-7b_zeroshot_preds.jsonl",
    "Llama 3.1-8B (ZS)": R2 / "results_v2" / "llama3.1-8b_zeroshot_preds.jsonl",
    "DeepSeek-R1-7B (ZS)": R2 / "results_v2" / "deepseek-7b_zeroshot_preds.jsonl",
}
FT_LLMS = ["Qwen 2.5-7B", "Llama 3.1-8B", "DeepSeek-R1-7B"]
ZS_LLMS = ["Qwen 2.5-7B (ZS)", "Llama 3.1-8B (ZS)", "DeepSeek-R1-7B (ZS)"]
CONTRASTS = [("Qwen 2.5-7B", "Llama 3.1-8B"), ("Qwen 2.5-7B", "TahrirchiBERT"),
             ("Qwen 2.5-7B", "BERTbek"), ("Llama 3.1-8B", "DeepSeek-R1-7B"),
             ("Llama 3.1-8B", "TahrirchiBERT")]


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ----------------------------------------------------------------------------
# Normalisation and light Uzbek stemming
# ----------------------------------------------------------------------------
_APOSTROPHES = str.maketrans({c: "'" for c in "‘’ʻʼ`´"})
_PUNCT = re.compile(r"[^\w' -]+")

# Nominal inflection is stripped right-to-left in morphotactic order
# stem + PLURAL + POSSESSIVE + CASE; derivational suffixes are left intact.
CASE_SUFFIXES = ["ning", "dagi", "dan", "ni", "ga", "ka", "qa", "da"]
POSS_SUFFIXES = ["imiz", "ingiz", "si", "i"]
PLURAL_SUFFIXES = ["lar"]
COLLOQUIAL_PLURAL = "la"     # r-dropped plural before a case suffix: ofitsantlani
MIN_STEM = 3


def norm(term):
    """Canonical normalisation of src/evaluation.py (lower-case, strip)."""
    return term.lower().strip()


def _strip_one(word, suffixes):
    for s in suffixes:
        if word.endswith(s) and len(word) - len(s) >= MIN_STEM:
            return word[: -len(s)], True
    return word, False


def stem_token(tok):
    w, had_case = _strip_one(tok, CASE_SUFFIXES)
    w, _ = _strip_one(w, POSS_SUFFIXES)
    w, had_pl = _strip_one(w, PLURAL_SUFFIXES)
    if had_case and not had_pl and w.endswith(COLLOQUIAL_PLURAL) and len(w) - 2 >= MIN_STEM:
        w = w[:-2]
    # stem-final consonant alternation before vowel suffixes (yurak/yuragi, o'rtoq/o'rtog'i)
    if w.endswith("g'"):
        w = w[:-2] + "q"
    elif w.endswith("g"):
        w = w[:-1] + "k"
    return w


def clean(term):
    """Orthographic normalisation only: apostrophe variants, punctuation, spacing."""
    return " ".join(_PUNCT.sub(" ", term.lower().translate(_APOSTROPHES)).split())


def stem(term):
    return " ".join(stem_token(tok) for tok in clean(term).split())


def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def char_rouge_l(a, b):
    """Character-level ROUGE-L F1 (beta = 1): 2 LCS / (|a| + |b|)."""
    return 2 * lcs_len(a, b) / (len(a) + len(b)) if a and b else 0.0


# ----------------------------------------------------------------------------
# One-to-one (bipartite) matching over multisets
# ----------------------------------------------------------------------------
def matched_weight(preds, golds, weight):
    """Maximum-weight one-to-one alignment; returns the summed weight."""
    if not preds or not golds:
        return 0.0
    w = np.array([[weight(p, g) for g in golds] for p in preds], dtype=float)
    r, c = linear_sum_assignment(-w)
    return float(w[r, c].sum())


MATCHERS = {
    "exact_1to1": lambda p, g: float(norm(p) == norm(g)),
    "partial_1to1": lambda p, g: float(norm(p) in norm(g) or norm(g) in norm(p)),
    "clean_1to1": lambda p, g: float(clean(p) == clean(g)),
    "stem_1to1": lambda p, g: float(stem(p) == stem(g)),
    "soft_rougeL": lambda p, g: char_rouge_l(norm(p), norm(g)),
}


def terms(aspects):
    return [a["term"] for a in aspects if a.get("term")]


def per_example_counts(rows, kind):
    """Per-review (tp, n_pred, n_gold) arrays for the given matching variant."""
    tp, npred, ngold = [], [], []
    for r in rows:
        p_asp = [a for a in r["pred_aspects"] if a.get("term")]
        g_asp = [a for a in r["gold_aspects"] if a.get("term")]
        if kind.startswith("pair_"):
            base = MATCHERS[kind[len("pair_"):]]
            same_pol = lambda a, b: norm(a.get("polarity", "")) == norm(b.get("polarity", ""))
            tp.append(matched_weight(p_asp, g_asp, lambda a, b: base(a["term"], b["term"]) * same_pol(a, b)))
        else:
            tp.append(matched_weight(terms(p_asp), terms(g_asp), MATCHERS[kind]))
        npred.append(len(p_asp))
        ngold.append(len(g_asp))
    return np.array(tp), np.array(npred), np.array(ngold)


def prf(tp, npred, ngold):
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def canonical(rows, idx=None):
    rows = rows if idx is None else [rows[i] for i in idx]
    pt = [terms(r["pred_aspects"]) for r in rows]
    gt = [terms(r["gold_aspects"]) for r in rows]
    pw = [[a for a in r["pred_aspects"] if a.get("term")] for r in rows]
    gw = [[a for a in r["gold_aspects"] if a.get("term")] for r in rows]
    ex = compute_ate_metrics(pt, gt)
    pa = compute_ate_metrics(pt, gt, partial_match=True)
    pair = compute_aspect_polarity_metrics(pw, gw)
    return {"ate_exact_p": ex["precision"], "ate_exact_r": ex["recall"], "ate_exact_f1": ex["f1"],
            "ate_partial_f1": pa["f1"], "pair_f1": pair["pair_f1"],
            "sentiment_accuracy": pair["sentiment_accuracy"],
            "sentiment_macro_f1": pair["sentiment_macro_f1"],
            # unique (term, polarity) predictions per gold term, as in the paper tables
            "pred_gold_ratio": round(pair["total_predictions"] / pair["total_references"], 2)}


def paired_bootstrap(a, b, n_boot=10000, seed=42):
    """a, b: (tp, npred, ngold) per-example arrays on the same reviews."""
    rng = np.random.default_rng(seed)
    n = len(a[0])
    idx = rng.integers(0, n, size=(n_boot, n))

    def f1(c):
        tp, npred, ngold = (x[idx].sum(axis=1) for x in c)
        p = np.divide(tp, npred, out=np.zeros(n_boot), where=npred > 0)
        r = np.divide(tp, ngold, out=np.zeros(n_boot), where=ngold > 0)
        return np.divide(2 * p * r, p + r, out=np.zeros(n_boot), where=(p + r) > 0)

    d = f1(a) - f1(b)
    obs = prf(*(x.sum() for x in a))[2] - prf(*(x.sum() for x in b))[2]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": round(obs, 4), "ci95": [round(float(np.percentile(d, 2.5)), 4),
            round(float(np.percentile(d, 97.5)), 4)], "p_value": round(min(1.0, p), 4)}


# ----------------------------------------------------------------------------
# Annotation-validity analyses
# ----------------------------------------------------------------------------
def tier(score):
    return "include" if score >= 3.5 else ("flag" if score >= 2.5 else "exclude")


def cohen_kappa(a, b, labels, weights=None):
    k = len(labels)
    pos = {l: i for i, l in enumerate(labels)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[pos[x], pos[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    i, j = np.indices((k, k))
    w = {None: (i != j).astype(float), "linear": np.abs(i - j) / (k - 1),
         "quadratic": ((i - j) / (k - 1)) ** 2}[weights]
    return 1 - (w * obs).sum() / (w * exp).sum()


def judge_vs_human(hv_dir):
    def read(p):
        return {r["review_id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
    sample = read(hv_dir / "sample_150.csv")
    raters = [read(p) for p in sorted((hv_dir / "returned").glob("rubric_*.csv"))]
    ids = [i for i in sample if all(i in r for r in raters)]
    judge = [float(sample[i]["judge_overall"]) for i in ids]
    human = [np.mean([float(r[i]["overall_1_5"]) for r in raters]) for i in ids]
    jt, ht = [tier(s) for s in judge], [tier(s) for s in human]
    labels = ["exclude", "flag", "include"]
    order = labels[::-1]
    inc_j = [h for j, h in zip(jt, ht) if j == "include"]
    return {
        "n": len(ids), "judge_mean": round(float(np.mean(judge)), 2),
        "human_mean": round(float(np.mean(human)), 2),
        "tier_agreement": round(float(np.mean([a == b for a, b in zip(jt, ht)])), 3),
        "kappa": round(cohen_kappa(jt, ht, labels), 3),
        "kappa_linear": round(cohen_kappa(jt, ht, labels, "linear"), 3),
        "judge_tiers": dict(Counter(jt)), "human_tiers": dict(Counter(ht)),
        "include_precision": round(inc_j.count("include") / len(inc_j), 3),
        "include_judged_human_exclude": inc_j.count("exclude"),
        "include_recall": round(sum(j == h == "include" for j, h in zip(jt, ht)) / ht.count("include"), 3),
        # rows: judge tier, columns: human tier, both include -> flag -> exclude
        "tier_confusion": [[sum(j == a and h == b for j, h in zip(jt, ht)) for b in order] for a in order],
    }


def domain_cis(judge_path, min_n=15, n_boot=10000, seed=42):
    rows = json.load(open(judge_path, encoding="utf-8"))
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["business_category"]].append(r["judge_scores"]["overall"])
    rng = np.random.default_rng(seed)

    def ci(xs):
        xs = np.array(xs, dtype=float)
        boots = xs[rng.integers(0, len(xs), size=(n_boot, len(xs)))]
        return {"n": len(xs), "mean": round(float(xs.mean()), 2),
                "ci95": [round(float(np.percentile(boots.mean(1), q)), 2) for q in (2.5, 97.5)],
                "include_rate": round(float((xs >= 3.5).mean()), 3),
                "tiers": {t: sum(tier(x) == t for x in xs) for t in ("include", "flag", "exclude")}}

    out = {d: ci(v) for d, v in sorted(by_dom.items(), key=lambda kv: -len(kv[1])) if len(v) >= min_n}
    small = [s for v in by_dom.values() if len(v) < min_n for s in v]
    out[f"pooled_small_domains (N<{min_n}, {sum(len(v) < min_n for v in by_dom.values())} domains)"] = ci(small)
    out["all"] = ci([s for v in by_dom.values() for s in v])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(R2 / "robustness" / "robustness.json"))
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    rows = {name: load_jsonl(p) for name, p in SYSTEMS.items()}
    gold0 = [r["gold_aspects"] for r in rows["Qwen 2.5-7B"]]
    assert all([r["gold_aspects"] for r in v] == gold0 for v in rows.values()), "gold must be shared"

    gold_terms = [terms(g) for g in gold0]
    collapsed = sum(len(t) - len(set(map(norm, t))) for t in gold_terms)
    report = {"gold": {"reviews": len(gold0), "terms": sum(map(len, gold_terms)),
                       "terms_lost_to_set_collapsing": collapsed,
                       "reviews_with_repeated_term": sum(len(t) != len(set(map(norm, t))) for t in gold_terms)}}

    kinds = ["exact_1to1", "partial_1to1", "clean_1to1", "stem_1to1", "soft_rougeL",
             "pair_exact_1to1", "pair_stem_1to1"]
    counts = {name: {k: per_example_counts(v, k) for k in kinds} for name, v in rows.items()}
    report["systems"] = {}
    for name, v in rows.items():
        entry = {"canonical": canonical(v)}
        for k in kinds:
            p, r, f = prf(*(x.sum() for x in counts[name][k]))
            entry[k] = {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)}
        report["systems"][name] = entry

    report["bootstrap"] = {
        f"{a} vs {b}": {k: paired_bootstrap(counts[a][k], counts[b][k], args.n_boot)
                        for k in ["exact_1to1", "stem_1to1", "pair_exact_1to1", "pair_stem_1to1"]}
        for a, b in CONTRASTS}

    # Stem-level matches that exact matching misses, and substring matches the
    # stemmer rejects: examples for a qualitative check of both criteria.
    gained, rejected = Counter(), Counter()
    for r in rows["Qwen 2.5-7B"]:
        for p in terms(r["pred_aspects"]):
            for g in terms(r["gold_aspects"]):
                if clean(p) != clean(g) and stem(p) == stem(g):
                    gained[(p, g)] += 1
                elif (norm(p) in norm(g) or norm(g) in norm(p)) and norm(p) != norm(g) and stem(p) != stem(g):
                    rejected[(p, g)] += 1
    report["examples"] = {"stem_not_exact": [list(k) for k, _ in gained.most_common(25)],
                          "substring_not_stem": [list(k) for k, _ in rejected.most_common(25)]}

    def parsed_subset(names):
        ok = [all(rows[n][i].get("parse_success", True) for n in names) for i in range(len(gold0))]
        return [i for i, f in enumerate(ok) if f]

    report["format_vs_extraction"] = {}
    for group in (FT_LLMS, ZS_LLMS):
        both = parsed_subset(group)
        for n in group:
            own = parsed_subset([n])
            report["format_vs_extraction"][n] = {
                "parse_failures": len(gold0) - len(own),
                "gold_terms_in_failures": sum(len(gold_terms[i]) for i in range(len(gold0)) if i not in set(own)),
                "own_parsed": canonical(rows[n], own),
                "common_parsed": {"n_reviews": len(both), **canonical(rows[n], both)},
            }

    report["judge_vs_human"] = judge_vs_human(R2 / "human_validation")
    report["domain_cis"] = domain_cis(ROOT / "data" / "judged" / "judge_results.json")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
