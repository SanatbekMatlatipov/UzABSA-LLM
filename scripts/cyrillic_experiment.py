#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: does transliterating Uzbek-Cyrillic reviews help the annotator?
# =============================================================================
"""
Corpus reviews written in Uzbek Cyrillic, a script the training data does
not contain, reach the Include tier far less often than Latin ones. This
script annotates each of them twice with the same fine-tuned model: once as
written ("cyrillic" arm), and once in Latin transliteration with the terms
mapped back to the Cyrillic original ("latin" arm, src/transliterate.py).
Both arms are written in the release format, so judge_panel.py scores them
exactly like the corpus, and `report` compares the arms review by review.

Usage:
    python scripts/cyrillic_experiment.py annotate --out experiments/v3/translit
    for arm in cyrillic latin; do
      python scripts/judge_panel.py score --rubric v1 \
          --annotations experiments/v3/translit/annotations_$arm.json \
          --out experiments/v3/translit/judge_$arm
    done
    python scripts/cyrillic_experiment.py report --out experiments/v3/translit

Author: UzABSA Team
License: MIT
"""

import argparse
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.transliterate import detect_script  # noqa: E402

ARMS = ["cyrillic", "latin"]
DIMS = ["completeness", "accuracy", "sentiment", "relevance", "overall"]
KEEP = ("review_id", "text", "business_name", "business_category", "user_rating")


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def select(corpus):
    return [a for a in json.load(open(corpus, encoding="utf-8")) if detect_script(a["text"]) == "uz_cyrl"]


def cmd_annotate(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    from src.inference import extract_aspects

    items = select(args.corpus)
    if args.limit:
        items = items[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / "raw.jsonl"
    done = {(r["review_id"], r["arm"]) for r in load_jsonl(raw_path)}
    todo = [(it, arm) for it in items for arm in ARMS if (it["review_id"], arm) not in done]
    print(f"{len(items)} Uzbek-Cyrillic reviews; {len(todo)} of {2 * len(items)} annotations to do")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    meta = {"model": args.model_path, "revision": args.revision, "device": device, "dtype": "bfloat16",
            "decoding": "greedy, max_new_tokens 512 (src.inference.extract_aspects)",
            "torch": torch.__version__, "transformers": transformers.__version__,
            "host": platform.node(), "corpus": str(args.corpus), "n_reviews": len(items),
            "started": datetime.now().isoformat(timespec="seconds")}
    if todo:
        tok = AutoTokenizer.from_pretrained(args.model_path, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, revision=args.revision,
                                                     dtype=torch.bfloat16).to(device).eval()
        with open(raw_path, "a", encoding="utf-8") as f:
            for it, arm in tqdm(todo, desc="Annotating"):
                t0 = time.time()
                res = extract_aspects(model, tok, it["text"], transliterate=(arm == "latin"))
                f.write(json.dumps({"review_id": it["review_id"], "arm": arm,
                                    "aspects": [a for a in res.get("aspects") or [] if isinstance(a, dict)],
                                    "parse_success": bool(res.get("parse_success")),
                                    "raw_output": res.get("raw_output", ""),
                                    "seconds": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")
                f.flush()

    raw = {(r["review_id"], r["arm"]): r for r in load_jsonl(raw_path)}
    for arm in ARMS:
        rows = []
        for it in items:
            r = raw.get((it["review_id"], arm))
            if r is None:
                continue
            rows.append({k: it.get(k) for k in KEEP}
                        | {"aspects": r["aspects"], "num_aspects": len(r["aspects"]),
                           "annotation_source": f"{args.source_name} [{arm} input]",
                           "parse_success": r["parse_success"]})
        (args.out / f"annotations_{arm}.json").write_text(
            json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{arm:9s} {len(rows)} reviews -> {args.out / f'annotations_{arm}.json'}")
    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    (args.out / "annotate_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def tier(x):
    return "include" if x >= 3.5 else "flag" if x >= 2.5 else "exclude"


def arm_stats(scores, ids):
    t = Counter(tier(scores[i]["overall"]) for i in ids)
    return {"n": len(ids), "tier_pct": {k: round(100 * t[k] / len(ids), 1) for k in ("include", "flag", "exclude")},
            "means": {d: round(float(np.mean([scores[i][d] for i in ids])), 2) for d in DIMS}}


def cmd_report(args):
    from scipy.stats import binomtest, wilcoxon

    judged = {arm: {r["review_id"]: r["judge_scores"]
                    for r in json.load(open(args.out / f"judge_{arm}" / "judge_results.json", encoding="utf-8"))}
              for arm in ARMS}
    released = {r["review_id"]: r["judge_scores"]
                for r in json.load(open(args.release_judge, encoding="utf-8"))}
    ids = [i for i in judged["cyrillic"] if i in judged["latin"] and i in released]

    a = np.array([judged["cyrillic"][i]["overall"] for i in ids])
    b = np.array([judged["latin"][i]["overall"] for i in ids])
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(ids), size=(args.n_boot, len(ids)))
    boots = (b[idx] - a[idx]).mean(1)
    inc_a, inc_b = a >= 3.5, b >= 3.5
    up, down = int(np.sum(~inc_a & inc_b)), int(np.sum(inc_a & ~inc_b))

    raw = [r for r in load_jsonl(args.out / "raw.jsonl") if r["arm"] == "latin"]
    terms = [x for r in raw for x in r["aspects"] if x.get("term")]
    by_arm = {arm: [r for r in load_jsonl(args.out / "raw.jsonl") if r["arm"] == arm] for arm in ARMS}
    # context: the released corpus by script, from the same judge run
    scripts = {}
    for rec in json.load(open(args.corpus, encoding="utf-8")):
        scripts.setdefault(detect_script(rec["text"]), []).append(rec["review_id"])
    by_script = {s: {"n": len(v), "include_pct": round(100 * np.mean([released[i]["overall"] >= 3.5 for i in v]), 1)}
                 for s, v in scripts.items()}
    rest = [i for s, v in scripts.items() if s != "uz_cyrl" for i in v]
    by_script["all but uz_cyrl"] = {"n": len(rest),
                                    "include_pct": round(100 * np.mean([released[i]["overall"] >= 3.5 for i in rest]), 1)}

    report = {
        "n_reviews": len(ids),
        "released_corpus_by_script": by_script,
        "arms": {"released annotation (published checkpoint, Cyrillic input)": arm_stats(released, ids),
                 **{f"{arm} input": arm_stats(judged[arm], ids) for arm in ARMS}},
        "latin_minus_cyrillic": {
            "overall_mean_diff": round(float((b - a).mean()), 3),
            "overall_diff_ci95": [round(float(np.percentile(boots, q)), 3) for q in (2.5, 97.5)],
            "wilcoxon_p": float(wilcoxon(b, a).pvalue) if np.any(a != b) else 1.0,
            "include_gained": up, "include_lost": down,
            "mcnemar_exact_p": float(binomtest(up, up + down).pvalue) if up + down else 1.0},
        "annotation": {arm: {"parse_success": round(float(np.mean([r["parse_success"] for r in rs])), 3),
                             "aspects_per_review": round(float(np.mean([len(r["aspects"]) for r in rs])), 2),
                             "empty": sum(not r["aspects"] for r in rs),
                             "seconds_per_review": round(float(np.mean([r["seconds"] for r in rs])), 1)}
                       for arm, rs in by_arm.items() if rs},
        "latin_terms_mapped_back": {"terms": len(terms), "mapped": sum("term_latin" in x for x in terms)},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"{len(ids)} Uzbek-Cyrillic reviews, judge panel (v1 rubric)")
    print(f"{'arm':62s} include  flag  excl  " + "  ".join(d[:5] for d in DIMS))
    for name, s in report["arms"].items():
        print(f"{name:62s} {s['tier_pct']['include']:6.1f} {s['tier_pct']['flag']:5.1f} {s['tier_pct']['exclude']:5.1f}  "
              + "  ".join(f"{s['means'][d]:5.2f}" for d in DIMS))
    d = report["latin_minus_cyrillic"]
    print(f"latin - cyrillic overall: {d['overall_mean_diff']:+.3f} CI {d['overall_diff_ci95']} "
          f"Wilcoxon p={d['wilcoxon_p']:.2g}; Include +{up} / -{down}, McNemar p={d['mcnemar_exact_p']:.2g}")
    print("annotation:", report["annotation"], "mapped back:", report["latin_terms_mapped_back"])
    print(f"Saved -> {args.out / 'report.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["annotate", "report"])
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "v3" / "translit")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "annotated" / "reviews_annotated.json")
    ap.add_argument("--model-path", default="Sanatbek/UzABSA-LLM",
                    help="merged model directory or Hugging Face repo")
    ap.add_argument("--revision", default="8636bd10074b6aa693a56885fc15cc830bd38a69",
                    help="Hugging Face revision; the default is the qwen2.5-7b branch of 2026-08-29")
    ap.add_argument("--source-name", default="qwen2.5-7b v2 (HF 8636bd1)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--release-judge", type=Path,
                    default=ROOT / "experiments" / "v3" / "judge" / "release_v1" / "judge_results.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    {"annotate": cmd_annotate, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
