#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM v3: index of checkpoints, logs, predictions and judge results
# =============================================================================
"""
Builds one index of everything behind the paper's numbers, for the v2 systems
the paper reports and for every v3 run, so any number can be traced to its
checkpoint, training log and predictions:

  experiments/v3/registry.json  every system and run: checkpoint location,
                                logs, prediction files, metrics, seed means
                                and spreads, ensembles, significance tests
                                (Holm-adjusted), judge calibration and corpus
                                judging
  experiments/v3/MANIFEST.json  SHA-256 of every result and log file (size
                                only for weights), git commit, platform
  experiments/v3/SUMMARY.md     the same as readable tables, plus what is
                                still missing

Metrics are recomputed from the per-example predictions with the paper's
metric code (canonical exact, pair and sentiment scores; one-to-one partial
F1), never copied from a log. Each Hugging Face branch is checked against the
training run it should hold: a branch last updated before that run finished
still serves older weights.

Usage:
    PYTHONHASHSEED=0 python scripts/collect_v3.py [--offline]

Author: UzABSA Team
License: MIT
"""

import argparse
import hashlib
import importlib.util as ilu
import json
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).parent.parent
V3 = ROOT / "experiments" / "v3"
R2 = ROOT / "paper_materials" / "revision_v2"
HF_REPO = "Sanatbek/UzABSA-LLM"
HASH_LIMIT = 100 * 2 ** 20            # hash files up to 100 MB; record size for weights

_ra = ilu.spec_from_file_location("uzabsa_robustness", ROOT / "scripts" / "robustness_analysis.py")
_robust = ilu.module_from_spec(_ra); _ra.loader.exec_module(_robust)
_en = ilu.spec_from_file_location("uzabsa_ensemble", ROOT / "scripts" / "ensemble_preds.py")
_ensemble = ilu.module_from_spec(_en); _en.loader.exec_module(_ensemble)

CFG = json.load(open(V3 / "config.json", encoding="utf-8"))
METRICS = ["ate_exact_f1", "ate_partial_1to1_f1", "pair_f1", "sentiment_accuracy", "sentiment_macro_f1"]

# The systems of the paper's Table 2. v2 encoder weights were not kept; the
# Hugging Face branches are listed as configured in push_to_hub.py.
V2_SYSTEMS = {
    "Qwen 2.5-7B": {"preds": R2 / "results_v2/qwen2.5-7b_preds.jsonl",
                    "logs": ROOT / "outputs/rerun_v2/uzabsa_qwen2.5-7b_v2",
                    "checkpoint": f"hf://{HF_REPO}@qwen2.5-7b"},
    "Qwen 2.5-7B (seed 43)": {"preds": R2 / "results_v2/qwen2.5-7b_seed43_preds.jsonl",
                              "logs": ROOT / "outputs/rerun_v2/uzabsa_qwen2.5-7b_v2_seed43",
                              "checkpoint": None},
    "Llama 3.1-8B": {"preds": R2 / "results_v2/llama3.1-8b_preds.jsonl",
                     "logs": ROOT / "outputs/rerun_v2/uzabsa_llama3.1-8b_v2",
                     "checkpoint": f"hf://{HF_REPO}@llama3.1-8b"},
    "DeepSeek-R1-7B": {"preds": R2 / "results_v2/deepseek-7b_preds.jsonl",
                       "logs": ROOT / "outputs/rerun_v2/uzabsa_deepseek-7b_v2",
                       "checkpoint": f"hf://{HF_REPO}@deepseek-r1-7b"},
    "BERTbek": {"preds": R2 / "baselines_v2/bertbek/preds_per_example.jsonl",
                "logs": R2 / "baselines_v2/bertbek", "checkpoint": None},
    "TahrirchiBERT": {"preds": R2 / "baselines_v2/tahrirchi-bert/preds_per_example.jsonl",
                      "logs": R2 / "baselines_v2/tahrirchi-bert", "checkpoint": None},
}


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT.resolve()))


def load(p):
    return json.load(open(p, encoding="utf-8"))


def metrics(pred_file):
    rows = _robust.load_jsonl(pred_file)
    c = _robust.canonical(rows)
    _, _, partial = _robust.prf(*(x.sum() for x in _robust.per_example_counts(rows, "partial_1to1")))
    parsed = [r.get("parse_success", True) for r in rows]
    return {"n": len(rows), "ate_exact_p": c["ate_exact_p"], "ate_exact_r": c["ate_exact_r"],
            "ate_exact_f1": c["ate_exact_f1"], "ate_partial_1to1_f1": round(partial, 4),
            "pair_f1": c["pair_f1"], "sentiment_accuracy": c["sentiment_accuracy"],
            "sentiment_macro_f1": c["sentiment_macro_f1"],
            "parse_rate": round(sum(parsed) / len(parsed), 4)}


def files(d, patterns=("*.json", "*.csv", "*.txt", "*.png", "*.md", "*.log")):
    d = Path(d)
    return sorted(rel(p) for pat in patterns for p in d.glob(pat)) if d.exists() else []


def best_checkpoint(run_dir):
    """best_model_checkpoint recorded by the HF Trainer in the newest checkpoint."""
    states = sorted(Path(run_dir).glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[1]))
    if not states:
        return None
    s = load(states[-1])
    return {"best_model_checkpoint": s.get("best_model_checkpoint"),
            "best_dev_loss": s.get("best_metric"), "global_step": s.get("global_step")}


def hub_status(checkpoint, logs_dir, offline):
    """Compare a Hub branch's last update with the end of the run it should hold."""
    if not checkpoint or offline:
        return None
    branch = checkpoint.split("@", 1)[1]
    try:
        resp = httpx.get(f"https://huggingface.co/api/models/{HF_REPO}/revision/{branch}", timeout=15)
        updated = datetime.fromisoformat(resp.json()["lastModified"].replace("Z", "+00:00"))
    except (httpx.HTTPError, KeyError, ValueError):
        return {"branch": branch, "status": "unknown (Hub not reachable)"}
    summary = Path(logs_dir) / "experiment_summary.json"
    if not summary.exists():
        return {"branch": branch, "last_updated": updated.isoformat(), "status": "unknown (no run log)"}
    # run timestamps are local time without a zone; a day of slack covers any zone
    trained = datetime.fromisoformat(load(summary)["experiment"]["timestamp"]).replace(tzinfo=timezone.utc)
    stale = updated < trained - timedelta(days=1)
    return {"branch": branch, "last_updated": updated.isoformat(), "run_started": trained.isoformat()[:19],
            "status": "older than the run: the Hub still serves earlier weights" if stale
            else "updated after the run"}


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------
def v2_reference(offline):
    out = {}
    for name, s in V2_SYSTEMS.items():
        if not s["preds"].exists():
            continue
        out[name] = {"checkpoint": s["checkpoint"], "hub": hub_status(s["checkpoint"], s["logs"], offline),
                     "logs": files(s["logs"]), "test_preds": rel(s["preds"]), "test": metrics(s["preds"])}
    return out


def v3_runs():
    runs = {}
    names = CFG.get("names", {})
    for d in sorted((V3 / "runs").glob("*_s*")):
        system, seed = d.name.rsplit("_s", 1)
        family = "encoder" if (d / "train_summary.json").exists() or system in CFG["encoders"]["models"] else "llm"
        entry = {"system": names.get(system, system), "seed": int(seed), "family": family, "dir": rel(d)}
        if family == "llm":
            merged, adapters = d / "merged_model", d / "lora_adapters"
            entry["checkpoint"] = {"merged_model": rel(merged) if merged.exists() else None,
                                   "lora_adapters": rel(adapters) if adapters.exists() else None,
                                   "selection": best_checkpoint(d)}
            if (d / "experiment_summary.json").exists():
                summ = load(d / "experiment_summary.json")
                entry["training"] = {k: summ.get(k) for k in
                                     ("model", "lora_config", "training_config", "system", "training_results")}
            preds = {"test": d / "test_preds.jsonl", "dev": d / "dev_preds.jsonl"}
        else:
            entry["checkpoint"] = {"model": rel(d / "model") if (d / "model").exists() else None}
            if (d / "train_summary.json").exists():
                entry["training"] = load(d / "train_summary.json")
            preds = {t: d / f"eval_{t}" / "preds_per_example.jsonl" for t in ("test", "dev")}
        entry["logs"] = files(d)
        for tag, p in preds.items():
            done = p.exists() and (family == "llm" or (p.parent / "eval_results.json").exists())
            if done:
                entry[f"{tag}_preds"] = rel(p)
                entry[tag] = metrics(p)
        runs[d.name] = entry
    return runs


def seed_summary(runs):
    by_system = {}
    for r in runs.values():
        if "test" in r:
            by_system.setdefault(r["system"], []).append(r)
    out = {}
    for system, rs in by_system.items():
        entry = {"seeds": sorted(r["seed"] for r in rs)}
        for m in METRICS:
            v = np.array([r["test"][m] for r in rs])
            entry[m] = {"mean": round(float(v.mean()), 4),
                        "sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
                        "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}
        dev = [r["dev"]["pair_f1"] for r in rs if "dev" in r]
        entry["dev_pair_f1_mean"] = round(float(np.mean(dev)), 4) if dev else None
        out[system] = entry
    return out


def ensembles():
    out = {}
    for p in sorted((V3 / "results" / "ensembles").glob("*_metrics.json")):
        m = load(p)
        out[m["name"]] = {"file": rel(p), "rule": m["rule"], "members": list(m["members"]),
                          "scores": m["scores"][m["name"]],
                          "vs_member": {k: {t: {"delta": v[t]["delta"], "p": v[t]["p_value"],
                                                "p_holm": v[t]["p_holm"]} for t in v}
                                        for k, v in m["ensemble_vs_member"].items()},
                          "complementarity": m["complementarity"]}
    return out


def significance():
    tests = []
    for p in sorted((V3 / "results" / "significance").glob("*.json")):
        m = load(p)
        for metric, v in m["per_metric"].items():
            tests.append({"a": m["A_name"], "b": m["B_name"], "metric": metric,
                          "delta": v["obs_delta_A_minus_B"], "ci95": v["ci95"], "p": v["p_value"],
                          "file": rel(p)})
    for t, adj in zip(tests, _ensemble.holm([t["p"] for t in tests])):
        t["p_holm"] = adj
    return tests


def judge():
    out = {"calibration": {}, "corpus": {}}
    for p in sorted((V3 / "judge").glob("calibration_*/calibration.json")):
        c = load(p)
        out["calibration"][p.parent.name] = {
            "file": rel(p), "n": c["n_reviews"],
            "systems": {n: {"spearman_overall": a["overall"]["spearman"], **a["tiers"],
                            "ci95": c["bootstrap"].get(n)} for n, a in c["agreement"].items()},
            "inter_judge": c.get("inter_judge"),
            "cost_usd": {n: j.get("usd") for n, j in c["judges"].items()}}
    for p in sorted((V3 / "judge").glob("*/summary.json")):
        s = load(p)
        out["corpus"][p.parent.name] = {
            "file": rel(p), "annotations": s["annotations"], "n_items": s["n_items"],
            "n_scored": s["n_scored"], "include_rate": s["panel_include_rate"],
            "tiers": s["panel_tiers"], "means": s["panel_means"], "inter_judge": s["inter_judge"],
            "judges": {n: {k: j[k] for k in ("model", "scored", "tokens_in", "tokens_out", "usd")}
                       for n, j in s["judges"].items()}}
    return out


def manifest():
    entries = []
    roots = [V3] + [s["logs"] for s in V2_SYSTEMS.values()] + [s["preds"] for s in V2_SYSTEMS.values()]
    seen = set()
    for root in roots:
        paths = [root] if Path(root).is_file() else sorted(Path(root).rglob("*")) if Path(root).exists() else []
        for p in paths:
            if not p.is_file() or p.name in ("MANIFEST.json", ".DS_Store") or "__pycache__" in p.parts:
                continue
            if p in seen:
                continue
            seen.add(p)
            size = p.stat().st_size
            entry = {"path": rel(p), "bytes": size}
            if size <= HASH_LIMIT:
                entry["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            entries.append(entry)
    return entries


def git_info():
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


# ----------------------------------------------------------------------------
# Readable summary
# ----------------------------------------------------------------------------
def fmt(x):
    return "–" if x is None else f"{x:.3f}"


def summary_md(reg):
    lines = ["# UzABSA-LLM v3: results index", "",
             f"Built {reg['meta']['created']} from git `{reg['meta']['git']['commit'][:10]}`"
             + (" (uncommitted changes)" if reg["meta"]["git"]["dirty"] else "") + ". "
             "Every number is recomputed from the prediction files listed in `registry.json`; "
             "`MANIFEST.json` holds their SHA-256 hashes.", "",
             "## Paper systems (v2, 609 held-out reviews)", "",
             "| System | Exact ATE F1 | Partial F1 (1:1) | Pair F1 | Sent. acc. | Macro-F1 | Checkpoint |",
             "|---|---|---|---|---|---|---|"]
    for n, s in reg["v2_reference"].items():
        t = s["test"]
        ckpt = s["checkpoint"] or "not kept"
        if s.get("hub") and s["hub"]["status"].startswith("older"):
            ckpt += f" (**stale**: last updated {s['hub']['last_updated'][:10]})"
        lines.append(f"| {n} | {fmt(t['ate_exact_f1'])} | {fmt(t['ate_partial_1to1_f1'])} | {fmt(t['pair_f1'])} "
                     f"| {fmt(t['sentiment_accuracy'])} | {fmt(t['sentiment_macro_f1'])} | {ckpt} |")
    if reg["ensembles"]:
        lines += ["", "## Ensembles (majority vote, no training, no tuned parameter)", "",
                  "| Ensemble | Exact ATE F1 | Partial F1 (1:1) | Pair F1 | Sent. acc. | vs. best member (pair F1) |",
                  "|---|---|---|---|---|---|"]
        for n, e in reg["ensembles"].items():
            s = e["scores"]
            best = max(e["vs_member"], key=lambda k: -e["vs_member"][k]["pair_f1"]["delta"])
            v = e["vs_member"][best]["pair_f1"]
            lines.append(f"| {n} | {fmt(s['canonical']['ate_exact_f1'])} | {fmt(s['partial_1to1']['f1'])} "
                         f"| {fmt(s['canonical']['pair_f1'])} | {fmt(s['canonical']['sentiment_accuracy'])} "
                         f"| {v['delta']:+.3f} vs {best} (p={v['p']}, Holm {v['p_holm']}) |")
    if reg["seed_summary"]:
        lines += ["", "## v3 runs: mean ± sd over seeds (test = the same 609 reviews)", "",
                  "| System | Seeds | Exact ATE F1 | Partial F1 (1:1) | Pair F1 | Sent. acc. | Dev pair F1 |",
                  "|---|---|---|---|---|---|---|"]
        for n, s in reg["seed_summary"].items():
            cell = lambda m: f"{s[m]['mean']:.3f}" + (f" ± {s[m]['sd']:.3f}" if s[m]["sd"] is not None else "")
            lines.append(f"| {n} | {len(s['seeds'])} | {cell('ate_exact_f1')} | {cell('ate_partial_1to1_f1')} "
                         f"| {cell('pair_f1')} | {cell('sentiment_accuracy')} | {fmt(s['dev_pair_f1_mean'])} |")
    if reg["significance"]:
        sig = [t for t in reg["significance"] if t["p_holm"] < 0.05]
        lines += ["", f"## Significance (first seed): {len(sig)} of {len(reg['significance'])} tests "
                  "survive Holm correction", ""]
        lines += [f"- {t['a']} vs {t['b']}, {t['metric']}: {t['delta']:+.4f} (p={t['p']}, Holm {t['p_holm']})"
                  for t in sig]
    for tag, c in reg["judge"]["calibration"].items():
        lines += ["", f"## Judge calibration: {tag} (n={c['n']} human-rated reviews)", "",
                  "| Judge | Spearman (overall) | Tier κ | Linear κ | Tier agreement | Include precision |",
                  "|---|---|---|---|---|---|"]
        for n, s in c["systems"].items():
            lines.append(f"| {n} | {fmt(s['spearman_overall'])} | {fmt(s['kappa'])} | {fmt(s['kappa_linear'])} "
                         f"| {fmt(s['agreement'])} | {fmt(s['include_precision'])} |")
    for tag, c in reg["judge"]["corpus"].items():
        lines += ["", f"## Corpus judging: {tag}", "",
                  f"{c['n_scored']} of {c['n_items']} reviews scored; Include rate {c['include_rate']:.1%}; "
                  f"tiers {c['tiers']}; cost (USD) "
                  + ", ".join(f"{n}: {j['usd']}" for n, j in c["judges"].items()) + "."]
    lines += ["", "## Still missing", ""] + [f"- {m}" for m in reg["missing"]] if reg["missing"] else []
    return "\n".join(lines) + "\n"


def missing(reg):
    out = [f"{n}: the Hub branch `{s['hub']['branch']}` was last updated {s['hub']['last_updated'][:10]}, "
           f"before the run behind the paper's numbers; push that run (scripts/push_to_hub.py)"
           for n, s in reg["v2_reference"].items() if s.get("hub") and s["hub"]["status"].startswith("older")]
    llm =[(m, s) for m in CFG["llm"]["models"] + CFG["llm"].get("extra_models", []) for s in CFG["llm"]["seeds"]]
    enc = [(m, s) for m in CFG["encoders"]["models"] for s in CFG["encoders"]["seeds"]]
    runs = reg["runs"]
    for label, todo in (("LLM runs with test predictions", llm), ("encoder runs with test predictions", enc)):
        have = sum(1 for m, s in todo if "test" in runs.get(f"{m}_s{s}", {}))
        if have < len(todo):
            out.append(f"{label}: {have}/{len(todo)} (run_v3.py gpu / encoders)")
    if not reg["judge"]["calibration"]:
        out.append("judge calibration against the 150 human ratings (run_v3.py judge-calibrate)")
    for tag, what in (("release", "judging of all 5,038 released annotations (run_v3.py judge-release)"),
                      ("v3", "re-annotation and judging of the corpus (run_v3.py annotate judge-v3 assemble)")):
        if not any(k.startswith(f"{tag}_") for k in reg["judge"]["corpus"]):
            out.append(what)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--offline", action="store_true", help="skip the Hugging Face branch check")
    args = ap.parse_args()

    split = V3 / "data" / "split_stats.json"
    runs = v3_runs()
    reg = {"meta": {"created": datetime.now().isoformat(timespec="seconds"), "git": git_info(),
                    "python": sys.version.split()[0], "platform": platform.platform()},
           "split": load(split) if split.exists() else None,
           "v2_reference": v2_reference(args.offline), "runs": runs, "seed_summary": seed_summary(runs),
           "ensembles": ensembles(), "significance": significance(), "judge": judge()}
    reg["missing"] = missing(reg)
    (V3 / "registry.json").write_text(json.dumps(reg, indent=1, ensure_ascii=False), encoding="utf-8")
    (V3 / "SUMMARY.md").write_text(summary_md(reg), encoding="utf-8")
    man = {"created": reg["meta"]["created"], "git": reg["meta"]["git"], "files": manifest()}
    (V3 / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"registry: {len(reg['v2_reference'])} paper systems, {len(runs)} v3 runs, "
          f"{len(reg['ensembles'])} ensembles, {len(reg['significance'])} tests; "
          f"manifest: {len(man['files'])} files")
    for m in reg["missing"]:
        print(f"missing: {m}")
    print(f"Saved -> {rel(V3 / 'registry.json')}, {rel(V3 / 'SUMMARY.md')}, {rel(V3 / 'MANIFEST.json')}")


if __name__ == "__main__":
    main()
