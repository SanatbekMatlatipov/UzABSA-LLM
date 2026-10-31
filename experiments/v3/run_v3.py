#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM v3: one runner for the stronger-paper experiments
# =============================================================================
"""
Runs the v3 protocol stage by stage and keeps every checkpoint, log,
prediction and judge output under experiments/v3/. A stage skips any job
whose output already exists, so a crash or power cut costs one job, not the
whole run. Each job's full console output goes to logs/<stage>/<job>.log, and
logs/runner.log records the command, host, git commit, duration and exit code.

Stages, in order:
  split            train / dev / test, where test = the 609 held-out reviews
  train-llm        QLoRA runs, every model x seed; best checkpoint by dev loss
  eval-llm         test and dev predictions of every LLM run
  bio              BIO version of the v3 split for the encoders
  train-enc        encoder runs, every model x seed; best epoch by dev F1;
                   then their test and dev predictions
  ensemble         majority votes over the seed-matched systems and all runs
  significance     paired bootstrap tests between the first-seed systems
  annotate         re-annotate the 5,038 reviews with the best LLM run on dev
  judge-calibrate  judge panel vs the 150 human ratings (API calls)
  judge-release    judge panel on the released annotations (API calls, no GPU)
  judge-v3         judge panel on the re-annotated corpus (API calls)
  assemble         Include / Flag / Exclude tiers for the released and the
                   re-annotated corpus, from their panel scores
  collect          registry.json, MANIFEST.json and the summary tables

Groups: gpu = split train-llm eval-llm annotate; encoders = split bio train-enc;
judge = judge-calibrate judge-release assemble; report = ensemble significance
collect; all = every stage.

Usage (from the repository root):
    python experiments/v3/run_v3.py gpu              # GPU machine
    python experiments/v3/run_v3.py encoders         # Mac (MPS) or GPU machine
    python experiments/v3/run_v3.py judge            # API keys from judges.json
    python experiments/v3/run_v3.py report
    python experiments/v3/run_v3.py all --dry-run    # print the commands only

Author: UzABSA Team
License: MIT
"""

import argparse
import importlib.util as ilu
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

V3 = Path(__file__).resolve().parent
ROOT = V3.parent.parent
PY = sys.executable
CFG = json.load(open(V3 / "config.json", encoding="utf-8"))
DATA, BIO, RUNS, LOGS, RES = V3 / "data", V3 / "data_bio", V3 / "runs", V3 / "logs", V3 / "results"
ANALYSIS_ENV = {"PYTHONHASHSEED": "0"}   # canonical partial F1 depends on set order
R2 = ROOT / "paper_materials" / "revision_v2"
# predictions behind the paper's Table 2, in the order that breaks polarity ties
V2_PREDS = {"Qwen 2.5-7B": R2 / "results_v2" / "qwen2.5-7b_preds.jsonl",
            "Llama 3.1-8B": R2 / "results_v2" / "llama3.1-8b_preds.jsonl",
            "DeepSeek-R1-7B": R2 / "results_v2" / "deepseek-7b_preds.jsonl",
            "TahrirchiBERT": R2 / "baselines_v2" / "tahrirchi-bert" / "preds_per_example.jsonl",
            "BERTbek": R2 / "baselines_v2" / "bertbek" / "preds_per_example.jsonl"}


def rel(p):
    return os.path.relpath(Path(p).resolve(), ROOT)


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def git_rev():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


# ----------------------------------------------------------------------------
# Runs and their files
# ----------------------------------------------------------------------------
def llm_runs():
    c = CFG["llm"]
    return [(m, s, f"{m}_s{s}") for m in c["models"] + c.get("extra_models", []) for s in c["seeds"]]


def enc_runs():
    c = CFG["encoders"]
    return [(name, hf, s, f"{name}_s{s}") for name, hf in c["models"].items() for s in c["seeds"]]


def model_dir(run):
    """Merged model, or the adapter-only fallback train_unsloth.py writes if merging fails."""
    merged = RUNS / run / "merged_model"
    return merged if merged.exists() or not (RUNS / run / "model").exists() else RUNS / run / "model"


def pred_path(run, tag, family):
    if family == "llm":
        return RUNS / run / f"{tag}_preds.jsonl"
    return RUNS / run / f"eval_{tag}" / "preds_per_example.jsonl"


def display(model):
    return CFG.get("names", {}).get(model, model)


def best_llm_on_dev():
    """LLM run with the highest dev pair F1 (dev, never test, decides)."""
    spec = ilu.spec_from_file_location("uzabsa_robustness", ROOT / "scripts" / "robustness_analysis.py")
    robust = ilu.module_from_spec(spec)
    spec.loader.exec_module(robust)
    scores = {}
    for _, _, run in llm_runs():
        p = pred_path(run, "dev", "llm")
        if p.exists():
            scores[run] = robust.canonical(robust.load_jsonl(p))["pair_f1"]
    if not scores:
        return None, {}
    return max(scores, key=scores.get), scores


# ----------------------------------------------------------------------------
# Job execution
# ----------------------------------------------------------------------------
class Runner:
    def __init__(self, dry, force):
        self.dry, self.force = dry, force
        self.failed, self.lock, self.rev = [], threading.Lock(), git_rev()

    def job(self, stage, name, cmd, done=None, env=None):
        """Run one command unless one of the `done` paths exists. Returns success."""
        done = [done] if isinstance(done, (str, Path)) else (done or [])
        hit = next((p for p in done if Path(p).exists()), None)
        if hit and not self.force:
            print(f"[skip] {stage}/{name}: {rel(hit)} exists")
            return True
        env = env or {}
        shown = " ".join(f"{k}={v}" for k, v in env.items()) + (" " if env else "") \
            + " ".join(shlex.quote(str(c)) for c in cmd)
        if self.dry:
            print(f"[dry] {stage}/{name}: {shown}")
            return True
        log = LOGS / stage / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        start, t0 = datetime.now().isoformat(timespec="seconds"), time.time()
        print(f"[run] {stage}/{name} -> {rel(log)}")
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"# {start} host={socket.gethostname()} git={self.rev}\n# {shown}\n")
            f.flush()
            rc = subprocess.run([str(c) for c in cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                env={**os.environ, **env}).returncode
            f.write(f"# exit {rc} after {time.time() - t0:.0f}s\n")
        record = {"stage": stage, "job": name, "cmd": shown, "start": start,
                  "seconds": round(time.time() - t0), "exit": rc, "host": socket.gethostname(),
                  "git": self.rev, "log": rel(log)}
        with self.lock:
            with open(LOGS / "runner.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if rc:
                self.failed.append(f"{stage}/{name}")
        print(f"[{'ok' if rc == 0 else 'FAIL'}] {stage}/{name} ({record['seconds']}s)")
        return rc == 0

    def on_gpus(self, stage, jobs):
        """jobs: (name, cmd, done). One worker per configured GPU, each running its jobs in turn."""
        gpus = CFG["llm"]["gpus"]
        queues = {g: jobs[i::len(gpus)] for i, g in enumerate(gpus)}

        def worker(g):
            for name, cmd, done in queues[g]:
                self.job(stage, name, cmd, done, env={"CUDA_VISIBLE_DEVICES": str(g)})
        threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
        for t in threads:
            t.start()
            if self.dry:
                t.join()                 # keep dry-run output in order
        for t in threads:
            t.join()


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def st_split(r):
    c = CFG["split"]
    r.job("split", "split", [PY, "scripts/make_dev_split.py", "--src", c["source"], "--out", rel(DATA),
                             "--dev-size", c["dev_size"], "--seed", c["seed"]], DATA / "split_stats.json")


def st_train_llm(r):
    jobs = []
    for model, seed, run in llm_runs():
        cmd = [PY, "scripts/train_unsloth.py", "--model", model, "--dataset", rel(DATA),
               "--output-dir", rel(RUNS), "--run-name", run, "--seed", seed, "--no-wandb"]
        for k, v in CFG["llm"]["train_args"].items():
            cmd += [k, v]
        jobs.append((run, cmd, [RUNS / run / "merged_model" / "config.json",
                                RUNS / run / "model" / "config.json"]))
    r.on_gpus("train-llm", jobs)


def st_eval_llm(r):
    jobs = []
    for _, _, run in llm_runs():
        for split, tag in (("test", "test"), ("validation", "dev")):
            out = pred_path(run, tag, "llm")
            jobs.append((f"{run}_{tag}", [PY, "scripts/evaluate.py", "--model-path", rel(model_dir(run)),
                                          "--test-data", rel(DATA), "--split", split,
                                          "--output-dir", rel(RUNS / run / f"eval_{tag}"),
                                          "--save-preds", rel(out)], out))
    r.on_gpus("eval-llm", jobs)


def st_bio(r):
    r.job("bio", "bio", [PY, "scripts/prepare_bio_dataset.py", "--data", rel(DATA), "--out", rel(BIO)],
          BIO / "stats.json")


def st_train_enc(r):
    c = CFG["encoders"]
    for _, hf, seed, run in enc_runs():
        cmd = [PY, "scripts/train_bert_baseline.py", "--model", hf, "--bio-dir", rel(BIO),
               "--out", rel(RUNS / run), "--seed", seed, "--device", c["device"]]
        for k, v in c["train_args"].items():
            cmd += [k, v]
        if not r.job("train-enc", run, cmd, RUNS / run / "train_summary.json"):
            continue
        for split, tag in (("test", "test"), ("validation", "dev")):
            out = RUNS / run / f"eval_{tag}"
            r.job("train-enc", f"{run}_{tag}",
                  [PY, "scripts/eval_bert_baseline.py", "--model-dir", rel(RUNS / run / "model"),
                   "--data", rel(DATA), "--split", split, "--out", rel(out), "--device", c["device"]],
                  out / "eval_results.json")       # written last; the preds file grows as it runs


def ensemble_job(r, name, members):
    missing = [rel(p) for _, p in members if not p.exists()]
    if missing and not r.dry:
        print(f"[wait] ensemble/{slug(name)}: {len(missing)} prediction files missing, e.g. {missing[0]}")
        return
    r.job("ensemble", slug(name),
          [PY, "scripts/ensemble_preds.py", "--name", name, "--min-votes", len(members) // 2 + 1,
           "--out-dir", rel(RES / "ensembles"), "--systems", *[f"{n}={rel(p)}" for n, p in members]],
          RES / "ensembles" / f"{slug(name)}_metrics.json", env=ANALYSIS_ENV)


def st_ensemble(r):
    v2 = list(V2_PREDS.items())
    ensemble_job(r, f"Majority of {len(v2)} (v2 systems)", v2)
    ensemble_job(r, "Majority of 3 LLMs (v2 systems)", v2[:3])
    llms, encs = CFG["llm"]["models"], list(CFG["encoders"]["models"])
    seeds = [s for s in CFG["llm"]["seeds"] if s in CFG["encoders"]["seeds"]]
    everything = []
    for s in seeds:
        members = [(display(m), pred_path(f"{m}_s{s}", "test", "llm")) for m in llms] \
            + [(e, pred_path(f"{e}_s{s}", "test", "enc")) for e in encs]
        ensemble_job(r, f"Majority of {len(members)} (seed {s})", members)
        ensemble_job(r, f"Majority of {len(llms)} LLMs (seed {s})", members[:len(llms)])
        everything += [(f"{n} s{s}", p) for n, p in members]
    if len(seeds) > 1:
        ensemble_job(r, f"Majority of all {len(everything)} runs", everything)


def st_significance(r):
    s = CFG["llm"]["seeds"][0]
    systems = {display(m): pred_path(f"{m}_s{s}", "test", "llm") for m in CFG["llm"]["models"]}
    systems.update({e: pred_path(f"{e}_s{s}", "test", "enc") for e in CFG["encoders"]["models"]})
    missing = [n for n, p in systems.items() if not p.exists()]
    if missing and not r.dry:
        print(f"[wait] significance: no seed-{s} test predictions for {', '.join(missing)}")
        return
    for a, b in combinations(systems, 2):
        out = RES / "significance" / f"{slug(a)}_vs_{slug(b)}.json"
        r.job("significance", f"{slug(a)}_vs_{slug(b)}",
              [PY, "scripts/significance_test.py", "--a", rel(systems[a]), "--a-name", a,
               "--b", rel(systems[b]), "--b-name", b, "--out", rel(out)], out, env=ANALYSIS_ENV)


def st_annotate(r):
    best, scores = best_llm_on_dev()
    if best is None and not r.dry:
        print("[wait] annotate: no dev predictions yet (run eval-llm first)")
        return
    run = best or "<best-run-on-dev>"
    print(f"annotator = {run}; dev pair F1 by run: {scores}")
    r.job("annotate", run, [PY, "scripts/annotate_reviews.py", "--model-path", rel(model_dir(run)),
                            "--output-dir", rel(V3 / "annotated"), "--source-name", f"{run} (v3)",
                            "--transliterate-cyrillic"],
          V3 / "annotated" / "reviews_annotated.json",
          env={"CUDA_VISIBLE_DEVICES": str(CFG["llm"]["gpus"][0])})


def judge_score(r, tag, annotations):
    rubric = CFG["judge"]["rubrics"][0]           # the primary rubric; others are ablations
    out = V3 / "judge" / f"{tag}_{rubric}"
    if not Path(annotations).exists() and not r.dry:
        print(f"[wait] judge/{tag}: {rel(annotations)} does not exist")
        return
    r.job("judge", f"{tag}_{rubric}",
          [PY, "scripts/judge_panel.py", "score", "--judges", rel(V3 / "judges.json"), "--rubric", rubric,
           "--annotations", rel(annotations), "--workers", CFG["judge"]["workers"], "--out", rel(out)],
          out / "judge_results.json")


def st_judge_calibrate(r):
    for rubric in CFG["judge"]["rubrics"]:
        out = V3 / "judge" / f"calibration_{rubric}"
        r.job("judge", f"calibration_{rubric}",
              [PY, "scripts/judge_panel.py", "calibrate", "--judges", rel(V3 / "judges.json"),
               "--rubric", rubric, "--workers", CFG["judge"]["workers"], "--out", rel(out)],
              out / "calibration.json")


def st_judge_release(r):
    judge_score(r, "release", ROOT / "data" / "annotated" / "reviews_annotated.json")


def st_judge_v3(r):
    judge_score(r, "v3", V3 / "annotated" / "reviews_annotated.json")


def assemble_job(r, tag, annotations):
    results = V3 / "judge" / f"{tag}_{CFG['judge']['rubrics'][0]}" / "judge_results.json"
    if not results.exists() and not r.dry:
        print(f"[wait] assemble/{tag}: {rel(results)} does not exist (run judge-{tag} first)")
        return
    out = V3 / f"final_dataset_{tag}"
    r.job("assemble", tag, [PY, "scripts/assemble_dataset.py", "--annotations", rel(annotations),
                            "--judge-results", rel(results), "--output-dir", rel(out)],
          out / "dataset_stats.json")


def st_assemble(r):
    assemble_job(r, "release", ROOT / "data" / "annotated" / "reviews_annotated.json")
    assemble_job(r, "v3", V3 / "annotated" / "reviews_annotated.json")


def st_collect(r):
    rubric = CFG["judge"]["rubrics"][0]
    corpus = V3 / "judge" / f"release_{rubric}" / "judge_results.json"
    if corpus.exists() or r.dry:              # the judge numbers the paper reports
        r.job("collect", "judge_paper_numbers",
              [PY, "scripts/judge_paper_numbers.py", "--corpus", rel(corpus),
               "--repeat", rel(V3 / "judge" / f"calibration_{rubric}" / "raw"),
               "--out", rel(RES / "judge_paper_numbers.json")], env=ANALYSIS_ENV)
    r.job("collect", "collect", [PY, "scripts/collect_v3.py"], env=ANALYSIS_ENV)


STAGES = {"split": st_split, "train-llm": st_train_llm, "eval-llm": st_eval_llm, "bio": st_bio,
          "train-enc": st_train_enc, "ensemble": st_ensemble, "significance": st_significance,
          "annotate": st_annotate, "judge-calibrate": st_judge_calibrate,
          "judge-release": st_judge_release, "judge-v3": st_judge_v3, "assemble": st_assemble,
          "collect": st_collect}
GROUPS = {"gpu": ["split", "train-llm", "eval-llm", "annotate"],
          "encoders": ["split", "bio", "train-enc"],
          "judge": ["judge-calibrate", "judge-release", "assemble"],
          "report": ["ensemble", "significance", "collect"],
          "all": list(STAGES)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n\n".join(__doc__.split("\n\n")[2:4]))
    ap.add_argument("stages", nargs="+", choices=list(STAGES) + list(GROUPS))
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--force", action="store_true", help="rerun jobs whose outputs exist")
    args = ap.parse_args()

    order = []
    for s in args.stages:
        for stage in GROUPS.get(s, [s]):
            if stage not in order:
                order.append(stage)
    runner = Runner(args.dry_run, args.force)
    LOGS.mkdir(parents=True, exist_ok=True)
    for stage in order:
        print(f"\n=== {stage} ===")
        STAGES[stage](runner)
    if runner.failed:
        print(f"\nFAILED ({len(runner.failed)}): " + ", ".join(runner.failed)
              + f"\nSee {rel(LOGS)}/<stage>/<job>.log")
        sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
