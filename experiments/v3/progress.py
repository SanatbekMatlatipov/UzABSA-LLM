#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM v3: live progress of the pipeline
# =============================================================================
"""Show how far run_v3.py has got, reading only its logs and output files.

Nothing here writes anything, so it is safe to run while the pipeline is
working. Job logs are appended when a job is retried, so every log is read
from its last "# <timestamp> host=" header onwards -- otherwise a finished
earlier attempt would be mistaken for the current one.

Usage (from the repository root):
    python experiments/v3/progress.py            # print once
    python experiments/v3/progress.py -w         # refresh every 10 s
    python experiments/v3/progress.py -w 5       # refresh every 5 s
    python experiments/v3/progress.py --no-color

Author: UzABSA Team
License: MIT
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

V3 = Path(__file__).resolve().parent
ROOT = V3.parent.parent
RUNS, LOGS, RES = V3 / "runs", V3 / "logs", V3 / "results"

HEADER_RE = re.compile(r"^# \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} host=", re.M)
# tqdm: "  53/1000 [08:24<2:56:05, 11.16s/it]"
TQDM_RE = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)(s/it|it/s)\]")
EXIT_RE = re.compile(r"^# exit (\d+) after (\d+)s", re.M)

C = {"reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m", "red": "\033[31m",
     "green": "\033[32m", "yellow": "\033[33m", "blue": "\033[36m", "grey": "\033[90m"}


def paint(s, *styles):
    if not COLOR:
        return s
    return "".join(C[x] for x in styles) + s + C["reset"]


# ----------------------------------------------------------------------------
# Reading logs and results
# ----------------------------------------------------------------------------
def latest_attempt(path):
    """Text of a job log from its last run header on, or "" if there is none."""
    p = Path(path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    starts = [m.start() for m in HEADER_RE.finditer(text)]
    return text[starts[-1]:] if starts else text


def job_state(log_path):
    """(state, step, total, eta, rate, seconds) for one job log.

    state is one of: missing, running, saving, ok, fail. "saving" matters for
    the LLM runs: merging and writing the 15 GB checkpoint takes minutes and
    has progress bars of its own, which would otherwise read as a step count.
    """
    chunk = latest_attempt(log_path)
    if not chunk:
        return "missing", None, None, None, None, None
    done = EXIT_RE.search(chunk)
    steps = TQDM_RE.findall(chunk.replace("\r", "\n"))
    step = total = eta = rate = None
    if steps:
        s = steps[-1]
        step, total, eta = int(s[0]), int(s[1]), s[3]
        rate = f"{s[4]}{s[5]}"
    if done:
        code, secs = int(done.group(1)), int(done.group(2))
        return ("ok" if code == 0 else "fail"), step, total, eta, rate, secs
    if "Saving model" in chunk:
        return "saving", step, total, eta, rate, None
    return "running", step, total, eta, rate, None


def metrics(eval_dir):
    """exact ATE F1 and pair F1 from an eval dir, whichever results file it has."""
    files = sorted(glob.glob(str(Path(eval_dir) / "eval_results*.json")))
    if not files:
        return None
    try:
        d = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
        return (d["aspect_term_extraction"]["exact_match"]["f1"],
                d["aspect_polarity_pairs"]["pair_f1"])
    except (OSError, KeyError, ValueError):
        return None


def bar(done, total, width=10):
    if not total:
        return " " * (width + 2)
    n = max(0, min(width, round(width * done / total)))
    return "[" + "#" * n + "-" * (width - n) + "]"


def fmt_secs(s):
    if s is None:
        return ""
    return f"{s // 60}m" if s < 3600 else f"{s / 3600:.1f}h"


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------
def cell_train(state, step, total, eta, rate, secs):
    if state == "missing":
        return paint("pending", "grey")
    if state == "ok":
        return paint("ok", "green") + paint(f" {fmt_secs(secs)}", "dim")
    if state == "fail":
        return paint("FAILED", "red")
    if state == "saving":
        return paint("merging + saving weights", "yellow")
    if step is None:
        return paint("starting", "yellow")
    return f"{bar(step, total)} {step}/{total} " + paint(f"eta {eta}", "dim")


def cell_eval(preds, eval_dir, log_path):
    """A prediction file is the proof an eval finished; the log shows live progress."""
    if Path(preds).exists():
        m = metrics(eval_dir)
        if m:
            return paint(f"ok {m[0]:.3f}/{m[1]:.3f}", "green")
        return paint("ok", "green")
    state, step, total, eta, _, _ = job_state(log_path)
    if state == "missing":
        return paint("-", "grey")
    if state == "fail":
        return paint("FAILED", "red")
    if step is None:
        return paint("starting", "yellow")
    return f"{bar(step, total, 6)} {step}/{total}"


def llm_section(cfg, out):
    models = cfg["llm"]["models"] + cfg["llm"].get("extra_models", [])
    seeds = cfg["llm"]["seeds"]
    runs = [(m, s, f"{m}_s{s}") for m in models for s in seeds]
    trained = sum(1 for _, _, r in runs
                  if (RUNS / r / "merged_model" / "config.json").exists()
                  or (RUNS / r / "model" / "config.json").exists())
    scored = sum(1 for _, _, r in runs if (RUNS / r / "test_preds.jsonl").exists())
    out.append(paint(f"LLM runs ({len(models)} models x {len(seeds)} seeds)", "bold")
               + paint(f"   trained {trained}/{len(runs)}   test-scored {scored}/{len(runs)}", "dim"))
    out.append(paint(f"  {'run':<20}{'train':<34}{'test exact/pair':<22}{'dev exact/pair'}", "dim"))
    for _, _, r in runs:
        tr = cell_train(*job_state(LOGS / "train-llm" / f"{r}.log"))
        te = cell_eval(RUNS / r / "test_preds.jsonl", RUNS / r / "eval_test",
                       LOGS / "eval-llm" / f"{r}_test.log")
        dv = cell_eval(RUNS / r / "dev_preds.jsonl", RUNS / r / "eval_dev",
                       LOGS / "eval-llm" / f"{r}_dev.log")
        out.append(f"  {r:<20}{pad(tr, 34)}{pad(te, 22)}{dv}")
    return trained + scored, 2 * len(runs)


def enc_section(cfg, out):
    names, seeds = list(cfg["encoders"]["models"]), cfg["encoders"]["seeds"]
    runs = [f"{n}_s{s}" for n in names for s in seeds]
    trained = sum(1 for r in runs if (RUNS / r / "train_summary.json").exists())
    scored = sum(1 for r in runs
                 if (RUNS / r / "eval_test" / "preds_per_example.jsonl").exists())
    out.append("")
    out.append(paint(f"Encoder runs ({len(names)} models x {len(seeds)} seeds)", "bold")
               + paint(f"   trained {trained}/{len(runs)}   test-scored {scored}/{len(runs)}", "dim"))
    for r in runs:
        tr = cell_train(*job_state(LOGS / "train-enc" / f"{r}.log"))
        te = cell_eval(RUNS / r / "eval_test" / "preds_per_example.jsonl",
                       RUNS / r / "eval_test", LOGS / "train-enc" / f"{r}_test.log")
        dv = cell_eval(RUNS / r / "eval_dev" / "preds_per_example.jsonl",
                       RUNS / r / "eval_dev", LOGS / "train-enc" / f"{r}_dev.log")
        out.append(f"  {r:<20}{pad(tr, 34)}{pad(te, 22)}{dv}")
    return trained + scored, 2 * len(runs)


def downstream_section(out):
    """The stages after the runs: re-annotation, judging, tiers and the report."""
    done = 0
    items = []

    ann = V3 / "annotated" / "reviews_annotated.json"
    ckpt = V3 / "annotated" / "annotation_checkpoint.json"
    if ann.exists():
        items.append(("annotate (5,038 reviews)", paint("ok", "green")))
        done += 1
    elif ckpt.exists():
        try:
            c = json.loads(ckpt.read_text(encoding="utf-8"))
            nxt = c.get("next_idx", 0)
            items.append(("annotate (5,038 reviews)", f"{bar(nxt, 5038)} {nxt}/5038"))
        except (OSError, ValueError):
            items.append(("annotate (5,038 reviews)", paint("running", "yellow")))
    else:
        items.append(("annotate (5,038 reviews)", paint("pending", "grey")))

    jv3 = V3 / "judge" / "v3_v1" / "judge_results.json"
    items.append(("judge-v3 (needs OPENAI_API_KEY)",
                  paint("ok", "green") if jv3.exists() else paint("pending", "grey")))
    done += jv3.exists()

    asm = V3 / "final_dataset_v3" / "dataset_stats.json"
    items.append(("assemble v3 tiers",
                  paint("ok", "green") if asm.exists() else paint("pending", "grey")))
    done += asm.exists()

    ens = len(glob.glob(str(RES / "ensembles" / "*_metrics.json")))
    items.append(("ensembles", paint(f"{ens} built", "green" if ens else "grey")))
    done += ens > 2          # 2 exist already from the v2-systems votes

    sig = len(glob.glob(str(RES / "significance" / "*.json")))
    items.append(("significance tests", paint(f"{sig} done", "green" if sig else "grey")))
    done += sig > 0

    reg = V3 / "registry.json"
    stamp = (datetime.fromtimestamp(reg.stat().st_mtime).strftime("%d %b %H:%M")
             if reg.exists() else "never")
    items.append(("collect (registry/SUMMARY/MANIFEST)", paint(stamp, "dim")))

    out.append("")
    out.append(paint("Downstream stages", "bold"))
    for label, val in items:
        out.append(f"  {label:<36}{val}")
    return done, 5


def machine_section(out):
    out.append("")
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        rows = [r.split(", ") for r in q.stdout.strip().splitlines() if r.strip()]
        cells = []
        for i, used, tot, util, pw in rows:
            busy = float(util) > 50
            cells.append(paint(f"GPU{i} {float(used)/1024:4.1f}/{float(tot)/1024:.0f}G "
                               f"{util:>3}% {float(pw):3.0f}W", "green" if busy else "grey"))
        out.append(paint("Machine", "bold") + "  " + "  ".join(cells))
    except (OSError, subprocess.SubprocessError, ValueError):
        out.append(paint("Machine", "bold") + "  (nvidia-smi unavailable)")
    free = shutil.disk_usage(ROOT).free / 1e9
    warn = free < 40
    out.append("  disk free on the repo drive: "
               + paint(f"{free:.0f} GB", "red" if warn else "dim")
               + (paint("  <-- low: merged models need ~15 GB each", "red") if warn else ""))


def active_section(out):
    """What the runner last said it started, so a stall is visible."""
    lines = []
    for name, f in (("llm", LOGS / "stage_llm.out"), ("encoders", LOGS / "stage_encoders.out")):
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        seed = re.findall(r"=== SEED (\d+) ===", text)
        started = re.findall(r"^\[run\] (\S+)", text, re.M)
        tail = f"seed {seed[-1]}" if seed else ""
        if started:
            tail = (tail + "  " if tail else "") + f"last started {started[-1]}"
        if tail:
            lines.append(f"  {name:<10}{tail}")
    if lines:
        out.append("")
        out.append(paint("Runner", "bold"))
        out.extend(lines)


def pad(s, width):
    """Left-justify ignoring ANSI escapes, which have no printed width."""
    bare = re.sub(r"\033\[[0-9;]*m", "", s)
    return s + " " * max(1, width - len(bare))


def plan():
    """The models and seeds the run covers.

    logs/plan.json wins when it exists: the batched LLM driver narrows
    config.json's llm.seeds to one seed at a time to cap peak disk, so
    config.json on its own would understate the plan while that runs.
    """
    cfg = json.loads((V3 / "config.json").read_text(encoding="utf-8"))
    p = LOGS / "plan.json"
    if p.exists():
        try:
            saved = json.loads(p.read_text(encoding="utf-8"))
            for fam in ("llm", "encoders"):
                for key in ("models", "seeds"):
                    if key in saved.get(fam, {}):
                        cfg[fam][key] = saved[fam][key]
            cfg["llm"]["extra_models"] = []
        except (OSError, KeyError, ValueError):
            pass
    return cfg


def render():
    cfg = plan()
    out = []
    a1, t1 = llm_section(cfg, out)
    a2, t2 = enc_section(cfg, out)
    a3, t3 = downstream_section(out)
    active_section(out)
    machine_section(out)
    pct = 100 * (a1 + a2 + a3) / (t1 + t2 + t3)
    head = (paint("UzABSA-LLM v3 pipeline", "bold") + "   "
            + datetime.now().strftime("%d %b %H:%M:%S") + "   "
            + paint(f"{bar(a1 + a2 + a3, t1 + t2 + t3, 24)} {pct:.0f}%", "blue"))
    return "\n".join([head, "=" * 78] + out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-w", "--watch", nargs="?", type=int, const=10, default=None,
                    metavar="SECONDS", help="refresh forever (default every 10 s)")
    ap.add_argument("--no-color", action="store_true", help="plain text, no ANSI colour")
    args = ap.parse_args()

    global COLOR
    COLOR = not args.no_color and sys.stdout.isatty()
    if COLOR and os.name == "nt":
        os.system("")                      # let the console interpret ANSI escapes

    if args.watch is None:
        print(render())
        return
    try:
        while True:
            # redraw from the top instead of scrolling, and clear what is below
            sys.stdout.write("\033[H\033[2J" if COLOR else "\n" * 3)
            print(render())
            print(paint(f"\nrefreshing every {args.watch}s -- Ctrl+C to stop", "dim"))
            sys.stdout.flush()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print()


COLOR = True

if __name__ == "__main__":
    main()
