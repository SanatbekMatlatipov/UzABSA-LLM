#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: LLM-as-Judge panel, calibrated against the human ratings
# =============================================================================
"""
Scores ABSA annotations with a panel of LLM judges instead of a single one,
and measures every judge and the panel against the native-speaker ratings
before any of them is used on the corpus.

Rubrics:
  v1         the prompt of llm_judge.py (the published GPT-4o-mini run), for a
             like-for-like comparison
  checklist  evidence first: the judge lists the review's opinions and the
             missed, spurious and mislabelled annotations, then scores the five
             dimensions as the human annotation protocol defines them. It sees
             what the human raters saw (domain, review, annotation), not the
             star rating or the business name.

Judges are OpenAI-compatible chat endpoints (OpenAI, Gemini's OpenAI endpoint,
OpenRouter, Together, Groq, a local vLLM server) listed in a JSON file with the
environment variable that holds each key. Every response is appended to
<out>/raw/<judge>.jsonl with its token usage as soon as it arrives, so an
interrupted run resumes without paying twice and the cost is measured.

Commands:
  calibrate  the 150 human-rated reviews: per judge and for the panel,
             Spearman rho and MAE per dimension, tier agreement, Cohen's kappa
             (unweighted and linear), Include precision/recall, bootstrap CIs,
             and paired bootstrap tests against the published judge
  score      any annotations file, e.g. all 5,038 reviews: writes
             judge_results.json for assemble_dataset.py, with the panel mean as
             judge_scores and every judge's scores alongside

Usage:
    python scripts/judge_panel.py calibrate --judges experiments/v3/judges.json \
        --rubric checklist --out experiments/v3/judge/calibration_checklist
    python scripts/judge_panel.py score --judges experiments/v3/judges.json \
        --annotations data/annotated/reviews_annotated.json --rubric checklist \
        --out experiments/v3/judge/release_checklist
    # --dry-run: no API calls; prints a prompt and the token estimate
    # --smoke:   one real call per judge, to check keys and model names

Author: UzABSA Team
License: MIT
"""

import argparse
import csv
import importlib.util as ilu
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path

import httpx
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

ROOT = Path(__file__).parent.parent


def _load(name, path):
    spec = ilu.spec_from_file_location(name, path)
    module = ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v1 = _load("uzabsa_llm_judge", ROOT / "scripts" / "llm_judge.py")
_robust = _load("uzabsa_robustness", ROOT / "scripts" / "robustness_analysis.py")
_hv = _load("uzabsa_human_validation", ROOT / "scripts" / "analyze_human_validation.py")
tier, cohen_kappa = _robust.tier, _robust.cohen_kappa

DIMS = ["completeness", "accuracy", "sentiment", "relevance", "overall"]
EVIDENCE = ["opinions", "missed", "spurious", "wrong_polarity", "bad_category", "explanation"]
TIERS = ["exclude", "flag", "include"]
HV_DIR = ROOT / "paper_materials" / "revision_v2" / "human_validation"
PUBLISHED_JUDGE = ROOT / "data" / "judged" / "judge_results.json"
# rough output length per call, for --dry-run estimates only
EST_OUTPUT_TOKENS = {"v1": 80, "checklist": 320}

CHECKLIST_SYSTEM = """\
You are a native-level expert in Uzbek (Latin and Cyrillic script; reviews \
sometimes mix in Russian) who audits aspect-based sentiment analysis (ABSA) \
annotations of customer reviews. Judge only from evidence in the review text. \
Return a single JSON object and nothing else."""

CHECKLIST_USER = """\
Domain: {business_category}
Review: \"\"\"{text}\"\"\"

Annotation to audit ({n} aspects, written as: term [category] -> polarity):
{annotation}

Step 1 - opinions: list every opinion target the review evaluates (an entity, \
part or attribute that receives a positive, negative or neutral evaluation) \
with its polarity, in the review's own words. Skip things mentioned without \
an evaluation.
Step 2 - compare the annotation with step 1:
  "missed": opinions from step 1 that the annotation does not cover;
  "spurious": annotated terms that do not occur in the review (inflected or \
lightly paraphrased forms count as occurring) or are not opinion targets;
  "wrong_polarity": annotated terms with the wrong polarity;
  "bad_category": annotated terms whose category does not fit the term or the domain.
Step 3 - score each dimension from 1 (very poor) to 5 (excellent):
  completeness: are all important opinions in the review captured?
  accuracy: do the annotated terms really occur in the review, rather than being invented?
  sentiment: are the polarities correct?
  relevance: do the aspects and categories fit this review and domain?
  overall: the overall quality of the annotation.
If the annotation is empty, give low completeness and overall scores when the \
review contains opinions, and high scores when it contains none.

Return exactly this JSON object:
{{"opinions": [{{"term": "<text>", "polarity": "<positive|negative|neutral>"}}], \
"missed": ["<term>"], "spurious": ["<term>"], "wrong_polarity": ["<term>"], \
"bad_category": ["<term>"], "completeness": <int 1-5>, "accuracy": <int 1-5>, \
"sentiment": <int 1-5>, "relevance": <int 1-5>, "overall": <int 1-5>}}"""


def build_prompt(item, rubric):
    if rubric == "v1":
        return _v1.JUDGE_SYSTEM_PROMPT, _v1.build_judge_prompt(item)
    aspects = item.get("aspects") or []
    lines = [f"{i}. {a.get('term', '')} [{a.get('category', '-')}] -> {a.get('polarity', '')}"
             for i, a in enumerate(aspects, 1)]
    return CHECKLIST_SYSTEM, CHECKLIST_USER.format(
        business_category=item.get("business_category", "Boshqa"), text=item["text"],
        n=len(aspects), annotation="\n".join(lines) or "(no aspects)")


# ----------------------------------------------------------------------------
# Judges
# ----------------------------------------------------------------------------
class Judge:
    """One OpenAI-compatible chat endpoint, configured by a judges.json entry."""

    def __init__(self, spec):
        self.spec = spec
        self.name = spec["name"]
        self.model = spec["model"]
        self.base = spec.get("api_base", "https://api.openai.com/v1").rstrip("/")
        self.key_env = spec.get("api_key_env", "OPENAI_API_KEY")
        self.key = os.environ.get(self.key_env, "")
        self.client = httpx.Client(timeout=spec.get("timeout", 120.0))

    def chat(self, system, user):
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        # OpenAI's own API wants max_completion_tokens (its reasoning models
        # reject max_tokens); other OpenAI-compatible servers want max_tokens.
        limit = "max_completion_tokens" if "api.openai.com" in self.base else "max_tokens"
        body[limit] = self.spec.get("max_tokens", 1000)
        if self.spec.get("temperature") is not None:     # null for models that refuse it
            body["temperature"] = self.spec["temperature"]
        if self.spec.get("json_mode", True):
            body["response_format"] = {"type": "json_object"}
        body.update(self.spec.get("extra", {}))
        resp = self.client.post(f"{self.base}/chat/completions", json=body,
                                headers={"Authorization": f"Bearer {self.key}"})
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"].get("content") or "",
                data.get("usage") or {}, data.get("model", self.model))

    def cost(self, usage_in, usage_out):
        p_in, p_out = self.spec.get("usd_per_1m_input"), self.spec.get("usd_per_1m_output")
        if p_in is None or p_out is None:
            return None
        return round((usage_in * p_in + usage_out * p_out) / 1e6, 4)


def call_with_retry(judge, system, user, attempts=10):
    for attempt in range(attempts):
        wait = min(60, 2 ** attempt * 2)
        try:
            t0 = time.time()
            text, usage, served = judge.chat(system, user)
            return text, usage, served, time.time() - t0
        except httpx.HTTPStatusError as e:
            code, body = e.response.status_code, e.response.text
            # a wrong key, model name or request, or an account without credit
            # (OpenAI sends that as a 429 too): retrying cannot help
            if (code != 429 and code < 500) or "insufficient_quota" in body:
                raise RuntimeError(f"{judge.name}: HTTP {code}: {body[:300]}") from e
            hdr = e.response.headers
            try:                                  # rate limited: wait as long as asked
                if hdr.get("retry-after-ms"):
                    wait = float(hdr["retry-after-ms"]) / 1000 + 0.5
                elif hdr.get("retry-after"):
                    wait = float(hdr["retry-after"]) + 0.5
            except ValueError:
                pass
        except httpx.TransportError:
            pass
        time.sleep(wait)
    raise RuntimeError(f"{judge.name}: no response after {attempts} attempts")


def load_raw(path):
    """Latest scored record per review from a judge's raw log."""
    done = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("scores"):
                done[rec["review_id"]] = rec
    return done


def run_judge(judge, items, rubric, raw_path, workers):
    done = load_raw(raw_path)
    todo = [it for it in items if it["review_id"] not in done]
    lock = threading.Lock()

    def work(item):
        system, user = build_prompt(item, rubric)
        rec = {"review_id": item["review_id"], "judge": judge.name, "model": judge.model,
               "rubric": rubric, "scores": None}
        for attempt in range(1, 4):             # re-ask when the reply does not parse
            text, usage, served, secs = call_with_retry(judge, system, user)
            parsed = _v1.parse_judge_response(text) if text else None
            rec.update(response=text, usage=usage, served_model=served, seconds=round(secs, 2),
                       attempts=attempt, time=datetime.now().isoformat(timespec="seconds"))
            if parsed:
                rec["scores"] = {d: parsed[d] for d in DIMS}
                rec["evidence"] = {k: parsed[k] for k in EVIDENCE if k in parsed}
                break
        with lock, open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    failed = []
    if todo:
        ex = ThreadPoolExecutor(max_workers=workers)
        futures = [ex.submit(work, it) for it in todo]
        try:
            for fut in tqdm(as_completed(futures), total=len(futures), desc=judge.name):
                rec = fut.result()
                if rec["scores"]:
                    done[rec["review_id"]] = rec
                else:
                    failed.append(rec["review_id"])
        except Exception:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown()
    return done, failed


# ----------------------------------------------------------------------------
# Agreement statistics
# ----------------------------------------------------------------------------
def tier_stats(judge_overall, human_overall):
    jt, ht = [tier(x) for x in judge_overall], [tier(x) for x in human_overall]
    inc = [h for j, h in zip(jt, ht) if j == "include"]
    return {"agreement": round(float(np.mean([a == b for a, b in zip(jt, ht)])), 3),
            "kappa": round(cohen_kappa(jt, ht, TIERS), 3),
            "kappa_linear": round(cohen_kappa(jt, ht, TIERS, "linear"), 3),
            "include_precision": round(inc.count("include") / len(inc), 3) if inc else None,
            "include_recall": round(sum(j == h == "include" for j, h in zip(jt, ht))
                                    / max(1, ht.count("include")), 3),
            "judge_include_human_exclude": inc.count("exclude"),
            "judge_tiers": dict(Counter(jt)), "human_tiers": dict(Counter(ht))}


def agreement(scores, human, ids):
    out = {"n": len(ids)}
    for d in DIMS:
        j = np.array([scores[i][d] for i in ids], dtype=float)
        h = np.array([human[i][d] for i in ids], dtype=float)
        out[d] = {"spearman": round(float(spearmanr(j, h).correlation), 3),
                  "mae": round(float(np.abs(j - h).mean()), 3),
                  "judge_mean": round(float(j.mean()), 2), "human_mean": round(float(h.mean()), 2)}
    out["tiers"] = tier_stats([scores[i]["overall"] for i in ids], [human[i]["overall"] for i in ids])
    return out


def bootstrap(systems, human, ids, baseline, n_boot=2000, seed=42):
    """95% CIs for overall rho and tier kappa; paired tests against `baseline`."""
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    h_all = np.array([human[i]["overall"] for i in ids], dtype=float)
    draws = {}
    for name, sc in systems.items():
        j_all = np.array([sc[i]["overall"] for i in ids], dtype=float)
        rho, kap = [], []
        with np.errstate(all="ignore"):
            for idx in resamples:
                j, h = j_all[idx], h_all[idx]
                rho.append(spearmanr(j, h).correlation)
                kap.append(cohen_kappa([tier(x) for x in j], [tier(x) for x in h], TIERS))
        draws[name] = {"spearman_overall": np.array(rho), "kappa": np.array(kap)}
    out = {}
    for name, d in draws.items():
        entry = {}
        for stat, vals in d.items():
            entry[stat] = {"ci95": [round(float(np.nanpercentile(vals, q)), 3) for q in (2.5, 97.5)]}
            if baseline in draws and name != baseline:
                diff = vals - draws[baseline][stat]
                diff = diff[~np.isnan(diff)]
                p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
                entry[stat][f"vs_{baseline}"] = {
                    "ci95": [round(float(np.percentile(diff, q)), 3) for q in (2.5, 97.5)],
                    "p_value": round(min(1.0, float(p)), 4)}
        out[name] = entry
    return out


def inter_judge(per_judge, ids):
    names = [n for n in per_judge if all(i in per_judge[n] for i in ids)]
    if len(names) < 2:
        return None
    pairs = {f"{a} | {b}": round(float(spearmanr([per_judge[a][i]["overall"] for i in ids],
                                                  [per_judge[b][i]["overall"] for i in ids]).correlation), 3)
             for a, b in combinations(names, 2)}
    alpha = {d: _hv.krippendorff_alpha([[per_judge[n][i][d] for n in names] for i in ids])
             for d in DIMS}
    return {"judges": names, "n": len(ids), "spearman_overall": pairs, "krippendorff_alpha": alpha}


def panel_mean(per_judge, ids):
    out = {}
    for i in ids:
        got = [per_judge[n][i] for n in per_judge if i in per_judge[n]]
        if got:
            out[i] = {d: round(float(np.mean([g[d] for g in got])), 2) for d in DIMS}
            out[i]["n_judges"] = len(got)
    return out


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def load_human(hv_dir):
    raters = [{r["review_id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
              for p in sorted((hv_dir / "returned").glob("rubric_*.csv"))]
    ids = [i for i in raters[0] if all(i in r for r in raters)]
    return {i: {d: float(np.mean([float(r[i][f"{d}_1_5"]) for r in raters])) for d in DIMS}
            for i in ids}


def calibration_items(hv_dir):
    """The human-rated reviews, with exactly the annotations the humans rated."""
    sample = list(csv.DictReader(open(hv_dir / "sample_150.csv", encoding="utf-8")))
    meta = {r["review_id"]: r for r in json.load(open(PUBLISHED_JUDGE, encoding="utf-8"))}
    return [{"review_id": r["review_id"], "text": r["text"],
             "business_category": r["business_category"],
             "business_name": meta.get(r["review_id"], {}).get("business_name", "Unknown"),
             "user_rating": meta.get(r["review_id"], {}).get("user_rating", "N/A"),
             "aspects": json.loads(r["model_aspects_json"])} for r in sample]


def imported_scores(spec):
    rows = json.load(open(ROOT / spec["file"], encoding="utf-8"))
    return {r["review_id"]: {d: r["judge_scores"][d] for d in DIMS}
            for r in rows if r.get("judge_scores")}


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def select_judges(cfg, names, need_keys=True):
    live = []
    for spec in cfg["judges"]:
        if names and spec["name"] not in names:
            continue
        if not names and not spec.get("enabled", True):
            continue
        judge = Judge(spec)
        if need_keys and not judge.key:
            print(f"SKIP {judge.name}: ${judge.key_env} is not set")
            continue
        live.append(judge)
    return live


def dry_run(judges, items, rubric):
    system, user = build_prompt(items[0], rubric)
    print(f"--- {rubric} prompt for {items[0]['review_id']} ---\n[system]\n{system}\n[user]\n{user}\n---")
    chars = sum(len("".join(build_prompt(it, rubric))) for it in items)
    tok_in, tok_out = chars / 3, EST_OUTPUT_TOKENS[rubric] * len(items)   # ~3 chars/token for Uzbek
    for j in judges:
        c = j.cost(tok_in, tok_out)
        print(f"{j.name:28s} {len(items)} calls  ~{tok_in / 1e6:.2f}M in + ~{tok_out / 1e6:.2f}M out tokens"
              + (f"  ~${c:.2f} at the judges.json prices" if c is not None else "  (no prices in judges.json)"))


def run_all(judges, items, rubric, out, workers):
    (out / "raw").mkdir(parents=True, exist_ok=True)
    per_judge, logs = {}, {}
    for judge in judges:
        raw_path = out / "raw" / f"{judge.name.replace('/', '_')}__{rubric}.jsonl"
        done, failed = run_judge(judge, items, rubric, raw_path, workers)
        recs = [done[i] for i in done]
        tok_in = sum(r["usage"].get("prompt_tokens", 0) for r in recs)
        tok_out = sum(r["usage"].get("completion_tokens", 0) for r in recs)
        per_judge[judge.name] = {i: r["scores"] for i, r in done.items()}
        logs[judge.name] = {"model": judge.model, "api_base": judge.base, "rubric": rubric,
                            "served_models": dict(Counter(r.get("served_model") for r in recs)),
                            "scored": len(done), "unparsed": failed,
                            "tokens_in": tok_in, "tokens_out": tok_out,
                            "usd": judge.cost(tok_in, tok_out), "raw_log": str(raw_path)}
    return per_judge, logs


def cmd_calibrate(args, cfg, judges):
    human = load_human(args.human_dir)
    items = [it for it in calibration_items(args.human_dir) if it["review_id"] in human]
    if args.limit:
        items = items[: args.limit]
    if args.dry_run:
        return dry_run(judges, items, args.rubric)
    per_judge, logs = run_all(judges, items, args.rubric, args.out, args.workers)
    systems = {f"{n} [{args.rubric}]": s for n, s in per_judge.items()}
    for spec in cfg.get("imported", []):
        systems[spec["name"]] = imported_scores(spec)
    if len(per_judge) > 1:
        systems[f"panel mean [{args.rubric}]"] = panel_mean(per_judge, [it["review_id"] for it in items])
    ids = [it["review_id"] for it in items if all(it["review_id"] in s for s in systems.values())]
    baseline = (cfg.get("imported") or [{}])[0].get("name")
    report = {"rubric": args.rubric, "n_reviews": len(ids),
              "human": "mean of the rubric_*.csv raters",
              "agreement": {n: agreement(s, human, ids) for n, s in systems.items()},
              "bootstrap": bootstrap(systems, human, ids, baseline, args.n_boot),
              "inter_judge": inter_judge(per_judge, ids), "judges": logs}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "calibration.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    print(f"\n{'judge':42s} rho_ovr  kappa  k_lin  agree  inc_prec")
    for n, a in report["agreement"].items():
        t = a["tiers"]
        print(f"{n:42s} {a['overall']['spearman']:6.3f} {t['kappa']:6.3f} {t['kappa_linear']:6.3f} "
              f"{t['agreement']:6.3f} {t['include_precision']}")
    print(f"Saved -> {args.out / 'calibration.json'}")


def cmd_score(args, cfg, judges):
    items = json.load(open(args.annotations, encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]
    if args.dry_run:
        return dry_run(judges, items, args.rubric)
    per_judge, logs = run_all(judges, items, args.rubric, args.out, args.workers)
    ids = [it["review_id"] for it in items]
    panel = panel_mean(per_judge, ids)
    results = []
    for it in items:
        if it["review_id"] not in panel:
            continue
        p = panel[it["review_id"]]
        results.append({k: it.get(k) for k in ("review_id", "text", "business_name",
                                                "business_category", "user_rating", "aspects")}
                       | {"num_aspects": len(it.get("aspects") or []),
                          "judge_scores": {d: p[d] for d in DIMS}, "n_judges": p["n_judges"],
                          "judge_panel": {n: s[it["review_id"]] for n, s in per_judge.items()
                                          if it["review_id"] in s},
                          "judge_rubric": args.rubric})
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "judge_results.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    tiers = Counter(tier(r["judge_scores"]["overall"]) for r in results)
    summary = {"annotations": str(args.annotations), "rubric": args.rubric, "n_items": len(items),
               "n_scored": len(results), "unscored": [i for i in ids if i not in panel],
               "panel_tiers": dict(tiers),
               "panel_include_rate": round(tiers["include"] / max(1, len(results)), 4),
               "panel_means": {d: round(float(np.mean([r["judge_scores"][d] for r in results])), 3)
                               for d in DIMS} if results else None,
               "inter_judge": inter_judge(per_judge, [i for i in ids if i in panel]),
               "domains": _robust.domain_cis(out_path) if results else None,
               "judges": logs}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"Scored {len(results)}/{len(items)}; panel tiers {dict(tiers)}")
    print(f"Saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["calibrate", "score"])
    ap.add_argument("--judges", type=Path, default=ROOT / "experiments" / "v3" / "judges.json")
    ap.add_argument("--only", nargs="*", default=None, help="judge names to run (default: enabled ones)")
    ap.add_argument("--rubric", choices=["v1", "checklist"], default="checklist")
    ap.add_argument("--annotations", type=Path, help="score: annotations JSON (list of reviews)")
    ap.add_argument("--human-dir", type=Path, default=HV_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8, help="parallel requests per judge")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true", help="no API calls")
    ap.add_argument("--smoke", action="store_true", help="one real call per judge, then stop")
    ap.add_argument("--imported-only", action="store_true",
                    help="calibrate: report only the imported judges, with no API calls")
    args = ap.parse_args()

    cfg = json.load(open(args.judges, encoding="utf-8"))
    judges = select_judges(cfg, args.only, need_keys=not args.dry_run)
    if args.smoke:
        item = calibration_items(args.human_dir)[0]
        for j in judges:
            text, usage, served, secs = call_with_retry(j, *build_prompt(item, args.rubric))
            print(f"{j.name} ({served}, {secs:.1f}s, {usage}):\n  parsed -> "
                  f"{_v1.parse_judge_response(text)}")
        return
    if args.command == "score" and not args.annotations:
        ap.error("score needs --annotations")
    if not judges and not args.dry_run and not (args.command == "calibrate" and args.imported_only):
        ap.error("no judge can run: export the key variable each judge names in judges.json, "
                 "e.g. export OPENAI_API_KEY=sk-...")
    {"calibrate": cmd_calibrate, "score": cmd_score}[args.command](args, cfg, judges)


if __name__ == "__main__":
    main()
