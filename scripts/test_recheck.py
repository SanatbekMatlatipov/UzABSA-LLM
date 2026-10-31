#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: re-check the gold labels of the 609-review test set
# =============================================================================
"""
Builds the workbooks for a blind re-check of the test-set gold, merges the
two annotators' answers, and applies the adjudicated result. The step-by-step
protocol is experiments/v3/test_recheck/INSTRUCTIONS.md.

  build    choose the reviews to re-check: every review where the gold and
           the five systems of Table 2 disagree, plus a random control sample
           of reviews where they agree. Each review lists its gold terms and
           every term that at least two systems predicted, shuffled and
           unlabelled, so annotators cannot tell where a term came from.
  merge    validate both workbooks, report agreement (Cohen's kappa), and
           write an adjudication workbook with the rows they disagree on.
  apply    build the cleaned test set from the agreed and adjudicated
           answers, log every change, and score the five systems on the
           original and the cleaned gold.
  rescore  score any predictions file (dump_llm_preds.py format) on both.

Usage:
    python scripts/test_recheck.py build
    python scripts/test_recheck.py merge
    python scripts/test_recheck.py apply
    python scripts/test_recheck.py rescore --preds my_preds.jsonl

Author: UzABSA Team
License: MIT
"""

import argparse
import csv
import importlib.util as ilu
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
_ra = ilu.spec_from_file_location("uzabsa_robustness", ROOT / "scripts" / "robustness_analysis.py")
_robust = ilu.module_from_spec(_ra); _ra.loader.exec_module(_robust)
load_jsonl, norm, canonical, cohen_kappa = _robust.load_jsonl, _robust.norm, _robust.canonical, _robust.cohen_kappa

DIR = ROOT / "experiments" / "v3" / "test_recheck"
R2 = ROOT / "paper_materials" / "revision_v2"
# the five systems of Table 2, as in experiments/v3/run_v3.py
SYSTEMS = {"Qwen 2.5-7B": R2 / "results_v2" / "qwen2.5-7b_preds.jsonl",
           "Llama 3.1-8B": R2 / "results_v2" / "llama3.1-8b_preds.jsonl",
           "DeepSeek-R1-7B": R2 / "results_v2" / "deepseek-7b_preds.jsonl",
           "TahrirchiBERT": R2 / "baselines_v2" / "tahrirchi-bert" / "preds_per_example.jsonl",
           "BERTbek": R2 / "baselines_v2" / "bertbek" / "preds_per_example.jsonl"}
POLARITIES = ["positive", "negative", "neutral"]
HEADERS = ["row_id", "review", "candidate_term", "is_aspect", "polarity", "corrected_term", "comment"]
ADD_LABEL = "+ aspects missing above (write: term = polarity; term = polarity)"
NOTHING = {"", "-", "none", "no", "yo'q", "yoq", "нет"}
APOS = "'‘’ʻʼ`´"
METRICS = ["ate_exact_f1", "ate_partial_f1", "pair_f1", "sentiment_accuracy"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def well_formed(term, text):
    """The term occurs in the text as whole words and starts and ends with a letter or digit."""
    t = " ".join(norm(term).split())
    low = " ".join(text.lower().split())
    return bool(t) and t[0].isalnum() and t[-1].isalnum() and bool(
        re.search(r"(?<![^\W_])" + re.escape(t) + r"(?![^\W_])", low))


def find_in_text(term, text):
    """The exact substring of `text` an annotator meant, ignoring case, spacing and apostrophe style."""
    term = " ".join(str(term).split())
    if not term:
        return None
    pat = "".join(f"[{re.escape(APOS)}]" if c in APOS else r"\s+" if c == " " else re.escape(c) for c in term)
    m = re.search(pat, text, re.I)
    return m.group(0) if m else None


def load_systems():
    rows = {name: load_jsonl(path) for name, path in SYSTEMS.items()}
    ref = next(iter(rows.values()))
    for name, rs in rows.items():
        assert len(rs) == len(ref) and all(a["text"] == b["text"] and a["gold_aspects"] == b["gold_aspects"]
                                           for a, b in zip(rs, ref)), f"{name} is not aligned with the test set"
    return rows, ref


def selection():
    path = DIR / "selection.json"
    if not path.exists():
        sys.exit(f"{path} is missing: run `python scripts/test_recheck.py build` first")
    return json.load(open(path, encoding="utf-8"))


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
def disputes(ref, rows, i):
    """Why review i needs a second look, and its candidate terms."""
    text = ref[i]["text"]
    gold = {}
    for a in ref[i]["gold_aspects"]:
        if a.get("term"):
            gold.setdefault(norm(a["term"]), (a["term"], norm(a.get("polarity", ""))))
    votes, surfaces, polv = Counter(), defaultdict(Counter), defaultdict(Counter)
    for rs in rows.values():
        mine = {}
        for a in rs[i]["pred_aspects"]:
            if a.get("term"):
                mine.setdefault(norm(a["term"]), (a["term"], norm(a.get("polarity", ""))))
        for t, (surface, pol) in mine.items():
            votes[t] += 1
            surfaces[t][surface] += 1
            polv[t][pol] += 1
    reasons = []
    for t, (surface, pol) in gold.items():
        if not well_formed(surface, text):
            reasons.append(("malformed_gold_span", surface))
        elif votes[t] <= 1:
            reasons.append(("gold_term_at_most_1_system_predicts", surface))
        elif polv[t].most_common(1)[0][1] >= 3 and polv[t].most_common(1)[0][0] != pol:
            reasons.append(("gold_polarity_against_3_systems", surface))
    for t, v in votes.items():
        if t not in gold and v >= 3:
            reasons.append(("non_gold_term_3_systems_predict", surfaces[t].most_common(1)[0][0]))
    cands = [{"term": s, "in_gold": True, "gold_polarity": p, "votes": votes[t]} for t, (s, p) in gold.items()]
    cands += [{"term": surfaces[t].most_common(1)[0][0], "in_gold": False, "gold_polarity": None, "votes": v}
              for t, v in votes.items() if t not in gold and v >= 2]
    return reasons, cands


def write_workbook(path, reviews, adjudication=None):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "items"
    headers = HEADERS if adjudication is None else adjudication
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(bold=True)
    yn = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    pol = DataValidation(type="list", formula1='"positive,negative,neutral"', allow_blank=True)
    ws.add_data_validation(yn)
    ws.add_data_validation(pol)
    col = {h: i + 1 for i, h in enumerate(headers)}
    yn_col = "is_aspect" if adjudication is None else "final_is_aspect"
    pol_col = "polarity" if adjudication is None else "final_polarity"
    shade = PatternFill("solid", fgColor="EEF2F7")
    for k, rev in enumerate(reviews):
        first = ws.max_row + 1
        for row in rev["rows"]:
            ws.append([row.get(h, "") for h in headers])
            r = ws.max_row
            if not row["row_id"].endswith("-add"):
                yn.add(ws.cell(r, col[yn_col]))
                pol.add(ws.cell(r, col[pol_col]))
            else:
                ws.cell(r, col["candidate_term"]).font = Font(italic=True, color="555555")
            for c in ws[r]:
                c.alignment = Alignment(wrap_text=True, vertical="top")
                if k % 2:
                    c.fill = shade
        if ws.max_row > first:
            ws.merge_cells(start_row=first, end_row=ws.max_row, start_column=col["review"], end_column=col["review"])
    widths = {"row_id": 9, "review": 60, "candidate_term": 26, "is_aspect": 10, "polarity": 11,
              "corrected_term": 26, "comment": 28}
    for h, i in col.items():
        ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(h, 14)
    ws.freeze_panes = "C2"

    how = wb.create_sheet("how_to")
    for line in (HOW_TO if adjudication is None else HOW_TO_ADJUDICATE).strip().splitlines():
        how.append([line])
    how.column_dimensions["A"].width = 120
    wb.save(path)


HOW_TO = """
Avval GUIDELINES_uz.md ni o'qing. Ishni mustaqil bajaring: boshqa annotator bilan maslahatlashmang.
Har bir nomzod (candidate_term) qatorida:
  is_aspect       yes = bu so'z sharhdagi aspekt termini; no = emas
  polarity        positive / negative / neutral (faqat is_aspect = yes bo'lsa)
  corrected_term  faqat narsa to'g'ri, lekin chegarasi noto'g'ri bo'lsa: to'g'ri so'zlarni sharhdan ko'chiring
Har bir sharhning oxirgi qatori ('+ aspects missing above'): ro'yxatda yo'q aspektlarni yozing,
  termin = polarity; termin = polarity   (terminni sharhdan ko'chiring). Hech narsa yetishmasa bo'sh qoldiring.
row_id, review va candidate_term ustunlarini o'zgartirmang, qatorlarni saralamang va o'chirmang.
"""
HOW_TO_ADJUDICATE = """
Bu yerda faqat ikki annotator kelisha olmagan qatorlar bor (A_* va B_* ustunlari).
Har birini GUIDELINES_uz.md bo'yicha hal qiling: final_is_aspect, final_polarity va, chegara o'zgarsa,
final_corrected_term ni to'ldiring. '+ aspects missing' qatorlarida yakuniy ro'yxatni final_corrected_term ga
yozing (termin = polarity; termin = polarity) yoki hech narsa yetishmasa bo'sh qoldiring.
"""


def cmd_build(args):
    rows, ref = load_systems()
    flagged, clean = {}, []
    for i in range(len(ref)):
        reasons, cands = disputes(ref, rows, i)
        if reasons:
            flagged[i] = (reasons, cands)
        else:
            clean.append(i)
    rng = random.Random(args.seed)
    control = sorted(rng.sample(clean, min(args.control, len(clean))))
    order = sorted(flagged) + control
    rng.shuffle(order)

    sel, reviews = {}, []
    for k, i in enumerate(order, 1):
        rid = f"R{k:03d}"
        reasons, cands = flagged.get(i) or disputes(ref, rows, i)
        rng.shuffle(cands)
        for j, c in enumerate(cands, 1):
            c["row_id"] = f"{rid}-{j}"
        sel[rid] = {"idx": i, "text": ref[i]["text"], "control": i not in flagged,
                    "reasons": reasons, "candidates": cands}
        reviews.append({"rows": [{"row_id": c["row_id"], "review": ref[i]["text"], "candidate_term": c["term"]}
                                 for c in cands]
                        + [{"row_id": f"{rid}-add", "review": ref[i]["text"], "candidate_term": ADD_LABEL}]})

    DIR.mkdir(parents=True, exist_ok=True)
    for name in args.annotators:
        path = DIR / f"annotator_{name}.xlsx"
        if path.exists() and not args.force:
            sys.exit(f"{path} exists (it may hold answers); pass --force to overwrite")
        write_workbook(path, reviews)
    meta = {"seed": args.seed, "systems": {n: str(p.relative_to(ROOT)) for n, p in SYSTEMS.items()},
            "rule": "flag a review if a gold span is malformed, a gold term is predicted by <= 1 system, "
                    ">= 3 systems agree on another polarity, or a non-gold term is predicted by >= 3 systems; "
                    "candidates = gold terms + terms predicted by >= 2 systems",
            "n_flagged": len(flagged), "n_control": len(control),
            "n_candidate_rows": sum(len(s["candidates"]) for s in sel.values())}
    (DIR / "selection.json").write_text(json.dumps({"meta": meta, "reviews": sel}, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"{len(flagged)} flagged + {len(control)} control reviews, {meta['n_candidate_rows']} candidate rows")
    print("reasons:", dict(Counter(code for s in sel.values() for code, _ in s["reasons"])))
    print("workbooks:", ", ".join(str(DIR / f"annotator_{n}.xlsx") for n in args.annotators))


# ----------------------------------------------------------------------------
# reading answers
# ----------------------------------------------------------------------------
def read_items(path):
    from openpyxl import load_workbook
    ws = load_workbook(path, data_only=True)["items"]
    it = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h else "" for h in next(it)]
    out = {}
    for row in it:
        rec = {h: ("" if v is None else str(v).strip()) for h, v in zip(header, row)}
        if rec.get("row_id"):
            out[rec["row_id"]] = rec
    return out


def decide(rec, cand, text, prefix=""):
    """(decision, error) for one candidate row; decision = ('no',) or ('yes', polarity, term)."""
    yn = rec.get(prefix + "is_aspect", "").lower()
    if yn not in ("yes", "no"):
        return None, "is_aspect must be yes or no"
    if yn == "no":
        return ("no",), None
    pol = rec.get(prefix + "polarity", "").lower()
    if pol not in POLARITIES:
        return None, "polarity must be positive, negative or neutral"
    term = cand
    fix = rec.get(prefix + "corrected_term", "")
    if fix:
        term = find_in_text(fix, text)
        if term is None:
            return None, f"corrected_term '{fix}' does not occur in the review"
    return ("yes", pol, term), None


def parse_added(cell, text):
    """'term = polarity; term = polarity' -> ({norm(term): (term, polarity)}, errors)."""
    out, errs = {}, []
    if cell.strip().lower() in NOTHING:
        return out, errs
    for part in re.split(r"[;\n]+", cell):
        if not part.strip():
            continue
        if "=" not in part:
            errs.append(f"'{part.strip()}': write term = polarity")
            continue
        t, p = (x.strip() for x in part.rsplit("=", 1))
        exact = find_in_text(t, text)
        if exact is None:
            errs.append(f"'{t}' does not occur in the review")
        elif p.lower() not in POLARITIES:
            errs.append(f"'{t}': polarity must be positive, negative or neutral")
        else:
            out[norm(exact)] = (exact, p.lower())
    return out, errs


def key(decision):
    return decision if decision[0] == "no" else ("yes", decision[1], norm(decision[2]))


def collect(path, sel):
    """All decisions of one workbook, keyed by row_id, plus validation errors."""
    if not path.exists():
        sys.exit(f"{path} is missing")
    answers, dec, errors = read_items(path), {}, []
    for rid, s in sel.items():
        for c in s["candidates"]:
            d, e = decide(answers.get(c["row_id"], {}), c["term"], s["text"])
            if e:
                errors.append(f"{c['row_id']} ({c['term']}): {e}")
            dec[c["row_id"]] = d
        added, errs = parse_added(answers.get(f"{rid}-add", {}).get("corrected_term", ""), s["text"])
        errors += [f"{rid}-add: {e}" for e in errs]
        dec[f"{rid}-add"] = added
    return dec, errors


def fmt(decision):
    """A decision as workbook cells: (is_aspect, polarity, term), or the 'term = polarity' list."""
    if isinstance(decision, dict):
        return "; ".join(f"{t} = {p}" for t, p in decision.values())
    return ("no", "", "") if decision[0] == "no" else decision


# ----------------------------------------------------------------------------
# merge
# ----------------------------------------------------------------------------
def cmd_merge(args):
    sel = selection()["reviews"]
    a_name, b_name = args.annotators
    dec, bad = {}, False
    for name in (a_name, b_name):
        dec[name], errors = collect(DIR / f"annotator_{name}.xlsx", sel)
        if errors:
            bad = True
            (DIR / f"errors_{name}.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
            print(f"annotator {name}: {len(errors)} rows to fix, listed in {DIR / f'errors_{name}.txt'}; first ones:")
            print("  " + "\n  ".join(errors[:10]))
    if bad:
        sys.exit(1)
    A, B = dec[a_name], dec[b_name]
    cands = [c["row_id"] for s in sel.values() for c in s["candidates"]]
    for name in (a_name, b_name):
        # an annotator who gives (almost) the same answer everywhere did not really check
        yes = np.mean([dec[name][r][0] == "yes" for r in cands])
        pols = Counter(dec[name][r][1] for r in cands if dec[name][r][0] == "yes")
        top = pols.most_common(1)[0][1] / max(1, sum(pols.values())) if pols else 0
        if not 0.05 <= yes <= 0.95 or top > 0.95:
            print(f"WARNING: annotator {name} answered yes on {yes:.0%} of the rows and gave one polarity to "
                  f"{top:.0%} of the aspects; check the workbook before going on")
    both_yes = [r for r in cands if A[r][0] == B[r][0] == "yes"]
    agree_review = np.mean([all(key(A[c["row_id"]]) == key(B[c["row_id"]]) for c in s["candidates"])
                            and {t: p for t, (_, p) in A[f"{rid}-add"].items()}
                            == {t: p for t, (_, p) in B[f"{rid}-add"].items()} for rid, s in sel.items()])
    report = {
        "annotators": [a_name, b_name], "candidate_rows": len(cands), "reviews": len(sel),
        "is_aspect": {"agreement": round(float(np.mean([A[r][0] == B[r][0] for r in cands])), 3),
                      "kappa": round(float(cohen_kappa([A[r][0] for r in cands], [B[r][0] for r in cands],
                                                       ["yes", "no"])), 3)},
        "polarity_when_both_yes": {
            "n": len(both_yes), "agreement": round(float(np.mean([A[r][1] == B[r][1] for r in both_yes])), 3),
            "kappa": round(float(cohen_kappa([A[r][1] for r in both_yes], [B[r][1] for r in both_yes],
                                             POLARITIES)), 3)},
        "same_span_when_both_yes": round(float(np.mean([norm(A[r][2]) == norm(B[r][2]) for r in both_yes])), 3),
        "reviews_fully_identical": round(float(agree_review), 3),
    }
    rows_out = []
    for rid, s in sel.items():
        rev = []
        for c in s["candidates"]:
            r = c["row_id"]
            if key(A[r]) != key(B[r]):
                a, b = fmt(A[r]), fmt(B[r])
                rev.append({"row_id": r, "review": s["text"], "candidate_term": c["term"],
                            "A_is_aspect": a[0], "A_polarity": a[1], "A_term": a[2],
                            "B_is_aspect": b[0], "B_polarity": b[1], "B_term": b[2]})
        ra, rb = A[f"{rid}-add"], B[f"{rid}-add"]
        if {t: p for t, (_, p) in ra.items()} != {t: p for t, (_, p) in rb.items()}:
            rev.append({"row_id": f"{rid}-add", "review": s["text"], "candidate_term": ADD_LABEL,
                        "A_term": fmt(ra), "B_term": fmt(rb)})
        if rev:
            rows_out.append({"rows": rev})
    report["rows_to_adjudicate"] = sum(len(r["rows"]) for r in rows_out)
    (DIR / "agreement.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    path = DIR / "adjudication.xlsx"
    if path.exists() and not args.force:
        sys.exit(f"{path} exists (it may hold decisions); pass --force to overwrite")
    write_workbook(path, rows_out, adjudication=[
        "row_id", "review", "candidate_term", "A_is_aspect", "A_polarity", "A_term", "B_is_aspect",
        "B_polarity", "B_term", "final_is_aspect", "final_polarity", "final_corrected_term", "comment"])
    print(f"Saved -> {DIR / 'agreement.json'} and {path}")


# ----------------------------------------------------------------------------
# apply
# ----------------------------------------------------------------------------
def classify(old, new):
    """Changes between two gold lists of one review, keyed by normalised term."""
    o = {norm(a["term"]): norm(a.get("polarity", "")) for a in old if a.get("term")}
    n = {norm(a["term"]): norm(a.get("polarity", "")) for a in new if a.get("term")}
    removed, added = [t for t in o if t not in n], [t for t in n if t not in o]
    out = [("polarity changed", t, f"{o[t]} -> {n[t]}") for t in o if t in n and o[t] != n[t]]
    for t in list(removed):
        match = next((u for u in added if t in u or u in t), None)
        if match:
            out.append(("span fixed", t, match))
            removed.remove(t)
            added.remove(match)
    out += [("removed", t, "") for t in removed] + [("added", t, n[t]) for t in added]
    return out


def clean_gold(args):
    sel = selection()["reviews"]
    a_name, b_name = args.annotators
    A, ea = collect(DIR / f"annotator_{a_name}.xlsx", sel)
    B, eb = collect(DIR / f"annotator_{b_name}.xlsx", sel)
    if ea or eb:
        sys.exit("the workbooks have rows to fix: run `merge` first")
    adj = read_items(DIR / "adjudication.xlsx") if (DIR / "adjudication.xlsx").exists() else {}
    gold, final, missing = {}, {}, []
    for rid, s in sel.items():
        kept = {}
        for c in s["candidates"]:
            r = c["row_id"]
            d = A[r]
            if key(A[r]) != key(B[r]):
                d, e = decide(adj.get(r, {}), c["term"], s["text"], prefix="final_")
                if e:
                    missing.append(f"{r} ({c['term']}): {e}")
                    continue
            final[r] = d
            if d[0] == "yes":
                kept.setdefault(norm(d[2]), {"term": d[2], "polarity": d[1]})
        added = A[f"{rid}-add"]
        if {t: p for t, (_, p) in added.items()} != {t: p for t, (_, p) in B[f"{rid}-add"].items()}:
            added, errs = parse_added(adj.get(f"{rid}-add", {}).get("final_corrected_term", ""), s["text"])
            missing += [f"{rid}-add: {e}" for e in errs]
        for t, (term, p) in added.items():
            kept.setdefault(t, {"term": term, "polarity": p})
        gold[s["idx"]] = (rid, s["control"], list(kept.values()))
    if missing:
        sys.exit("adjudication.xlsx is incomplete:\n  " + "\n  ".join(missing[:20]))
    return gold, final


def outcomes_by_reason(final):
    """What the annotators decided about each kind of disputed item, and about the control reviews."""
    out = defaultdict(Counter)
    for s in selection()["reviews"].values():
        reasons = {norm(t): code for code, t in s["reasons"]}
        for c in s["candidates"]:
            t, d = norm(c["term"]), final[c["row_id"]]
            group = reasons.get(t) or ("control_" if s["control"] else "other_") + ("gold" if c["in_gold"] else "model")
            if d[0] == "no":
                verdict = "not an aspect"
            elif norm(d[2]) != t:
                verdict = "aspect, span corrected"
            elif c["in_gold"] and d[1] != c["gold_polarity"]:
                verdict = "aspect, polarity changed"
            else:
                verdict = "aspect, kept as is"
            out[group][verdict] += 1
    return {g: {"n": sum(c.values()), **dict(c.most_common())} for g, c in out.items()}


def score_table(rows_by_system, clean_by_idx):
    table = {}
    for name, rows in rows_by_system.items():
        cleaned = [dict(r, gold_aspects=clean_by_idx[i]) for i, r in enumerate(rows)]
        orig, new = canonical(rows), canonical(cleaned)
        table[name] = {"original": {m: round(orig[m], 4) for m in METRICS},
                       "cleaned": {m: round(new[m], 4) for m in METRICS}}
    return table


def cmd_apply(args):
    rows, ref = load_systems()
    gold, final = clean_gold(args)
    clean_by_idx = [gold[i][2] if i in gold else r["gold_aspects"] for i, r in enumerate(ref)]
    with open(DIR / "test_clean.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(ref):
            f.write(json.dumps({"idx": i, "text": r["text"], "gold_aspects": clean_by_idx[i],
                                "rechecked": i in gold}, ensure_ascii=False) + "\n")
    changes = []
    for i, (rid, control, new) in gold.items():
        for kind, term, detail in classify(ref[i]["gold_aspects"], new):
            changes.append({"review": rid, "idx": i, "control": control, "change": kind, "term": term,
                            "detail": detail, "text": ref[i]["text"]})
    with open(DIR / "changes.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["review", "idx", "control", "change", "term", "detail", "text"])
        w.writeheader()
        w.writerows(changes)

    ctrl = [i for i, (_, c, _) in gold.items() if c]
    ctrl_changed = len({c["idx"] for c in changes if c["control"]})
    unchecked = len(ref) - len(gold)
    summary = {
        "reviews_rechecked": len(gold), "control_reviews": len(ctrl),
        "gold_terms": {"original": sum(len(r["gold_aspects"]) for r in ref),
                       "cleaned": sum(len(g) for g in clean_by_idx)},
        "changes": dict(Counter(c["change"] for c in changes)),
        "reviews_changed": len({c["idx"] for c in changes}),
        "control_reviews_changed": ctrl_changed,
        # the control sample estimates how many of the unchecked reviews still hold an error
        "unchecked_reviews": unchecked,
        "unchecked_reviews_with_error_est": round(unchecked * ctrl_changed / max(1, len(ctrl)), 1),
        "outcomes_by_reason": outcomes_by_reason(final),
        "scores": score_table(rows, clean_by_idx),
    }
    (DIR / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    out = DIR / "preds_clean"
    out.mkdir(exist_ok=True)
    for name, rs in rows.items():
        with open(out / (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".jsonl"), "w", encoding="utf-8") as f:
            for i, r in enumerate(rs):
                f.write(json.dumps(dict(r, gold_aspects=clean_by_idx[i]), ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "scores"}, indent=1))
    print(f"\n{'system':16s} " + "  ".join(f"{m:>22s}" for m in METRICS))
    for name, s in summary["scores"].items():
        print(f"{name:16s} " + "  ".join(f"{s['original'][m]:.3f} -> {s['cleaned'][m]:.3f} ({s['cleaned'][m] - s['original'][m]:+.3f})"
                                         .rjust(22) for m in METRICS))
    print(f"\nSaved -> {DIR / 'test_clean.jsonl'}, changes.csv, summary.json, preds_clean/")


def cmd_rescore(args):
    clean = load_jsonl(DIR / "test_clean.jsonl")
    rows = load_jsonl(args.preds)
    assert len(rows) == len(clean) and all(r["text"] == c["text"] for r, c in zip(rows, clean)), \
        "the predictions are not aligned with the 609-review test set"
    s = score_table({args.preds.stem: rows}, [c["gold_aspects"] for c in clean])[args.preds.stem]
    print(json.dumps(s, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["build", "merge", "apply", "rescore"])
    ap.add_argument("--annotators", nargs=2, default=["A", "B"], help="names used in the workbook file names")
    ap.add_argument("--control", type=int, default=60, help="random reviews without disputes to include")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--preds", type=Path, help="rescore: predictions file")
    ap.add_argument("--force", action="store_true", help="overwrite existing workbooks")
    ap.add_argument("--dir", type=Path, default=DIR, help="working directory of the re-check")
    args = ap.parse_args()
    globals()["DIR"] = args.dir
    {"build": cmd_build, "merge": cmd_merge, "apply": cmd_apply, "rescore": cmd_rescore}[args.command](args)


if __name__ == "__main__":
    main()
