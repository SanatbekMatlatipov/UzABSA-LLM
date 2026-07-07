#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: BIO-Polarity Dataset Builder for BERT Baselines  (P1)
# =============================================================================
"""
Builds a token-classification (BIO-polarity) dataset for the encoder baselines,
directly from the SAME ChatML splits the LLMs were trained/evaluated on
(``data/processed`` — train 5480 / validation 609, seed 42). This guarantees the
BERT baselines are comparable to the LLMs on the identical 609-example set.

Tag scheme (joint extraction + polarity):
    O, B-positive, I-positive, B-negative, I-negative,
    B-neutral, I-neutral, B-conflict, I-conflict

For each review we parse (text, gold_aspects) with the existing
``parse_chatml_example`` helper, tokenize the text on whitespace/punctuation into
"words", then locate each gold aspect *term* among those words and tag it. Terms
that cannot be localized (category-only aspects, or surface forms absent from the
text) are skipped and counted — the alignment rate is reported and logged, as it
is a caveat worth stating in the paper.

Outputs (to ``data/bio_processed/``):
    train.json, validation.json   — each a list of
        {"tokens": [...], "ner_tags": [int...], "text": str, "aspects": [{term,polarity}]}
    label_map.json                — {label: id}
    stats.json                    — counts + term-alignment rate

Runs in the project env (needs ``datasets`` to read the arrow splits). Use
``--selftest`` to exercise the conversion logic with no external dependency.

Usage:
    python scripts/prepare_bio_dataset.py --data ./data/processed --out ./data/bio_processed
    python scripts/prepare_bio_dataset.py --selftest

Author: UzABSA Team
License: MIT
"""

import argparse
import importlib.util as ilu
import json
import re
from pathlib import Path

# Reuse parse_chatml_example without triggering the heavy src/__init__.py.
_spec = ilu.spec_from_file_location(
    "uzabsa_evaluation", Path(__file__).parent.parent / "src" / "evaluation.py")
_evaluation = ilu.module_from_spec(_spec)
_spec.loader.exec_module(_evaluation)
parse_chatml_example = _evaluation.parse_chatml_example

POLARITIES = ["positive", "negative", "neutral", "conflict"]
LABELS = ["O"] + [f"{bi}-{p}" for p in POLARITIES for bi in ("B", "I")]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}

# Word tokenizer: keep alnum runs (incl. Uzbek/Cyrillic letters and apostrophes) as
# words; punctuation becomes its own token. Character offsets are preserved implicitly
# by ordering (we only need word-level BIO for the encoder).
_WORD_RE = re.compile(r"\w+(?:['’ʻ`]\w+)*|[^\w\s]", re.UNICODE)


def tokenize_words(text):
    return _WORD_RE.findall(text or "")


def _norm(s):
    return s.lower().replace("’", "'").replace("ʻ", "'").replace("`", "'").strip()


def align_term_to_words(words_norm, term_words_norm):
    """Return the (start, end) word-index span for a term, or None.

    1) exact contiguous word-subsequence match; else
    2) single-word substring match (handles Uzbek morphology, e.g. term 'ovqat'
       occurring as 'ovqatlari' in the text — matches on either-contains-other).
    """
    n, m = len(words_norm), len(term_words_norm)
    if m == 0:
        return None
    # 1) exact contiguous match
    for i in range(n - m + 1):
        if words_norm[i:i + m] == term_words_norm:
            return (i, i + m)
    # 2) morphological single-word fallback (only for single-word terms)
    if m == 1:
        t = term_words_norm[0]
        for i, w in enumerate(words_norm):
            if t and (t in w or w in t) and abs(len(t) - len(w)) <= max(3, len(t) // 2):
                return (i, i + 1)
    return None


def convert_example(chatml_text):
    """ChatML string -> {tokens, ner_tags, text, aspects, n_terms, n_aligned}."""
    text, gold = parse_chatml_example(chatml_text)
    words = tokenize_words(text)
    words_norm = [_norm(w) for w in words]
    tags = [LABEL2ID["O"]] * len(words)

    term_aspects = [a for a in gold if a.get("term")]
    n_terms = len(term_aspects)
    n_aligned = 0
    used = [False] * len(words)
    kept_aspects = []
    for a in term_aspects:
        pol = (a.get("polarity") or "neutral").lower()
        if pol not in POLARITIES:
            pol = "neutral"
        tw = [_norm(w) for w in tokenize_words(a["term"])]
        span = align_term_to_words(words_norm, tw)
        if span is None:
            continue
        s, e = span
        if any(used[s:e]):  # avoid double-tagging overlaps
            continue
        for k in range(s, e):
            used[k] = True
            tags[k] = LABEL2ID[f"{'B' if k == s else 'I'}-{pol}"]
        n_aligned += 1
        kept_aspects.append({"term": a["term"], "polarity": pol})

    return {"tokens": words, "ner_tags": tags, "text": text,
            "aspects": kept_aspects, "n_terms": n_terms, "n_aligned": n_aligned}


def decode_bio(tokens, tag_ids):
    """Inverse: BIO-polarity tag ids -> [{term, polarity}] (used at eval time)."""
    aspects, cur, cur_pol = [], [], None
    for tok, tid in zip(tokens, tag_ids):
        lab = LABELS[tid] if 0 <= tid < len(LABELS) else "O"
        if lab == "O":
            if cur:
                aspects.append({"term": " ".join(cur), "polarity": cur_pol})
                cur, cur_pol = [], None
            continue
        bi, pol = lab.split("-", 1)
        if bi == "B" or cur_pol != pol:
            if cur:
                aspects.append({"term": " ".join(cur), "polarity": cur_pol})
            cur, cur_pol = [tok], pol
        else:  # I- continuation of same polarity
            cur.append(tok)
    if cur:
        aspects.append({"term": " ".join(cur), "polarity": cur_pol})
    return aspects


def build(data_dir, out_dir):
    from datasets import load_from_disk
    ds = load_from_disk(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ["train", "validation"]:
        if split not in ds:
            continue
        rows, terms, aligned = [], 0, 0
        for ex in ds[split]:
            conv = convert_example(ex["text"])
            terms += conv.pop("n_terms")
            aligned += conv.pop("n_aligned")
            rows.append(conv)
        json.dump(rows, open(out / f"{split}.json", "w", encoding="utf-8"),
                  ensure_ascii=False)
        stats[split] = {"n_examples": len(rows), "n_gold_terms": terms,
                        "n_aligned_terms": aligned,
                        "alignment_rate": round(aligned / terms, 4) if terms else 0.0}
        print(f"{split}: {len(rows)} examples, term alignment "
              f"{aligned}/{terms} = {stats[split]['alignment_rate']:.1%}")
    json.dump(LABEL2ID, open(out / "label_map.json", "w"), indent=2)
    json.dump(stats, open(out / "stats.json", "w"), indent=2)
    print(f"Wrote BIO dataset + label_map + stats to {out}")


def selftest():
    sys_p = "<|im_start|>system\nSiz mutaxassis.<|im_end|>\n"
    def chatml(text, aspects):
        user = f'<|im_start|>user\nMatn: "{text}"<|im_end|>\n'
        asst = ('<|im_start|>assistant\n'
                + json.dumps({"aspects": aspects}, ensure_ascii=False)
                + '<|im_end|>')
        return sys_p + user + asst

    cases = [
        ("Ovqatlari juda mazali, lekin narxi qimmat.",
         [{"term": "Ovqatlari", "category": "ovqat", "polarity": "positive"},
          {"term": "narxi", "category": "narx", "polarity": "negative"}]),
        ("Ilova sekin ishlaydi",
         [{"term": "Ilova", "polarity": "negative"},
          {"category": "xizmat", "polarity": "neutral"}]),  # category-only -> skipped
        ("Xizmat yaxshi",
         [{"term": "ovqat", "polarity": "positive"}]),  # term absent -> not aligned
    ]
    ok = True
    for text, asp in cases:
        conv = convert_example(chatml(text, asp))
        rt = decode_bio(conv["tokens"], conv["ner_tags"])
        print(f"\nTEXT: {text}")
        print(f"  tokens: {conv['tokens']}")
        print(f"  tags:   {[LABELS[t] for t in conv['ner_tags']]}")
        print(f"  aligned {conv['n_aligned']}/{conv['n_terms']} terms; kept={conv['aspects']}")
        print(f"  round-trip decode: {rt}")
    # assertions
    c0 = convert_example(chatml(*cases[0]))
    assert c0["n_aligned"] == 2, "both terms should align (incl. morphological narxi->narx)"
    rt0 = decode_bio(c0["tokens"], c0["ner_tags"])
    assert {a["polarity"] for a in rt0} == {"positive", "negative"}, rt0
    c1 = convert_example(chatml(*cases[1]))
    assert c1["n_terms"] == 1 and c1["n_aligned"] == 1, "category-only aspect must be ignored"
    c2 = convert_example(chatml(*cases[2]))
    assert c2["n_aligned"] == 0, "absent term must not align"
    print("\nSELFTEST PASSED — BIO conversion + round-trip decode correct.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/processed")
    ap.add_argument("--out", default="./data/bio_processed")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        build(args.data, args.out)


if __name__ == "__main__":
    main()
