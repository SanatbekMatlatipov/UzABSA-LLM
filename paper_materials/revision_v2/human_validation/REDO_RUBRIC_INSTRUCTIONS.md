# ⚠️ Rubric re-annotation needed (annotator: Sanatbek) — ~2–3 hours

## Why

Your returned `rubric_Sanatbek.csv` rates essentially everything 5:

| dimension | your ratings |
|---|---|
| completeness | 145×"5", 3×"3", 2×"1" |
| accuracy | 145×"5", 5×"4" |
| sentiment | **150×"5"** |
| relevance | **150×"5"** |
| overall | 145×"5", 5×"4" |

With zero variance, inter-annotator agreement is mathematically undefined/zero
(weighted κ ≈ 0.0–0.08, Krippendorff's α negative against Jaloliddin's ratings, which show a
realistic spread). Publishing these numbers would tell reviewers the human validation
*failed*. It also isn't credible on its face: the LLM judge found mean completeness 3.32 and
your co-annotator rated 44/150 reviews below 4 overall — the model demonstrably misses
aspects on many of these reviews.

The current manuscript therefore reports the judge calibration **against Jaloliddin only**
("expert calibration study") and makes **no IAA claim**. That's honest but weaker than what
the reviewers asked for. A genuine second set of ratings upgrades it to a real IAA study.

## What to do (one task, ~2–3 h)

1. Open a FRESH copy of the template (don't edit your old file):
   `paper_materials/revision_v2/human_validation/rubric_template.csv`
2. For each of the 150 rows, rate the model's predicted aspects 1–5 per dimension,
   **using the full scale**. Ask yourself for each review:
   - Did the model miss any opinion expressed in the text? → completeness < 5
   - Is any extracted "term" not really an aspect (e.g., a verb, a greeting)? → accuracy < 5
   - Is any polarity wrong? → sentiment < 5
   - Is the category (ovqat/xizmat/…) appropriate, or a lazy "boshqalar"? → relevance < 5
   - Honest overall impression → overall
   A useful anchor: if you'd include the annotation in a published dataset unchanged, overall
   ≥ 4; if it needs fixes, 3; if it's mostly wrong, ≤ 2.
3. Do NOT look at Jaloliddin's file or the judge scores while rating (independence is the
   whole point).
4. Save as `paper_materials/revision_v2/human_validation/returned/rubric_Sanatbek_v2.csv`
   (UTF-8). Then tell Claude — I'll rename/swap it in, re-run
   `python scripts/analyze_human_validation.py`, and add the IAA numbers (κ, α) to the
   Human Validation subsection.

## NOT required (already handled)

- Jaloliddin's rubric and gold files are fine and already used in the paper.
- Your `gold_Sanatbek.csv` is usable as the second (sparser) gold annotation; the paper
  discloses the convention divergence honestly. Optional improvement for a future pass:
  both annotators agree on span conventions (surface forms from the text, annotate mentions
  not referents) and re-check — this would raise the inter-human ceiling and likely the
  model-vs-human scores too.
