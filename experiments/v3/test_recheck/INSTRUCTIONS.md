# Fix 1: re-check the test-set gold, step by step

**Goal.** Correct wrong gold labels in the 609-review test set without favouring any model, then
report every system on both the original and the cleaned gold.

**Why it is needed.** On many test items the gold and the systems disagree:

| Dispute | Items |
|---|---|
| Gold term that at most one of the five systems predicts | 152 |
| Term that three or more systems predict but the gold lacks | 130 |
| Gold polarity contradicted by three or more systems | 29 |
| Malformed gold span (e.g. `r qilish u`, `bu yerda`) | 15 |

Some of these items are model errors and some are gold errors. Only people can tell which.

**What the re-check covers.**
- The 203 reviews with at least one dispute, plus 60 random reviews without disputes as a control sample.
- That makes 263 reviews and 575 candidate terms.
- Each review lists its gold terms and every term that at least two systems predicted, shuffled and
  unlabelled. Annotators cannot tell which terms are gold.

## What you need

- **Two annotators (A and B)** who read Uzbek fluently.
  - Best: two people who are not you, for example Jaloliddin and one more native speaker.
  - You then act as the adjudicator. If you must annotate yourself, let someone else adjudicate.
- **One adjudicator** for the rows on which A and B disagree.
- **Time:**
  - about 3–4 h per annotator;
  - about 1–2 h for adjudication;
  - about 30 min for you to run the scripts.
- **Software:** Excel, LibreOffice or Google Sheets. In Google Sheets, upload the `.xlsx` and download it
  again as `.xlsx`; the drop-down lists are kept.

## Step 1. Build the workbooks (done on 31 Oct 2026)

```bash
python scripts/test_recheck.py build
```

This created the following in `experiments/v3/test_recheck/`:

| File | What it is |
|---|---|
| `annotator_A.xlsx`, `annotator_B.xlsx` | Identical workbooks, one per annotator. Each has a sheet `items` and a sheet `how_to` (short help in Uzbek). |
| `selection.json` | Which test review each `R…` number is, why it was chosen, and which candidates are gold. **Never show it to annotators.** |

Do not run `build` again. The selection is fixed by `--seed 2026`, and `--force` would overwrite
workbooks that may already hold answers.

## Step 2. Brief the annotators (30 min)

1. **Send each annotator only two files:** their own workbook and `GUIDELINES_uz.md`. Do not send
   `selection.json`, the paper, or any model output.
2. **Ask both to read the guidelines and solve the six practice examples** in section 5, then compare
   with the answers there.
3. **Answer their questions about the rules.** Talk only about the guideline examples, never about
   items in the workbook.
4. **Say clearly:**
   - they work alone and do not discuss items with each other;
   - every candidate row needs `is_aspect`;
   - "+ aspects missing above" rows are for terms the list lacks.

## Step 3. Annotate (each annotator alone, about 3–4 h)

- **Return the files with their original names**, `annotator_A.xlsx` and `annotator_B.xlsx`, into
  `experiments/v3/test_recheck/`, replacing the empty ones.
- **Keep a copy of each returned file** somewhere outside the repository as a backup.

## Step 4. Merge and check agreement

```bash
python scripts/test_recheck.py merge
```

1. **If it prints `rows to fix`:** send `errors_A.txt` or `errors_B.txt` back to that annotator. They
   correct only the listed rows. Then run `merge` again.
2. **If it prints a `WARNING`** (for example "yes" on 98% of rows), the annotator did not really check.
   Look at the workbook and ask them to redo it before going on.
3. **When it succeeds**, it writes:
   - `agreement.json`, with Cohen's κ for the aspect decision (`is_aspect`) and for polarity, plus span agreement;
   - `adjudication.xlsx`, which holds only the rows on which A and B disagree.
4. **Check κ for `is_aspect`:**
   - **κ ≥ 0.6:** go on.
   - **κ < 0.6:** the guideline is unclear somewhere.
     1. Read 20 disagreeing rows and find the rule behind most of them.
     2. Make that rule clearer in `GUIDELINES_uz.md`.
     3. Let both annotators revise their own answers on rows that rule covers.
     4. Run `merge` again.
     5. In the paper, report the κ of the **first** run.

## Step 5. Adjudicate (about 1–2 h)

1. **Open `adjudication.xlsx`.** Each row shows the answers of A and B.
2. **For every row, fill in:**
   - `final_is_aspect`;
   - `final_polarity`;
   - `final_corrected_term`, only if the span changes.

   Decide by `GUIDELINES_uz.md`, not by majority or by which annotator you trust more.
3. **On "+ aspects missing" rows,** write the final list into `final_corrected_term` as
   `term = polarity; term = polarity`, or leave the cell empty.
4. **Save the file under the same name.**

## Step 6. Apply

```bash
python scripts/test_recheck.py apply
```

It writes:

| File | Content |
|---|---|
| `test_clean.jsonl` | The cleaned test set: 609 reviews, same order and texts, `rechecked` marks the 263 re-checked ones |
| `changes.csv` | Every change, with the review text: removed, added, span fixed, polarity changed. **Read it once from top to bottom** and look for anything absurd. |
| `summary.json` | Counts of changes, the control-sample estimate of errors left in the 346 unchecked reviews, and all five systems scored on the original and the cleaned gold |
| `preds_clean/*.jsonl` | Each system's predictions with the cleaned gold, for significance tests |

**Significance tests on the cleaned gold** use the existing script unchanged, for example:

```bash
python scripts/significance_test.py \
    --a experiments/v3/test_recheck/preds_clean/qwen_2_5_7b.jsonl --a-name Qwen \
    --b experiments/v3/test_recheck/preds_clean/llama_3_1_8b.jsonl --b-name Llama \
    --out experiments/v3/test_recheck/significance_qwen_vs_llama.json
```

**Any future system** (for example a v3 run) is scored on both gold versions with:

```bash
python scripts/test_recheck.py rescore --preds <its preds.jsonl>
```

## Step 7. Report it in the paper (camera-ready or journal version)

- **Keep Table 2 on the original gold,** so it stays comparable with earlier work.
- **Add the cleaned-gold scores:** as an extra column, or as a sentence in §5.
- **Template**, with the numbers to fill in from `agreement.json` and `summary.json`:

  > Two annotators re-checked the 263 test reviews on which the gold and the five systems disagree,
  > plus 60 random control reviews. They did not know which candidate terms came from the gold
  > (Cohen's κ = … for aspect decisions and … for polarity), and an adjudicator resolved the remaining
  > disagreements. The re-check changed … gold terms in … reviews: … removed, … added, … span fixes,
  > and … polarity changes. On the cleaned gold, exact ATE F1 is … for Qwen and … for Llama. Only …
  > of the 60 control reviews changed, so about … of the 346 unchecked reviews may still contain an error.

- **One honest caveat:** the reviews were chosen where systems disagree with the gold, so the cleaning
  can only fix errors the systems exposed, and it tends to raise every system's score. This is why
  you report both versions.
- **Never choose models or hyperparameters on the cleaned set.**

## Step 8. Release

- **Publish the cleaned set next to the original, never in its place:** for example as
  "UzABSA test v1.1 (re-checked)" on Zenodo or as a Hugging Face dataset revision.
- **Files to include:**
  - `test_clean.jsonl`
  - `changes.csv`
  - `GUIDELINES_uz.md`
  - `agreement.json`

## Re-use for fix 2 (cleaned training copy)

The same guideline defines the cleaned training labels. Its rules A2–A4 and S1–S6 decide which
training terms to drop or re-span (*hammasi*, *bu yerda*, truncated spans). Keep the two cleanups
consistent.
