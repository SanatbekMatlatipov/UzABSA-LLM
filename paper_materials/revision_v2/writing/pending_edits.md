# Pending manuscript edits — status after 2026-08-28 BDCC pass

Target journal is now **MDPI BDCC** (the manuscript was previously built for *Computers*;
`\documentclass` fixed on 2026-08-28). `main.tex` compiles clean: 0 undefined refs,
0 overfull boxes, 20 pp.

## Still pending

1. **Aripov's email address** — `main.tex` affiliation block contains the literal placeholder
   `TODO-ARIPOV-EMAIL`. MDPI requires an email for every author. Replace before submission.

2. **LLM significance tests — BLOCKED on this machine (disk).** Running
   `scripts/dump_llm_preds.py` needs the merged fp16 models, which are **not** in
   `outputs/my_run/` (only ~440 KB of metrics per run is stored locally; the weights live on
   HF Hub at ~14 GB per model). This Mac has **2.6 GB free of 926 GB**, so neither model can be
   downloaded. To finish:
   ```
   # free ~35 GB first, then:
   python scripts/dump_llm_preds.py --model Sanatbek/UzABSA-LLM --revision qwen2.5-7b \
       --out paper_materials/revision_v2/results/qwen_preds.jsonl
   python scripts/dump_llm_preds.py --model Sanatbek/UzABSA-LLM --revision llama3.1-8b \
       --out paper_materials/revision_v2/results/llama_preds.jsonl
   python scripts/significance_test.py \
       --a paper_materials/revision_v2/results/qwen_preds.jsonl --a-name Qwen \
       --b paper_materials/revision_v2/results/llama_preds.jsonl --b-name Llama \
       --out paper_materials/revision_v2/significance/qwen_vs_llama.json
   ```
   Note `dump_llm_preds.py` prints a warning that `load_model()` may not forward `--revision`;
   if so, `git clone --branch qwen2.5-7b` the HF repo locally and pass the path.
   The paper does **not** currently over-claim: the Qwen-vs-Llama gap is already described as
   "effectively a tie" and only the encoder-pair bootstrap is reported as significance-tested.
   Eq. (8) in §Evaluation Framework is where the p-values would be cited.

3. **Zenodo upload of the v2 bundle** — bundle is built and ready at
   `paper_materials/revision_v2/zenodo_release/` (see "Zenodo release" below). Upload as a
   **New version** of record 18790639 so the Data Availability claim is true at submission.

4. **Authorship housekeeping** — author list changed 2026-08-28 to
   Matlatipov (1st, corresponding) · Djalilov (2nd) · Aripov (3rd); Rajabov and Almarashi
   removed. Confirm all three consent, and that Djalilov's ORCID `0009-0007-3089-0867` and
   affiliation ("Independent Researcher, Tashkent, Uzbekistan") are exactly as he wants them.

5. **Pre-submission email** — `writing/presubmission_email.md` predates the author change and
   the BDCC reframe; refresh the author line before sending.

6. **Confirm the special issue** — the manuscript targets SI `O9A9UWB542`, whose page is not
   machine-readable (MDPI serves 403 to automated fetches) and which is not indexed in search.
   The SI the earlier revision targeted, "AI and NLP" (`W633E7395P`), **closed 20 July 2026**.
   Verify in susy.mdpi.com that `O9A9UWB542` is open and NLP-scoped; if its framing is narrower
   than generic NLP, the Introduction's scope paragraph should be tuned to match it.

## Done

### 2026-08-28 — Item 1 (IAA) COMPLETE
The rubric redo was finished and `scripts/analyze_human_validation.py` re-run on
`human_validation/returned/` (4 files: rubric + gold from Jaloliddin and Sanatbek).
Regenerated `results/human_validation_report.json`. Numbers now **in the paper** as
`\Cref{tab:iaa}` plus an "Inter-annotator agreement" paragraph in `sec:res_human`:

| Dimension | Weighted κ | Krippendorff's α | Spearman ρ |
|---|---|---|---|
| Sentiment | 0.970 | 0.970 | 0.941 |
| Accuracy | 0.936 | 0.936 | 0.920 |
| Completeness | 0.930 | 0.930 | 0.937 |
| Overall | 0.923 | 0.922 | 0.923 |
| Relevance | 0.911 | 0.911 | 0.925 |

n = 150 for every dimension. The paper previously made no IAA claim; it now reports the full
per-dimension table, which is what a reviewer asking "how reliable are your human raters?"
wants to see. The old parenthetical ("κ of 0.91–0.97") in the judge-calibration paragraph was
removed as redundant.

### 2026-08-28 — Zenodo release bundle built
`scripts/build_zenodo_release.py` (new) generates `revision_v2/zenodo_release/` — 11.3 MB,
9 files. Re-run it whenever the dataset changes; it regenerates everything deterministically.

### 2026-08-28 — Authorship + annotator wording
Author list changed (above). Because Jaloliddin Rajabov was one of the **two annotators** but is
no longer an author, three claims in the paper were corrected: `sec:res_human` now says "the
first author and an independent trained annotator" (was "two native Uzbek-speaking authors"),
Limitations says "one of the two annotators is the first author" (was "the annotators are
authors of this paper"), and Rajabov is thanked in Acknowledgments. The `note` field inside
`uzbek_multi_domain_absa_gold80.json` was updated the same way (80/80 records).

### Earlier (2026-07-07/08) — for the record
- P0-B writing pass; encoder baselines integrated (5-system table, Discussion subsection);
  human validation subsection + judge-calibration table; gold80 release file; abstract rewrite.
- BDCC citations: DONE 2026-08-28 — Related Work cites `bdcc_ner_disaster`,
  `bdcc_offensive_lowresource`, `bdcc10050161`, `bioengineering12070687`,
  `electronics14040690`, `smartcities8020062`; all DOIs verified against Crossref.

---

## Zenodo release — how to upload

Bundle: `paper_materials/revision_v2/zenodo_release/` (rebuild with
`python scripts/build_zenodo_release.py`).

### Why the current record says "Cannot preview file"

Zenodo previews **only one file** on the landing page: the first previewable file in
alphanumeric order, unless another is ticked in the upload form's **Preview** column. Its
JSON/text previewer refuses anything over **~1 MB**. The current record leads with
`uzbek_multi_domain_absa_full.json` at **3.3 MB**, so the previewer gives up — that error is
purely a size/format issue, not a corrupted upload.

The bundle fixes this by leading with `00_PREVIEW_uzbek_absa_sample.csv` (80 KB, 296 rows,
all 23 domains represented). CSV gets Zenodo's *table* previewer, so the record opens on a
readable sortable grid rather than a wall of JSON. `01_README.md` sorts second and previews as
formatted Markdown if the CSV is ever removed.

### Steps

1. Open the record <https://zenodo.org/records/18790639> while logged in.
2. Click **New version** (top right). This clones the metadata and **carries over all files
   from the previous version** — you do not start from an empty record.
3. Delete the carried-over files (the file list has a delete control per file while the new
   version is a draft). This is the clean path, since filenames changed between versions.
4. Upload all 9 files from `zenodo_release/`.
5. In the file list, tick the **Preview** checkbox on `00_PREVIEW_uzbek_absa_sample.csv`.
   (If your record's form does not show that column, the alphanumeric rule still selects it —
   that is why it is named `00_`.)
6. Set **Version** to `v2` and confirm the license is CC BY 4.0.
7. **Publish.**

The concept DOI `10.5281/zenodo.18790639` keeps resolving to the newest version, so the DOI
already printed in the manuscript stays correct. The new version also mints its own
version-specific DOI.

### Caveat worth knowing
Zenodo files are **immutable once published** — a published version cannot be edited, only
superseded by another new version. Get the file set right in the draft before hitting Publish.
