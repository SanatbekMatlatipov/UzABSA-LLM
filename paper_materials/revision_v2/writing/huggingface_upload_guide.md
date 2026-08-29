# HuggingFace upload — exact steps

Target repo: **`Sanatbek/UzABSA-LLM`** (one repo, one branch per model).
This replaces the currently published weights, which were trained under the
defective pipeline (literal `"None"` system prompt; contaminated split).

Everything below was verified on this machine on 2026-08-29:
token `depparse` has `repo.write` on `user/Sanatbek`, and
`python scripts/push_to_hub.py --all --dry-run` runs clean.

---

## Before you start

| Check | Command | Expected |
|---|---|---|
| Logged in | `python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"` | `Sanatbek` |
| Free disk | — | nothing needed; upload streams from disk |
| Upload size | `du -sch outputs/rerun_v2/uzabsa_*_v2/merged_model outputs/rerun_v2/uzabsa_*_v2/lora_adapters` | **44 GB total** |

**Time:** at 10 MB/s this is roughly 75 minutes per model, ~4 hours total. Use a
wired/stable connection. `huggingface_hub` resumes interrupted uploads, so a
dropped connection is recoverable — just re-run the same command.

If you are not logged in:
```bash
huggingface-cli login          # paste a token with WRITE scope
```

---

## Step 1 — Dry run (no upload, ~10 seconds)

```bash
cd C:/Users/User/code/UzABSA-LLM
python scripts/push_to_hub.py --all --dry-run
```

Confirm in the output:
- each model reports **19 files** and **~15.0 GB**;
- the paths read `outputs/rerun_v2/uzabsa_*_v2/...` (**not** `outputs/my_run/...`);
- `eval_results_20260829_*.json` is the eval file being attached.

Then read one generated card to be sure the corrected numbers and the
correction notice are present:

```bash
grep -A 6 "Corrected release" outputs/hub_preview/qwen2.5-7b_README.md
grep -E "ATE Exact F1|Pair F1" outputs/hub_preview/main_README.md
```

Expected: exact-ATE **0.7077**, pair F1 **0.6448** for Qwen — *not* 0.6603/0.5795.

---

## Step 2 — Upload

Recommended: one model at a time, so a failure costs one branch, not four.

```bash
python scripts/push_to_hub.py --branch main            # README only, ~7 KB, do this first
python scripts/push_to_hub.py --branch qwen2.5-7b      # ~15 GB
python scripts/push_to_hub.py --branch llama3.1-8b     # ~15 GB
python scripts/push_to_hub.py --branch deepseek-r1-7b  # ~15 GB
```

Or all at once:
```bash
python scripts/push_to_hub.py --all
```

Each model branch receives: the merged fp16 model (4 safetensors shards),
`lora_adapters/`, `eval_results.json`, `experiment_summary.json`, and a model
card carrying the corrected metrics plus the **"Corrected release (2026-08-29)"**
notice explaining both defects and telling earlier downloaders to re-pull.

---

## Step 3 — Verify on the Hub

1. <https://huggingface.co/Sanatbek/UzABSA-LLM> — the landing page table should
   show ATE Exact F1 **0.7077 / 0.7036 / 0.6776**.
2. Switch branch (top-left dropdown) to `qwen2.5-7b` → the card should open with
   the correction notice, and **Files** should list 4 safetensors shards.
3. Spot-check the model actually loads:
   ```python
   from transformers import AutoTokenizer
   AutoTokenizer.from_pretrained("Sanatbek/UzABSA-LLM", revision="qwen2.5-7b")
   ```

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `401`/`403` on upload | token lacks write scope | `huggingface-cli login` with a **Write** token |
| Upload stalls then resumes | normal xet/LFS chunking | leave it; it resumes automatically |
| Interrupted mid-model | connection dropped | re-run the same `--branch` command; completed shards are skipped |
| Card shows old numbers | stale `eval_results` picked up | confirm `outputs/rerun_v2/uzabsa_*_v2/eval_results_20260829_*.json` exists |
| `UnicodeEncodeError` | old script version | already fixed (stdout forced to UTF-8); `git pull` if you see it |

---

## What this does **not** cover

- The **dataset** repo `Sanatbek/aspect-based-sentiment-analysis-uzbek` is the
  upstream UzABSA corpus and is unchanged — nothing to push there.
- The **Zenodo** multi-domain corpus is a separate artifact (uploaded 2026-08-28).
