# BERT baselines — how to run on your Mac M2 Max

These commands add the classical encoder baselines the reviewers asked for (R1).
Everything runs on Apple MPS (no CUDA, no bitsandbytes, no unsloth — none of that installs
usefully on Apple Silicon and none of it is needed for a BERT-base token classifier).

## 0. Set up a dedicated environment (do this once)

**Check you have a native arm64 Python first.** MPS acceleration only works if Python
itself is an arm64 build — an x86_64 build (e.g. an Intel-Rosetta pyenv install) silently
falls back to CPU with no error, just much slower.

```bash
python3 -c "import platform; print(platform.machine())"
# must print: arm64        (x86_64 => wrong Python build, see fix below)
```

If it prints `x86_64`, install a native build and use that instead, e.g.:
```bash
# Homebrew Python (arm64 by default on Apple Silicon):
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv-bert
# or, with pyenv, make sure the arm64 toolchain is used:
#   arch -arm64 pyenv install 3.11.9   (then use that version below)
```

Create the venv and install dependencies **for this baseline only** (do NOT run the
project's root `requirements.txt` — it's CUDA/unsloth-oriented and will try to pull in
bitsandbytes/trl, which are irrelevant here and may fail to build on Apple Silicon):

```bash
python3 -m venv .venv-bert
source .venv-bert/bin/activate

# Install torch first (no --index-url needed — the default PyPI wheel for macOS
# already ships Apple MPS support, unlike the CUDA case in the main requirements.txt).
pip install --upgrade pip
pip install torch

# Then the rest of this baseline's dependencies:
pip install -r paper_materials/revision_v2/baselines/requirements_bert_baseline.txt
```

**Verify the setup** before running anything else:
```bash
python3 -c "
import platform, torch, transformers, datasets, seqeval, sklearn, scipy
print('arch      :', platform.machine())
print('torch     :', torch.__version__)
print('mps avail :', torch.backends.mps.is_available())
print('transformers:', transformers.__version__)
"
```
Expect `arch: arm64` and `mps avail: True`. If `mps avail` is `False` on an arm64 Mac,
reinstall torch (`pip install --upgrade --force-reinstall torch`) — very old torch
versions (<2.0) predate MPS support.

Run everything below **from the repo root**, with this venv activated.

## 1. Build the BIO dataset (once, ~seconds)

```bash
python scripts/prepare_bio_dataset.py --data ./data/processed --out ./data/bio_processed
```
Expect a printed **term-alignment rate** for train/val (some gold terms can't be located
in the text by surface form — that's expected and logged in `data/bio_processed/stats.json`;
we report it as a caveat). This uses the identical 5,480/609 split the LLMs used.

## 2. Fine-tune each encoder (~10–40 min each on M2 Max)

```bash
python scripts/train_bert_baseline.py \
  --model tahrirchi/tahrirchi-bert-base \
  --out paper_materials/revision_v2/baselines/tahrirchi-bert

python scripts/train_bert_baseline.py \
  --model elmurod1202/bertbek-news-big-cased \
  --out paper_materials/revision_v2/baselines/bertbek
```
If MPS reports out-of-memory: add `--batch-size 8` (or `4`). To force CPU: `--device cpu`
(slower but safe). Full console output is saved to `<out>/training_log.txt`.

## 3. Evaluate each on the SAME 609 set the LLMs used (~1–2 min each)

```bash
python scripts/eval_bert_baseline.py \
  --model-dir paper_materials/revision_v2/baselines/tahrirchi-bert/model \
  --data ./data/processed \
  --out paper_materials/revision_v2/baselines/tahrirchi-bert

python scripts/eval_bert_baseline.py \
  --model-dir paper_materials/revision_v2/baselines/bertbek/model \
  --data ./data/processed \
  --out paper_materials/revision_v2/baselines/bertbek
```

This prints ATE (exact/partial) F1, pair F1, and sentiment — the **same metrics and same
reference set** as the LLM table — and writes `eval_results.json` +
`preds_per_example.jsonl` (the latter feeds the significance tests in P2-C).

## 4. Send me the numbers

Paste the two `eval_results.json` (or the console output) back to me. I'll:
- add a baseline block to the results table (LLMs vs Tahrirchi-BERT vs BERTbek),
- write the Results/Discussion sentences answering R1,
- run the paired-bootstrap significance test (best LLM vs best BERT).

### Sanity check (recommended, per the plan's verification step)
After eval, spot-check 5 reviews: open `preds_per_example.jsonl`, eyeball that
`pred_aspects` are real spans from the `text` with plausible polarities. If the model
predicts almost nothing (all `O`), the term-alignment rate in step 1 was likely very low
— tell me and I'll adjust the alignment fallback.
