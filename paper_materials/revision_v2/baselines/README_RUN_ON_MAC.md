# BERT baselines — how to run on your Mac M2 Max

These three commands add the classical encoder baselines the reviewers asked for (R1).
Everything runs on Apple MPS (no CUDA, no bitsandbytes). Run from the repo root in the
**project env** (the one with `torch`, `transformers`, `datasets`, `evaluate`, `seqeval`).

> Verify the env first: `python -c "import torch;print(torch.backends.mps.is_available())"`
> should print `True`. If `evaluate`/`seqeval` are missing: `pip install evaluate seqeval`.

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
