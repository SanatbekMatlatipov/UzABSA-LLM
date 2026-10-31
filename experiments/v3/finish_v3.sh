#!/bin/bash
# =============================================================================
# UzABSA-LLM v3: carry the pipeline to the end unattended
# =============================================================================
# Waits for the batched LLM driver to finish all nine runs, then runs every
# remaining stage and commits the results after each one, so an interruption
# never loses more than the stage that was running. Every stage skips work whose
# output already exists, so re-running this script is safe.
#
#   report      ensembles, significance tests, registry/SUMMARY/MANIFEST
#   annotate    re-annotate the 5,038 reviews with the best run on dev (~6 h)
#   judge-v3    the v1 two-judge panel on those annotations (needs OPENAI_API_KEY)
#   assemble    Include / Flag / Exclude tiers for the re-annotated corpus
#   collect     rebuild the registry and summary with the judge numbers folded in
#
# Usage:  nohup bash experiments/v3/finish_v3.sh &
# Log:    experiments/v3/logs/finish_v3.out
# =============================================================================
set -u
cd /c/Users/User/code/UzABSA-LLM || exit 1
exec >> experiments/v3/logs/finish_v3.out 2>&1

PY=.venv/Scripts/python.exe
LLM_GLOB="experiments/v3/runs/qwen2.5-7b_s*  experiments/v3/runs/llama3.1-8b_s*  experiments/v3/runs/deepseek-7b_s*"

say() { echo ""; echo "### $(date '+%Y-%m-%d %H:%M:%S')  $*"; }

count_preds() {   # $1 = test_preds.jsonl | dev_preds.jsonl
  local n=0 d
  for d in $LLM_GLOB; do [ -f "$d/$1" ] && n=$((n + 1)); done
  echo "$n"
}

commit_push() {   # $1 = commit subject
  git add -A experiments/v3 >/dev/null 2>&1
  # The batched LLM driver narrows config.json's llm.seeds to one seed while it
  # runs and restores the full list when it exits. Committing that transient
  # state would tell anyone reproducing the work to train a single seed, so
  # keep whatever is already committed until the file has all three seeds back.
  if ! git diff --cached --quiet -- experiments/v3/config.json; then
    seeds=$($PY -c "import json;print(len(json.load(open('experiments/v3/config.json'))['llm']['seeds']))" 2>/dev/null || echo 0)
    if [ "${seeds:-0}" -lt 3 ]; then
      git restore --staged experiments/v3/config.json 2>/dev/null
      echo "  kept committed config.json (working copy is the driver's ${seeds}-seed version)"
    fi
  fi
  if git diff --cached --quiet; then
    echo "  nothing new to commit"
    return
  fi
  git commit -q -m "$1" && echo "  committed: $(git log --oneline -1)"
  if git push -q origin main 2>&1; then
    echo "  pushed to origin/main"
  else
    echo "  PUSH FAILED -- the commit is safe locally; push by hand later"
  fi
}

say "finish_v3 starting"

# ---------------------------------------------------------------------------
# 1. Wait for the nine LLM runs. Cap the wait so a dead driver cannot hang
#    this script forever; whatever finished still gets reported.
# ---------------------------------------------------------------------------
say "waiting for 9 test + 9 dev LLM predictions"
for i in $(seq 1 720); do          # 720 x 60 s = 12 h ceiling
  t=$(count_preds test_preds.jsonl)
  d=$(count_preds dev_preds.jsonl)
  [ "$t" -ge 9 ] && [ "$d" -ge 9 ] && break
  if [ $((i % 30)) -eq 0 ]; then echo "  ${i}m: test $t/9, dev $d/9"; fi
  sleep 60
done
say "LLM predictions present: test $(count_preds test_preds.jsonl)/9, dev $(count_preds dev_preds.jsonl)/9"

# ---------------------------------------------------------------------------
# 2. Report stages first: they are minutes of CPU and make the run's numbers
#    citable straight away, before the six-hour annotation.
# ---------------------------------------------------------------------------
say "stage: report (ensemble, significance, collect)"
$PY -u experiments/v3/run_v3.py report
commit_push "v3: ensembles, significance tests and rebuilt registry for the v3 runs"

# ---------------------------------------------------------------------------
# 3. Re-annotate the corpus with the best run on dev (limitation (v)).
# ---------------------------------------------------------------------------
say "stage: annotate (about 6 h on one GPU)"
$PY -u experiments/v3/run_v3.py annotate
commit_push "v3: re-annotate the 5,038 reviews with the best v3 run on dev"

# ---------------------------------------------------------------------------
# 4. Judge the new annotations. judge_panel.py reads the key from the
#    environment and nothing in the repo loads .env, so export it here.
# ---------------------------------------------------------------------------
if [ -f .env ]; then
  set -a; . ./.env; set +a
  echo "  loaded .env"
fi
if [ -n "${OPENAI_API_KEY:-}" ]; then
  say "stage: judge-v3 then assemble (about \$10 of API calls, resumable)"
  $PY -u experiments/v3/run_v3.py judge-v3 assemble
  commit_push "v3: judge the re-annotated corpus with the v1 panel and build its tiers"
else
  say "SKIPPED judge-v3 and assemble: OPENAI_API_KEY is not set"
  echo "  put it in .env as OPENAI_API_KEY=sk-... and re-run this script"
fi

# ---------------------------------------------------------------------------
# 5. Final rebuild so SUMMARY.md and registry.json include everything above.
# ---------------------------------------------------------------------------
say "stage: collect (final rebuild)"
$PY -u experiments/v3/run_v3.py collect
commit_push "v3: final registry, SUMMARY and MANIFEST for the completed run"

say "finish_v3 done"
echo ""
echo "--- remaining by hand ---"
echo "  Hugging Face weights for the v3 champion, once its numbers are checked:"
echo "    python scripts/push_to_hub.py --branch <name>      (15 GB upload)"
echo "  Only the best-on-dev merged model is still on disk; the others were"
echo "  pruned to fit the drive and would need retraining to republish."
$PY experiments/v3/progress.py --no-color
