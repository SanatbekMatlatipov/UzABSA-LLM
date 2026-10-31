#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Three-way split (train / dev / test) for the v3 protocol
# =============================================================================
"""
Carves a development set out of the training split so that checkpoint
selection never touches the 609-review held-out set. The held-out set is copied
unchanged as ``test``, so every v3 score stays comparable with the v2 tables
and with the BERT baselines.

The trainer keeps the best checkpoint on the split called ``validation`` and
``scripts/evaluate.py`` prefers a split called ``test``, so the output uses
exactly those names and neither script needs a new flag.

Usage:
    python scripts/make_dev_split.py --src ./data/processed \
        --out ./experiments/v3/data --dev-size 0.1

Author: UzABSA Team
License: MIT
"""

import argparse
import importlib.util as ilu
import json
from pathlib import Path

from datasets import DatasetDict, load_from_disk

_dp = ilu.spec_from_file_location(
    "uzabsa_data_prep", Path(__file__).parent.parent / "src" / "data_prep.py")
_data_prep = ilu.module_from_spec(_dp); _dp.loader.exec_module(_data_prep)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="./data/processed")
    ap.add_argument("--out", default="./experiments/v3/data")
    ap.add_argument("--dev-size", type=float, default=0.1, help="fraction of train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = load_from_disk(args.src)
    # grouped by normalised review text, as for the train/validation split
    inner = _data_prep._grouped_train_val_split(src["train"], args.dev_size, args.seed)
    out = DatasetDict({"train": inner["train"], "validation": inner["validation"],
                       "test": src["validation"]})

    keys = {name: {_data_prep._review_key(t) for t in split["text"]} for name, split in out.items()}
    overlap = {f"{a}&{b}": len(keys[a] & keys[b])
               for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    assert not any(overlap.values()), f"review texts shared across splits: {overlap}"

    out.save_to_disk(args.out)
    stats = {"source": args.src, "dev_size": args.dev_size, "seed": args.seed,
             "rows": {name: len(split) for name, split in out.items()},
             "unique_texts": {name: len(k) for name, k in keys.items()},
             "shared_texts": overlap,
             "test_is_v2_heldout": True}
    Path(args.out, "split_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
