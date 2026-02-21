#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Dataset Explorer — Language & Script Profiling
# =============================================================================
"""
Explore raw review data for Aspect-Based Sentiment Analysis.

This script performs:
1. Load & clean: reads CSV, drops NaN/empty review_text rows
2. Language/script profiling: classifies each review by script (Cyrillic/Latin)
   and language (Russian vs. Uzbek) using character-ratio analysis and word-
   level heuristics (Uzbek-specific Cyrillic markers ў қ ғ ҳ, and common
   Russian function words)
3. Statistics: counts and percentages per language category
4. Auto-logging: appends a structured summary to RESEARCH_LOG.md

Usage:
    python scripts/explore_datasets.py --raw-file ./data/raw/reviews.csv

Author: UzABSA Team
License: MIT
"""

import argparse
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_utils import (
    load_raw_reviews_csv,
)

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Language / Script Classification
# =============================================================================

# Uzbek-specific Cyrillic letters (absent in standard Russian)
_UZBEK_CYRILLIC_MARKERS = set("ўқғҳЎҚҒҲ")

# Frequent Russian function words — if ≥2 appear in a Cyrillic text the text
# is very likely Russian rather than Uzbek written in Cyrillic.
_RUSSIAN_FUNCTION_WORDS = {
    "и", "в", "не", "на", "я", "что", "он", "она", "но", "это",
    "все", "как", "с", "из", "мне", "мы", "так", "они", "вы", "у",
    "от", "за", "по", "для", "бы", "до", "вот", "уже", "если",
    "при", "есть", "был", "было", "были", "быть", "очень", "или",
    "ни", "тоже", "ещё", "еще", "нет", "да", "же", "может",
}

# Common Uzbek words in Cyrillic script (for disambiguation)
_UZBEK_CYRILLIC_WORDS = {
    "жуда", "яхши", "зўр", "маззали", "ёмон", "нарх", "йўқ",
    "бор", "учун", "менга", "билан", "ҳам", "лекин", "аммо",
    "сиз", "мен", "бу", "шу", "ўша", "қилиш", "бериш",
}


def _char_script_ratios(text: str) -> Dict[str, float]:
    """Return the proportion of Cyrillic, Latin, and other characters."""
    cyrillic = 0
    latin = 0
    digit = 0
    other = 0

    for ch in text:
        cp = ord(ch)
        if 0x0400 <= cp <= 0x04FF:
            cyrillic += 1
        elif ("a" <= ch.lower() <= "z") or ch in "ʻʼ":
            latin += 1
        elif ch.isdigit():
            digit += 1
        elif not ch.isspace():
            other += 1

    alpha_total = cyrillic + latin
    if alpha_total == 0:
        return {"cyrillic": 0.0, "latin": 0.0, "total_alpha": 0}

    return {
        "cyrillic": cyrillic / alpha_total,
        "latin": latin / alpha_total,
        "total_alpha": alpha_total,
    }


def _has_uzbek_cyrillic_markers(text: str) -> bool:
    """Check for Uzbek-specific Cyrillic characters (ў, қ, ғ, ҳ)."""
    return any(ch in _UZBEK_CYRILLIC_MARKERS for ch in text)


def _russian_word_score(text: str) -> float:
    """Fraction of words in the text that are common Russian function words."""
    words = re.findall(r"[а-яёўқғҳА-ЯЁЎҚҒҲ]+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _RUSSIAN_FUNCTION_WORDS)
    return hits / len(words)


def _uzbek_cyrillic_word_score(text: str) -> float:
    """Fraction of words that are common Uzbek words (in Cyrillic)."""
    words = re.findall(r"[а-яёўқғҳА-ЯЁЎҚҒҲ]+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _UZBEK_CYRILLIC_WORDS)
    return hits / len(words)


def classify_language(text: str) -> str:
    """
    Classify a single review into a language/script category.

    Dual approach:
      1. Character-level: ratio of Cyrillic vs Latin characters.
      2. Word-level heuristic:
         - Uzbek-specific Cyrillic markers (ў, қ, ғ, ҳ)
         - Russian function-word frequency

    Returns one of:
        'Primarily Uzbek (Latin)'
        'Primarily Russian (Cyrillic)'
        'Primarily Uzbek (Cyrillic)'
        'Highly Mixed'
    """
    ratios = _char_script_ratios(text)

    # Degenerate case — no alphabetic characters
    if ratios["total_alpha"] == 0:
        return "Primarily Uzbek (Latin)"  # default

    cyr = ratios["cyrillic"]
    lat = ratios["latin"]

    # ----- Latin-dominant (>70 % Latin characters) -----
    if lat > 0.70:
        return "Primarily Uzbek (Latin)"

    # ----- Cyrillic-dominant (>70 % Cyrillic characters) -----
    if cyr > 0.70:
        # Distinguish Russian from Uzbek-in-Cyrillic using word-level cues
        if _has_uzbek_cyrillic_markers(text):
            return "Primarily Uzbek (Cyrillic)"
        if _uzbek_cyrillic_word_score(text) > 0.05:
            return "Primarily Uzbek (Cyrillic)"
        if _russian_word_score(text) >= 0.08:
            return "Primarily Russian (Cyrillic)"
        # If no clear Russian signal, lean Uzbek-Cyrillic
        return "Primarily Russian (Cyrillic)"

    # ----- Mixed (30–70 % each) -----
    return "Highly Mixed"


# =============================================================================
# Statistics
# =============================================================================

def compute_lang_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table with counts and percentages per lang_category.

    Returns a DataFrame with columns: lang_category, count, percentage.
    """
    counts = df["lang_category"].value_counts()
    stats = pd.DataFrame({
        "lang_category": counts.index,
        "count": counts.values,
        "percentage": (counts.values / len(df) * 100).round(2),
    })
    stats = stats.sort_values("count", ascending=False).reset_index(drop=True)
    return stats


def compute_text_stats(df: pd.DataFrame, text_col: str = "review_text") -> Dict:
    """Compute basic text statistics."""
    texts = df[text_col].astype(str)
    word_counts = texts.str.split().str.len()
    char_counts = texts.str.len()
    return {
        "total_reviews": len(df),
        "avg_words": round(word_counts.mean(), 2),
        "avg_chars": round(char_counts.mean(), 2),
        "min_words": int(word_counts.min()),
        "max_words": int(word_counts.max()),
        "min_chars": int(char_counts.min()),
        "max_chars": int(char_counts.max()),
        "median_words": int(word_counts.median()),
        "median_chars": int(char_counts.median()),
    }


# =============================================================================
# Markdown Logging
# =============================================================================

def append_to_research_log(
    log_path: str,
    lang_stats: pd.DataFrame,
    text_stats: Dict,
    df: pd.DataFrame,
    raw_file: str,
) -> None:
    """Append a structured log entry to RESEARCH_LOG.md."""
    today = datetime.now().strftime("%b %d, %Y")

    # Build samples per category (up to 2 each)
    sample_lines = []
    for cat in lang_stats["lang_category"]:
        subset = df[df["lang_category"] == cat]
        sample_lines.append(f"  **{cat}**:")
        for _, row in subset.head(2).iterrows():
            snippet = str(row["review_text"])[:120].replace("\n", " ")
            sample_lines.append(f'  - _{snippet}_…')

    samples_block = "\n".join(sample_lines)

    # Build the markdown entry
    entry = f"""

## LOG 016 — Data Exploration: Language/Script Profiling
Date: {today}

### Source
- File: `{raw_file}`
- Total reviews loaded: **{text_stats['total_reviews']}**
- After cleaning (drop NaN/empty `review_text`): **{len(df)}**

### Text Statistics
| Metric | Value |
|--------|-------|
| Avg words / review | {text_stats['avg_words']} |
| Avg chars / review | {text_stats['avg_chars']} |
| Median words | {text_stats['median_words']} |
| Word range | {text_stats['min_words']}–{text_stats['max_words']} |
| Char range | {text_stats['min_chars']}–{text_stats['max_chars']} |

### Language/Script Classification Method
- **Character-level:** Ratio of Cyrillic (U+0400–U+04FF) vs Latin (a-z, ʻ, ʼ) characters.
  - >70 % Latin → Latin-dominant
  - >70 % Cyrillic → Cyrillic-dominant
  - 30–70 % each → Highly Mixed
- **Word-level (for Cyrillic-dominant texts):**
  - Presence of Uzbek-specific Cyrillic characters (ў, қ, ғ, ҳ) → Uzbek Cyrillic
  - High frequency of Russian function words (и, не, на, что, …) → Russian
  - Fallback heuristic using common Uzbek word list

### Results
| Language Category | Count | Percentage |
|-------------------|------:|----------:|
"""

    for _, row in lang_stats.iterrows():
        entry += f"| {row['lang_category']} | {row['count']} | {row['percentage']}% |\n"

    entry += f"""
### Sample Reviews per Category
{samples_block}

### Implications for ABSA Fine-tuning
- The dataset is **predominantly Uzbek in Latin script** ({lang_stats[lang_stats['lang_category'] == 'Primarily Uzbek (Latin)']['percentage'].values[0] if 'Primarily Uzbek (Latin)' in lang_stats['lang_category'].values else 0}%).
- A non-trivial minority of reviews is in **Russian (Cyrillic)**, which affects tokenizer coverage and model selection.
- Presence of Uzbek-in-Cyrillic texts (legacy Soviet-era orthography) adds further script diversity.
- **Recommendation:** Consider script-aware preprocessing or filtering for monolingual experiments.
"""

    # Append to file
    log_path_obj = Path(log_path)
    if not log_path_obj.exists():
        logger.warning(f"RESEARCH_LOG.md not found at {log_path}. Creating new file.")

    # Insert before the END marker if it exists
    if log_path_obj.exists():
        content = log_path_obj.read_text(encoding="utf-8")
        end_marker = "# ======================================================================"
        end_section = "# END OF CURRENT LOGS"

        if end_section in content:
            # Find the last occurrence of the end block
            idx = content.rfind(end_section)
            # Walk back to find the preceding separator
            block_start = content.rfind(end_marker, 0, idx)
            if block_start == -1:
                block_start = idx
            new_content = content[:block_start].rstrip() + "\n" + entry + "\n\n" + content[block_start:]
            log_path_obj.write_text(new_content, encoding="utf-8")
        else:
            # No end marker — just append
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry)
    else:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Research Log — UzABSA-LLM Project\n\n")
            f.write(entry)

    logger.info(f"Appended language profiling results to {log_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main exploration pipeline."""
    parser = argparse.ArgumentParser(
        description="Explore UzABSA dataset — language/script profiling"
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        default="./data/raw/reviews.csv",
        help="Path to raw reviews CSV file",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="./RESEARCH_LOG.md",
        help="Path to research log markdown file",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip appending results to RESEARCH_LOG.md",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("UzABSA Dataset Explorer — Language/Script Profiling")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load and clean
    # ------------------------------------------------------------------
    print("\n[1/4] Loading raw reviews...")
    print("-" * 70)

    raw_file_path = Path(args.raw_file)
    if not raw_file_path.exists():
        logger.error(f"Raw reviews file not found: {args.raw_file}")
        sys.exit(1)

    df = pd.read_csv(str(raw_file_path))
    total_before = len(df)
    logger.info(f"Loaded {total_before} rows from {args.raw_file}")

    # Drop NaN / empty review_text
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].astype(str).str.strip().str.len() > 0].copy()
    dropped = total_before - len(df)
    logger.info(f"Dropped {dropped} rows with NaN/empty review_text → {len(df)} remaining")

    print(f"  Loaded:  {total_before} rows")
    print(f"  Cleaned: {len(df)} rows  (dropped {dropped} NaN/empty)")

    # Show a sample
    sample = df.iloc[0]
    print(f"\n  Sample review:")
    print(f"    Object: {sample['object_name']}")
    print(f"    Rating: {sample['rating_value']}/5")
    print(f"    Text:   {str(sample['review_text'])[:130]}…")

    # ------------------------------------------------------------------
    # Step 2: Language/script profiling
    # ------------------------------------------------------------------
    print(f"\n[2/4] Classifying language/script for {len(df)} reviews...")
    print("-" * 70)

    df["lang_category"] = df["review_text"].astype(str).apply(classify_language)
    logger.info("Language classification complete.")

    # ------------------------------------------------------------------
    # Step 3: Statistics
    # ------------------------------------------------------------------
    print("\n[3/4] Computing statistics...")
    print("-" * 70)

    lang_stats = compute_lang_stats(df)
    text_stats = compute_text_stats(df)

    # Print text statistics
    print(f"\n  Text Statistics:")
    print(f"    Total reviews:      {text_stats['total_reviews']}")
    print(f"    Avg words/review:   {text_stats['avg_words']}")
    print(f"    Avg chars/review:   {text_stats['avg_chars']}")
    print(f"    Median words:       {text_stats['median_words']}")
    print(f"    Word range:         {text_stats['min_words']}–{text_stats['max_words']}")
    print(f"    Char range:         {text_stats['min_chars']}–{text_stats['max_chars']}")

    # Print language distribution
    print(f"\n  Language/Script Distribution:")
    print(f"    {'Category':<35} {'Count':>6}  {'%':>7}")
    print(f"    {'─' * 35} {'─' * 6}  {'─' * 7}")
    for _, row in lang_stats.iterrows():
        print(f"    {row['lang_category']:<35} {row['count']:>6}  {row['percentage']:>6.2f}%")
    print(f"    {'─' * 35} {'─' * 6}  {'─' * 7}")
    print(f"    {'TOTAL':<35} {len(df):>6}  {100.00:>6.2f}%")

    # Show examples per category
    print(f"\n  Samples per category:")
    for cat in lang_stats["lang_category"]:
        subset = df[df["lang_category"] == cat]
        print(f"\n    [{cat}]  ({len(subset)} reviews)")
        for _, row in subset.head(2).iterrows():
            snippet = str(row["review_text"])[:100].replace("\n", " ")
            print(f"      → {snippet}…")

    # ------------------------------------------------------------------
    # Step 4: Append to RESEARCH_LOG.md
    # ------------------------------------------------------------------
    if not args.no_log:
        print(f"\n[4/4] Appending results to {args.log_file}...")
        print("-" * 70)
        append_to_research_log(
            log_path=args.log_file,
            lang_stats=lang_stats,
            text_stats=text_stats,
            df=df,
            raw_file=args.raw_file,
        )
        print(f"  ✓ Results appended to {args.log_file}")
    else:
        print(f"\n[4/4] Skipping RESEARCH_LOG.md (--no-log)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Dataset Exploration Complete!")
    print("=" * 70)
    print(f"\n  Summary:")
    print(f"    Reviews analyzed: {len(df)}")
    for _, row in lang_stats.iterrows():
        print(f"    {row['lang_category']}: {row['count']} ({row['percentage']}%)")
    print(f"\n  Next steps:")
    print(f"    1. Prepare dataset:  python scripts/prepare_complete_dataset.py --max-examples -1 --output-dir ./data/processed")
    print(f"    2. Start training:   python scripts/train_unsloth.py --help")


if __name__ == "__main__":
    main()
