#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: LLM-as-Judge Quality Scoring
# =============================================================================
"""
Score ABSA annotation quality using an external LLM judge.
Implements Phase 2 from LOG 021 / LOG 024: judge on unannotated reviews.

Supports OpenAI, Anthropic, and any OpenAI-compatible API.

Usage:
    # OpenAI
    python scripts/llm_judge.py \
        --annotations ./data/annotated/reviews_annotated.json \
        --provider openai --model gpt-4o-mini \
        --sample-size 300 --output-dir ./data/judged

    # Anthropic
    python scripts/llm_judge.py \
        --annotations ./data/annotated/reviews_annotated.json \
        --provider anthropic --model claude-3-5-haiku-20241022 \
        --sample-size 300

    # Custom OpenAI-compatible endpoint (e.g., local vLLM, Together, Groq)
    python scripts/llm_judge.py \
        --annotations ./data/annotated/reviews_annotated.json \
        --provider openai --model meta-llama/Meta-Llama-3.1-70B-Instruct \
        --api-base https://api.together.xyz/v1 \
        --sample-size 300

Environment variables:
    OPENAI_API_KEY    — for OpenAI / OpenAI-compatible providers
    ANTHROPIC_API_KEY — for Anthropic

Author: UzABSA Team
License: MIT
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Judge Prompt Template (Phase 2 — no gold standard)
# =============================================================================

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for Aspect-Based Sentiment Analysis (ABSA) on Uzbek-language text.
Your task is to judge the quality of ABSA predictions made by a fine-tuned language model.

ABSA prediction format: a JSON with "aspects", where each aspect has:
- "term": the aspect phrase extracted from the text
- "category": aspect category (e.g., xizmat, ovqat, narx, muhit, boshqalar)
- "polarity": sentiment (positive, negative, neutral)

Score each dimension on a 1–5 scale where 5 is best.
Always return your evaluation as a valid JSON object — nothing else."""

JUDGE_USER_TEMPLATE = """\
Evaluate the following ABSA prediction. There is no gold-standard reference — \
judge based on your understanding of the text.

== REVIEW ==
Text: "{text}"
Business: "{business_name}" (Category: {business_category})
User rating: {user_rating}/5

== MODEL PREDICTION ==
{prediction_json}

== SCORING RUBRIC ==
1. completeness (1–5): Does the output capture ALL opinions/sentiments expressed in the text?
   5 = all opinions captured, 1 = most opinions missed
2. accuracy (1–5): Are the extracted aspect terms actually present or closely paraphrased in the text?
   5 = all terms present, 1 = mostly hallucinated terms
3. sentiment (1–5): Are the polarity labels correct for each aspect?
   5 = all correct, 1 = all wrong
4. relevance (1–5): Are the predicted categories appropriate and meaningful?
   5 = all appropriate, 1 = all irrelevant
5. overall (1–5): Overall annotation quality combining all dimensions.

Return ONLY a JSON object in this exact format (no markdown, no extra text):
{{"completeness": <int>, "accuracy": <int>, "sentiment": <int>, "relevance": <int>, "overall": <int>, "explanation": "<1-2 sentence justification>"}}"""


# =============================================================================
# Stratified Sampling
# =============================================================================

# Target sample sizes per domain (from LOG 024)
DOMAIN_SAMPLE_TARGETS = {
    "Restoran/Ovqatlanish": 50,   # In-domain
    "Bank/Moliya": 30,
    "Telekommunikatsiya": 25,
    "Tibbiyot/Sog'liqni saqlash": 25,
    "Ta'lim": 20,
    "E-tijorat": 20,
    "Transport/Logistika": 15,
    "Mehmonxona/Turizm": 15,
    # Remaining domains get proportional allocation from the budget
}


def stratified_sample(
    annotations: List[Dict],
    total_sample_size: int = 300,
    seed: int = 42,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Create a stratified sample across business domains.

    Priority domains get fixed allocation; remaining budget is spread
    proportionally across other domains (min 3 per domain).
    """
    import random
    random.seed(seed)

    # Group by domain
    by_domain: Dict[str, List[Dict]] = {}
    for ann in annotations:
        cat = ann.get("business_category", "Boshqa")
        by_domain.setdefault(cat, []).append(ann)

    # Compute allocation
    allocation: Dict[str, int] = {}
    budget_used = 0

    # Fixed allocations for priority domains
    for domain, target in DOMAIN_SAMPLE_TARGETS.items():
        available = len(by_domain.get(domain, []))
        n = min(target, available)
        if n > 0:
            allocation[domain] = n
            budget_used += n

    # Remaining budget for other domains
    remaining_budget = max(0, total_sample_size - budget_used)
    other_domains = [d for d in by_domain if d not in allocation]

    if other_domains and remaining_budget > 0:
        total_other = sum(len(by_domain[d]) for d in other_domains)
        for domain in other_domains:
            available = len(by_domain[domain])
            if total_other > 0:
                proportional = int(remaining_budget * available / total_other)
            else:
                proportional = 0
            n = max(3, min(proportional, available))  # At least 3 per domain
            allocation[domain] = n

    # Sample from each domain
    sampled = []
    actual_allocation: Dict[str, int] = {}
    for domain, n in allocation.items():
        pool = by_domain.get(domain, [])
        if len(pool) <= n:
            chosen = pool
        else:
            chosen = random.sample(pool, n)
        sampled.extend(chosen)
        actual_allocation[domain] = len(chosen)

    random.shuffle(sampled)
    logger.info(f"Stratified sample: {len(sampled)} reviews from {len(actual_allocation)} domains")
    return sampled, actual_allocation


# =============================================================================
# API Clients
# =============================================================================

class OpenAIClient:
    """Client for OpenAI and OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def chat(self, system: str, user: str) -> str:
        """Send a chat completion request and return the assistant message."""
        resp = self.client.post(
            f"{self.api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
                "max_tokens": 300,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def close(self):
        self.client.close()


class AnthropicClient:
    """Client for Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-haiku-20241022",
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=timeout)

    def chat(self, system: str, user: str) -> str:
        """Send a message request and return the assistant response."""
        resp = self.client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "system": system,
                "messages": [
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
                "max_tokens": 300,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    def close(self):
        self.client.close()


def get_client(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
):
    """Create an API client based on provider."""
    if provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass --api-key."
            )
        base = api_base or "https://api.openai.com/v1"
        return OpenAIClient(api_key=key, model=model, api_base=base)

    elif provider == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass --api-key."
            )
        return AnthropicClient(api_key=key, model=model)

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")


# =============================================================================
# Score Parsing
# =============================================================================

SCORE_KEYS = ["completeness", "accuracy", "sentiment", "relevance", "overall"]


def parse_judge_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse the judge's JSON response into scores."""
    cleaned = response_text.strip()

    # Remove markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try direct JSON parse
    try:
        result = json.loads(cleaned)
        if all(k in result for k in SCORE_KEYS):
            # Clamp scores to 1-5
            for k in SCORE_KEYS:
                result[k] = max(1, min(5, int(result[k])))
            return result
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: extract JSON object from text
    match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if all(k in result for k in SCORE_KEYS):
                for k in SCORE_KEYS:
                    result[k] = max(1, min(5, int(result[k])))
                return result
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Last resort: regex extraction
    scores = {}
    for key in SCORE_KEYS:
        m = re.search(rf'"{key}"\s*:\s*(\d)', cleaned, re.IGNORECASE)
        if m:
            scores[key] = max(1, min(5, int(m.group(1))))
    if len(scores) == len(SCORE_KEYS):
        # Try to get explanation
        exp_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', cleaned)
        scores["explanation"] = exp_match.group(1) if exp_match else ""
        return scores

    logger.warning(f"Failed to parse judge response: {response_text[:200]}...")
    return None


# =============================================================================
# Judge Pipeline
# =============================================================================

def build_judge_prompt(annotation: Dict) -> str:
    """Build the user prompt for the judge from an annotation."""
    prediction_json = json.dumps(
        {"aspects": annotation.get("aspects", [])},
        indent=2,
        ensure_ascii=False,
    )
    return JUDGE_USER_TEMPLATE.format(
        text=annotation["text"],
        business_name=annotation.get("business_name", "Unknown"),
        business_category=annotation.get("business_category", "Boshqa"),
        user_rating=annotation.get("user_rating", "N/A"),
        prediction_json=prediction_json,
    )


def run_judge(
    client,
    annotations: List[Dict],
    max_retries: int = 3,
    delay_between_calls: float = 0.5,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 25,
) -> List[Dict]:
    """
    Run the LLM judge on all annotations.

    Returns a list of dicts with original annotation + judge scores.
    """
    from tqdm import tqdm

    results = []
    start_idx = 0

    # Resume from checkpoint
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        results = checkpoint["results"]
        start_idx = checkpoint["next_idx"]
        logger.info(f"Resuming judge from checkpoint: {start_idx}/{len(annotations)} done")

    successes = sum(1 for r in results if r.get("judge_scores"))
    failures = sum(1 for r in results if not r.get("judge_scores"))

    for i in tqdm(range(start_idx, len(annotations)), desc="Judging",
                  total=len(annotations) - start_idx):
        ann = annotations[i]
        user_prompt = build_judge_prompt(ann)

        scores = None
        for attempt in range(max_retries):
            try:
                response = client.chat(JUDGE_SYSTEM_PROMPT, user_prompt)
                scores = parse_judge_response(response)
                if scores:
                    break
                logger.warning(f"Parse failed for review {ann['review_id']} (attempt {attempt + 1})")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = min(60, 2 ** attempt * 5)
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"HTTP error for {ann['review_id']}: {e}")
                    break
            except Exception as e:
                logger.error(f"Error for {ann['review_id']}: {e}")
                break

        result = {
            "review_id": ann["review_id"],
            "text": ann["text"],
            "business_name": ann.get("business_name"),
            "business_category": ann.get("business_category"),
            "user_rating": ann.get("user_rating"),
            "aspects": ann.get("aspects", []),
            "num_aspects": ann.get("num_aspects", 0),
            "judge_scores": scores,
            "judge_raw": response if scores else None,
        }
        results.append(result)

        if scores:
            successes += 1
        else:
            failures += 1

        # Rate limiting
        time.sleep(delay_between_calls)

        # Checkpoint
        if checkpoint_path and (len(results) % checkpoint_every == 0):
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"results": results, "next_idx": i + 1},
                          f, ensure_ascii=False)
            logger.info(f"Judge checkpoint: {len(results)}/{len(annotations)}")

    logger.info(f"Judge complete: {successes} scored, {failures} failed out of {len(results)}")

    # Clean up checkpoint
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()

    return results


# =============================================================================
# Aggregation & Reporting
# =============================================================================

def aggregate_scores(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate judge scores per domain and overall."""
    scored = [r for r in results if r.get("judge_scores")]

    if not scored:
        return {"error": "No scored results"}

    # Overall averages
    overall = {k: 0.0 for k in SCORE_KEYS}
    for r in scored:
        for k in SCORE_KEYS:
            overall[k] += r["judge_scores"][k]
    for k in SCORE_KEYS:
        overall[k] = round(overall[k] / len(scored), 2)

    # Per-domain
    domain_scores: Dict[str, Dict] = {}
    for r in scored:
        cat = r.get("business_category", "Boshqa")
        if cat not in domain_scores:
            domain_scores[cat] = {k: [] for k in SCORE_KEYS}
            domain_scores[cat]["count"] = 0
        domain_scores[cat]["count"] += 1
        for k in SCORE_KEYS:
            domain_scores[cat][k].append(r["judge_scores"][k])

    domain_averages = {}
    for cat, data in sorted(domain_scores.items(), key=lambda x: -x[1]["count"]):
        domain_averages[cat] = {
            "count": data["count"],
            **{k: round(sum(data[k]) / len(data[k]), 2) for k in SCORE_KEYS},
        }

    # Score distribution
    distribution = {k: {str(i): 0 for i in range(1, 6)} for k in SCORE_KEYS}
    for r in scored:
        for k in SCORE_KEYS:
            distribution[k][str(r["judge_scores"][k])] += 1

    # Quality tiers (based on overall score)
    tiers = {"include_gte_3.5": 0, "flag_2.5_to_3.5": 0, "exclude_lt_2.5": 0}
    for r in scored:
        ov = r["judge_scores"]["overall"]
        if ov >= 3.5:
            tiers["include_gte_3.5"] += 1
        elif ov >= 2.5:
            tiers["flag_2.5_to_3.5"] += 1
        else:
            tiers["exclude_lt_2.5"] += 1

    return {
        "total_judged": len(results),
        "successfully_scored": len(scored),
        "parse_failures": len(results) - len(scored),
        "overall_averages": overall,
        "domain_averages": domain_averages,
        "score_distribution": distribution,
        "quality_tiers": tiers,
    }


def save_judge_results(
    results: List[Dict],
    report: Dict,
    output_dir: Path,
    provider: str,
    model: str,
):
    """Save judge results and aggregated report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Full results
    results_path = output_dir / "judge_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved judge results: {results_path}")

    # 2. Aggregated report
    report["judge_provider"] = provider
    report["judge_model"] = model
    report["timestamp"] = timestamp

    report_path = output_dir / "judge_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved judge report: {report_path}")

    # 3. Human-readable summary
    summary_lines = [
        "=" * 60,
        "LLM-AS-JUDGE QUALITY REPORT",
        f"Judge: {provider} / {model}",
        f"Date: {timestamp}",
        "=" * 60,
        "",
        f"Total judged: {report['total_judged']}",
        f"Successfully scored: {report['successfully_scored']}",
        f"Parse failures: {report['parse_failures']}",
        "",
        "--- OVERALL AVERAGES (1-5 scale) ---",
    ]
    for k, v in report["overall_averages"].items():
        summary_lines.append(f"  {k:15s}: {v:.2f}")

    summary_lines.append("")
    summary_lines.append("--- QUALITY TIERS ---")
    tiers = report["quality_tiers"]
    total_scored = report["successfully_scored"]
    for tier, count in tiers.items():
        pct = count / total_scored * 100 if total_scored > 0 else 0
        summary_lines.append(f"  {tier:25s}: {count:4d} ({pct:5.1f}%)")

    summary_lines.append("")
    summary_lines.append("--- PER-DOMAIN SCORES ---")
    summary_lines.append(
        f"{'Domain':<35s} {'N':>4s} {'Comp':>5s} {'Acc':>5s} {'Sent':>5s} {'Rel':>5s} {'Ovrl':>5s}"
    )
    summary_lines.append("-" * 70)
    for cat, data in report["domain_averages"].items():
        summary_lines.append(
            f"{cat:<35s} {data['count']:>4d} "
            f"{data['completeness']:>5.2f} {data['accuracy']:>5.2f} "
            f"{data['sentiment']:>5.2f} {data['relevance']:>5.2f} "
            f"{data['overall']:>5.2f}"
        )

    summary_lines.append("=" * 60)

    summary_text = "\n".join(summary_lines)
    summary_path = output_dir / "judge_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info(f"Saved judge summary: {summary_path}")

    # Print to console
    print("\n" + summary_text)

    return results_path, report_path, summary_path


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge quality scoring for ABSA annotations"
    )
    parser.add_argument(
        "--annotations", type=str,
        default="./data/annotated/reviews_annotated.json",
        help="Path to annotated reviews JSON",
    )
    parser.add_argument(
        "--provider", type=str, default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key (overrides env var)",
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="Custom API base URL (for OpenAI-compatible providers)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=300,
        help="Total stratified sample size (default: 300)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/judged",
        help="Output directory for judge results",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified sampling (default: 42)",
    )
    parser.add_argument(
        "--no-sample", action="store_true",
        help="Judge ALL annotations (no sampling)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("UzABSA-LLM: LLM-as-Judge Quality Scoring")
    logger.info("=" * 60)

    # 1. Load annotations
    logger.info(f"Loading annotations from: {args.annotations}")
    with open(args.annotations, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    logger.info(f"Loaded {len(annotations)} annotations")

    # Filter to only those with parse_success and >=1 aspect
    scoreable = [a for a in annotations if a.get("parse_success") and a.get("num_aspects", 0) > 0]
    logger.info(f"Scoreable annotations (parse OK + ≥1 aspect): {len(scoreable)}")

    # 2. Stratified sampling
    if args.no_sample:
        sampled = scoreable
        allocation = {}
    else:
        sampled, allocation = stratified_sample(
            scoreable, total_sample_size=args.sample_size, seed=args.seed
        )
        logger.info(f"Domain allocation: {json.dumps(allocation, indent=2)}")

    # 3. Create API client
    logger.info(f"Connecting to {args.provider} / {args.model}...")
    client = get_client(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )

    # 4. Run judge
    checkpoint_path = output_dir / "judge_checkpoint.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nJudging {len(sampled)} annotations...")
    results = run_judge(
        client=client,
        annotations=sampled,
        delay_between_calls=args.delay,
        checkpoint_path=checkpoint_path,
    )

    client.close()

    # 5. Aggregate
    logger.info("\nAggregating scores...")
    report = aggregate_scores(results)
    report["sample_allocation"] = allocation

    # 6. Save
    save_judge_results(results, report, output_dir, args.provider, args.model)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
