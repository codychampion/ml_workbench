#!/usr/bin/env python3
"""
CUAD MAKER Evaluation Pipeline
================================
Implements MAKER framework for contract review with zero errors.

Based on: Meyerson et al., "Solving a Million-Step LLM Task with Zero Errors"
Dataset: CUAD (Contract Understanding Atticus Dataset)

Usage:
    # Estimate baseline success rate
    python -m pipelines.evaluate.cuad_maker --mode estimate --samples 100

    # Run full MAKER experiment
    python -m pipelines.evaluate.cuad_maker --mode experiment --k 3 --samples 1000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from utils.decorators import task
from tqdm import tqdm


# ==============================================================================
# MAKER Framework Implementation
# ==============================================================================

class MAKERVoter:
    """
    Implements MAKER voting with red-flagging.

    Components:
    1. Maximal Agentic Decomposition - each clause is a separate task
    2. First-to-ahead-by-K voting - vote until one answer is k ahead
    3. Red-flagging - discard unreliable responses
    """

    def __init__(
        self,
        model: str = "llm",
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "ml-workbench-key",
        k: int = 3,
        max_tokens: int = 750,
        temperature_first: float = 0.0,
        temperature_subsequent: float = 0.1,
    ):
        """
        Initialize MAKER voter.

        Args:
            model: Model name (or "llm" for local vLLM)
            api_base: API base URL
            api_key: API key
            k: Voting threshold (first to k ahead wins)
            max_tokens: Maximum tokens (also red-flag threshold)
            temperature_first: Temperature for first vote
            temperature_subsequent: Temperature for subsequent votes
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required: pip install openai")

        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.k = k
        self.max_tokens = max_tokens
        self.temp_first = temperature_first
        self.temp_subsequent = temperature_subsequent

        # Statistics
        self.stats = {
            "total_votes": 0,
            "red_flagged": 0,
            "avg_votes_per_decision": [],
        }

    def is_red_flagged(self, response: str) -> Tuple[bool, str]:
        """
        Check if response should be red-flagged (discarded).

        Red flags:
        1. Length > max_tokens (over-analysis)
        2. Missing required format (YES/NO)
        3. Uncertainty markers (low confidence)
        4. Generic boilerplate (no specifics)

        Returns:
            (is_flagged, reason)
        """
        # Length check
        word_count = len(response.split())
        if word_count > self.max_tokens:
            return True, f"too_long ({word_count} words)"

        # Format check - must contain YES or NO
        response_upper = response.upper()
        has_yes = "YES" in response_upper
        has_no = "NO" in response_upper

        if not (has_yes or has_no):
            return True, "missing_answer"

        # Uncertainty markers
        uncertainty_markers = [
            "maybe", "possibly", "unclear", "might", "could be",
            "i'm not sure", "uncertain", "hard to say", "difficult to determine"
        ]
        response_lower = response.lower()
        for marker in uncertainty_markers:
            if marker in response_lower:
                return True, f"uncertainty ({marker})"

        # Generic boilerplate check
        if word_count < 10:
            return True, "too_short"

        return False, ""

    def parse_answer(self, response: str) -> Optional[str]:
        """
        Extract YES/NO answer from response.

        Returns:
            "YES", "NO", or None if ambiguous
        """
        response_upper = response.upper()

        # Look for clear indicators
        yes_indicators = ["YES", "CONTAINS", "PRESENT", "FOUND"]
        no_indicators = ["NO", "DOES NOT CONTAIN", "NOT PRESENT", "NOT FOUND", "ABSENT"]

        has_yes = any(ind in response_upper for ind in yes_indicators)
        has_no = any(ind in response_upper for ind in no_indicators)

        # Unambiguous
        if has_yes and not has_no:
            return "YES"
        if has_no and not has_yes:
            return "NO"

        # Ambiguous - return None
        return None

    def vote_on_task(
        self,
        contract_text: str,
        clause_category: str,
        max_votes: int = 20,
    ) -> Dict[str, Any]:
        """
        Vote on a single clause identification task.

        Args:
            contract_text: Contract text
            clause_category: Clause category to identify
            max_votes: Maximum votes before giving up

        Returns:
            {
                "answer": "YES" or "NO",
                "votes": {"YES": count, "NO": count},
                "total_votes": int,
                "red_flagged_count": int,
                "converged": bool,
            }
        """
        # Create prompt
        prompt = f"""Given this contract text, does it contain a {clause_category} clause?

Contract:
{contract_text[:4000]}  # Truncate if too long

Instructions:
- Answer YES or NO
- Provide a brief explanation citing specific contract text
- Be concise (under 200 words)

Answer:"""

        votes = Counter()
        vote_count = 0
        red_flagged_count = 0
        temperature = self.temp_first

        while vote_count < max_votes:
            # Get response
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                )
                response_text = response.choices[0].message.content
            except Exception as e:
                print(f"  Error calling LLM: {e}")
                break

            self.stats["total_votes"] += 1

            # Check red flag
            is_flagged, reason = self.is_red_flagged(response_text)
            if is_flagged:
                red_flagged_count += 1
                self.stats["red_flagged"] += 1
                # Continue voting (don't count this one)
                continue

            # Parse answer
            answer = self.parse_answer(response_text)
            if answer is None:
                red_flagged_count += 1
                continue

            votes[answer] += 1
            vote_count += 1

            # Use temperature > 0 for subsequent votes (diversity)
            temperature = self.temp_subsequent

            # Check if converged (first to ahead by k)
            if len(votes) >= 2:
                sorted_votes = votes.most_common(2)
                if sorted_votes[0][1] >= sorted_votes[1][1] + self.k:
                    # Converged!
                    self.stats["avg_votes_per_decision"].append(vote_count)
                    return {
                        "answer": sorted_votes[0][0],
                        "votes": dict(votes),
                        "total_votes": vote_count,
                        "red_flagged_count": red_flagged_count,
                        "converged": True,
                    }
            elif len(votes) == 1 and vote_count >= self.k:
                # Only one answer, and we have k votes for it
                answer = list(votes.keys())[0]
                if votes[answer] >= self.k:
                    self.stats["avg_votes_per_decision"].append(vote_count)
                    return {
                        "answer": answer,
                        "votes": dict(votes),
                        "total_votes": vote_count,
                        "red_flagged_count": red_flagged_count,
                        "converged": True,
                    }

        # Did not converge
        most_common = votes.most_common(1)
        if most_common:
            answer = most_common[0][0]
        else:
            answer = "NO"  # Default if no valid votes

        return {
            "answer": answer,
            "votes": dict(votes),
            "total_votes": vote_count,
            "red_flagged_count": red_flagged_count,
            "converged": False,
        }


# ==============================================================================
# Evaluation Functions
# ==============================================================================

@task(name="estimate-success-rate")
def estimate_success_rate(
    data_path: Path,
    sample_size: int = 100,
    model: str = "llm",
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "ml-workbench-key",
) -> Dict[str, Any]:
    """
    Estimate per-step success rate (p) on a sample.

    This is Phase 1 from the experiment plan: Single-Clause Classification baseline.

    Returns:
        {
            "success_rate": float,
            "total_tasks": int,
            "correct": int,
            "per_category_accuracy": dict,
        }
    """
    print(f"\n{'='*60}")
    print("Phase 1: Baseline Success Rate Estimation")
    print(f"{'='*60}")

    # Load CUAD data
    data_file = data_path / "cuad_contracts.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"CUAD data not found: {data_file}")

    # Load samples
    samples = []
    with data_file.open("r") as f:
        for idx, line in enumerate(f):
            if idx >= sample_size:
                break
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples")

    # Create voter with k=1 (no voting, just baseline)
    voter = MAKERVoter(
        model=model,
        api_base=api_base,
        api_key=api_key,
        k=1,
    )

    # Evaluate
    correct = 0
    total = 0
    category_stats = {}

    for sample in tqdm(samples, desc="Evaluating baseline"):
        contract_text = sample["text"]

        # Test each clause category
        for key, value in sample.items():
            if key in ["id", "text", "text_length", "timestamp"]:
                continue

            # Ground truth
            ground_truth = "YES" if value else "NO"

            # Get prediction
            result = voter.vote_on_task(contract_text, key, max_votes=1)
            prediction = result["answer"]

            # Check correctness
            is_correct = (prediction == ground_truth)
            if is_correct:
                correct += 1
            total += 1

            # Track per-category
            if key not in category_stats:
                category_stats[key] = {"correct": 0, "total": 0}
            category_stats[key]["total"] += 1
            if is_correct:
                category_stats[key]["correct"] += 1

    success_rate = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Success Rate: {success_rate:.4f} ({correct}/{total})")
    print(f"{'='*60}")

    # Calculate per-category accuracy
    per_category = {}
    for category, stats in category_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        per_category[category] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    return {
        "success_rate": success_rate,
        "total_tasks": total,
        "correct": correct,
        "per_category_accuracy": per_category,
        "voter_stats": voter.stats,
    }


@task(name="run-maker-experiment")
def run_maker_experiment(
    data_path: Path,
    output_dir: Path,
    k: int = 3,
    sample_size: int = 100,
    model: str = "llm",
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "ml-workbench-key",
) -> Dict[str, Any]:
    """
    Run full MAKER experiment with voting.

    This is Phase 3 from the experiment plan: Error Correction via Voting.

    Returns:
        {
            "accuracy": float,
            "zero_error_rate": float,
            "total_contracts": int,
            "total_cost_estimate": float,
        }
    """
    print(f"\n{'='*60}")
    print(f"Phase 3: MAKER Experiment (k={k})")
    print(f"{'='*60}")

    # Load CUAD data
    data_file = data_path / "cuad_contracts.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"CUAD data not found: {data_file}")

    # Load samples
    samples = []
    with data_file.open("r") as f:
        for idx, line in enumerate(f):
            if idx >= sample_size:
                break
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} contracts")

    # Create voter with specified k
    voter = MAKERVoter(
        model=model,
        api_base=api_base,
        api_key=api_key,
        k=k,
    )

    # Evaluate
    contract_results = []

    for sample in tqdm(samples, desc="MAKER voting"):
        contract_id = sample["id"]
        contract_text = sample["text"]

        # Process each clause category
        clause_results = []
        errors = 0

        for key, value in sample.items():
            if key in ["id", "text", "text_length", "timestamp"]:
                continue

            # Ground truth
            ground_truth = "YES" if value else "NO"

            # Vote
            result = voter.vote_on_task(contract_text, key)
            prediction = result["answer"]

            # Check correctness
            is_correct = (prediction == ground_truth)
            if not is_correct:
                errors += 1

            clause_results.append({
                "clause_category": key,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "correct": is_correct,
                "votes": result["votes"],
                "total_votes": result["total_votes"],
                "converged": result["converged"],
            })

        contract_results.append({
            "contract_id": contract_id,
            "clauses": clause_results,
            "errors": errors,
            "zero_errors": (errors == 0),
        })

    # Calculate metrics
    total_clauses = sum(len(r["clauses"]) for r in contract_results)
    correct_clauses = sum(
        sum(1 for c in r["clauses"] if c["correct"])
        for r in contract_results
    )
    zero_error_contracts = sum(1 for r in contract_results if r["zero_errors"])

    accuracy = correct_clauses / total_clauses if total_clauses > 0 else 0
    zero_error_rate = zero_error_contracts / len(contract_results) if contract_results else 0

    # Cost estimate (rough)
    avg_votes = (
        sum(voter.stats["avg_votes_per_decision"]) / len(voter.stats["avg_votes_per_decision"])
        if voter.stats["avg_votes_per_decision"] else k
    )
    cost_per_clause = 0.03  # Estimated for GPT-4 mini
    total_cost_estimate = total_clauses * avg_votes * cost_per_clause

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%} ({correct_clauses}/{total_clauses})")
    print(f"Zero-error rate: {zero_error_rate:.2%} ({zero_error_contracts}/{len(contract_results)})")
    print(f"Avg votes per decision: {avg_votes:.1f}")
    print(f"Red-flagged responses: {voter.stats['red_flagged']}")
    print(f"Estimated cost: ${total_cost_estimate:.2f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"maker_k{k}_results.json"
    with results_file.open("w") as f:
        json.dump({
            "config": {
                "k": k,
                "sample_size": sample_size,
                "model": model,
            },
            "metrics": {
                "accuracy": accuracy,
                "zero_error_rate": zero_error_rate,
                "total_contracts": len(contract_results),
                "total_clauses": total_clauses,
                "correct_clauses": correct_clauses,
                "avg_votes_per_decision": avg_votes,
            },
            "voter_stats": voter.stats,
            "contract_results": contract_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return {
        "accuracy": accuracy,
        "zero_error_rate": zero_error_rate,
        "total_contracts": len(contract_results),
        "total_cost_estimate": total_cost_estimate,
    }


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="CUAD MAKER Evaluation")
    parser.add_argument(
        "--mode",
        choices=["estimate", "experiment"],
        required=True,
        help="Mode: estimate baseline or run full experiment"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/collected/cuad/train"),
        help="CUAD data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/maker_results"),
        help="Output directory"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of contracts to evaluate"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Voting threshold (for experiment mode)"
    )
    parser.add_argument(
        "--model",
        default="llm",
        help="Model name"
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="API base URL"
    )
    parser.add_argument(
        "--api-key",
        default="ml-workbench-key",
        help="API key"
    )

    args = parser.parse_args()

    if args.mode == "estimate":
        estimate_success_rate(
            data_path=args.data_dir,
            sample_size=args.samples,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
        )
    else:  # experiment
        run_maker_experiment(
            data_path=args.data_dir,
            output_dir=args.output_dir,
            k=args.k,
            sample_size=args.samples,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
        )


if __name__ == "__main__":
    main()
