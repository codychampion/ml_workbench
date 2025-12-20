#!/usr/bin/env python3
"""
Model Metrics Calculation
=========================
Calculate evaluation metrics for models.

Usage:
    python -m pipelines.evaluate.metrics --predictions ./outputs/predictions.json --references ./data/test/labels.json
    python -m pipelines.evaluate.metrics --model ./models/captioner/v1 --test-dir ./data/test
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


class CaptioningMetrics:
    """Calculate captioning metrics (BLEU, ROUGE, etc.)."""

    def __init__(self):
        self._load_metrics()

    def _load_metrics(self):
        """Load metric calculators."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from rouge_score import rouge_scorer
            self.bleu_smooth = SmoothingFunction().method1
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
            self._has_nltk = True
        except ImportError:
            print("Warning: nltk/rouge_score not installed, some metrics unavailable")
            self._has_nltk = False

    def calculate_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate BLEU scores."""
        if not self._has_nltk:
            return {"error": "nltk not installed"}

        from nltk.translate.bleu_score import sentence_bleu

        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = [ref.lower().split()]
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=self.bleu_smooth)
            scores.append(score)

        return {
            "bleu_mean": np.mean(scores),
            "bleu_std": np.std(scores),
            "bleu_min": np.min(scores),
            "bleu_max": np.max(scores)
        }

    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate ROUGE scores."""
        if not self._has_nltk:
            return {"error": "rouge_score not installed"}

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return {
            "rouge1": np.mean(rouge1_scores),
            "rouge2": np.mean(rouge2_scores),
            "rougeL": np.mean(rougeL_scores)
        }

    def calculate_all(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate all metrics."""
        return {
            "bleu": self.calculate_bleu(predictions, references),
            "rouge": self.calculate_rouge(predictions, references),
            "count": len(predictions)
        }


class ClassificationMetrics:
    """Calculate classification metrics."""

    def calculate(
        self,
        predictions: List[int],
        references: List[int],
        labels: Optional[List[str]] = None
    ) -> Dict:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )

        return {
            "accuracy": accuracy_score(references, predictions),
            "precision_macro": precision_score(references, predictions, average='macro'),
            "recall_macro": recall_score(references, predictions, average='macro'),
            "f1_macro": f1_score(references, predictions, average='macro'),
            "confusion_matrix": confusion_matrix(references, predictions).tolist(),
            "classification_report": classification_report(
                references, predictions, target_names=labels, output_dict=True
            ) if labels else None
        }


def save_metrics(metrics: Dict, output_path: Path, model_name: str = "unknown"):
    """Save metrics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }

    output_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Metrics saved to: {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Calculate model metrics")

    parser.add_argument(
        "--predictions", "-p",
        type=Path,
        help="JSON file with predictions"
    )
    parser.add_argument(
        "--references", "-r",
        type=Path,
        help="JSON file with ground truth"
    )
    parser.add_argument(
        "--task",
        choices=["captioning", "classification"],
        default="captioning",
        help="Task type"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./outputs/metrics"),
        help="Output directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="unknown",
        help="Model name for the report"
    )

    args = parser.parse_args()

    if not args.predictions or not args.references:
        print("Error: --predictions and --references are required")
        print("\nExpected format:")
        print("  predictions.json: {\"predictions\": [\"caption1\", \"caption2\", ...]}")
        print("  references.json: {\"references\": [\"ref1\", \"ref2\", ...]}")
        return

    # Load data
    with open(args.predictions) as f:
        preds_data = json.load(f)
    with open(args.references) as f:
        refs_data = json.load(f)

    predictions = preds_data.get("predictions", preds_data)
    references = refs_data.get("references", refs_data)

    print(f"\nLoaded {len(predictions)} predictions and {len(references)} references")

    # Calculate metrics
    if args.task == "captioning":
        calculator = CaptioningMetrics()
        metrics = calculator.calculate_all(predictions, references)
    else:
        calculator = ClassificationMetrics()
        metrics = calculator.calculate(predictions, references)

    # Save and display
    output_file = args.output / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_metrics(metrics, output_file, args.model_name)

    print("\n" + "=" * 60)
    print("Metrics Summary")
    print("=" * 60)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
