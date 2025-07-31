"""
Evaluation metrics for math word problems.
"""

import re
import logging
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score
import numpy as np

logger = logging.getLogger(__name__)


class MathMetrics:
    """Metrics specifically designed for evaluating math word problem solutions."""

    @staticmethod
    def extract_number(text: str) -> float:
        """Extract the final numerical answer from generated text."""
        if not text:
            return None

        # Look for patterns like "The answer is X" or just numbers at the end
        patterns = [
            r"(?:answer|result|solution)(?:\s+is)?\s*:?\s*([-+]?\d*\.?\d+)",
            r"(?:therefore|thus|so)\s*,?\s*([-+]?\d*\.?\d+)",
            r"([-+]?\d*\.?\d+)\s*$",  # Number at the end
            r"=\s*([-+]?\d*\.?\d+)",  # After equals sign
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.lower(), re.MULTILINE)
            if matches:
                try:
                    return float(matches[-1])  # Take the last match
                except ValueError:
                    continue

        # If no pattern matches, try to find any number in the text
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        if numbers:
            try:
                return float(numbers[-1])  # Take the last number
            except ValueError:
                pass

        return None

    @staticmethod
    def exact_match_accuracy(predictions: List[str], references: List[str]) -> float:
        """Calculate exact match accuracy between predictions and references."""
        pred_numbers = []
        ref_numbers = []

        for pred, ref in zip(predictions, references):
            pred_num = MathMetrics.extract_number(pred)
            ref_num = MathMetrics.extract_number(str(ref))

            pred_numbers.append(pred_num)
            ref_numbers.append(ref_num)

        # Filter out None values
        valid_pairs = [
            (p, r)
            for p, r in zip(pred_numbers, ref_numbers)
            if p is not None and r is not None
        ]

        if not valid_pairs:
            return 0.0

        pred_valid, ref_valid = zip(*valid_pairs)

        # Check for exact numerical matches (with small tolerance for floating point)
        matches = [abs(p - r) < 1e-6 for p, r in valid_pairs]

        return sum(matches) / len(valid_pairs)

    @staticmethod
    def parse_rate(predictions: List[str]) -> float:
        """Calculate the rate at which numerical answers can be parsed."""
        parsed_count = 0

        for pred in predictions:
            if MathMetrics.extract_number(pred) is not None:
                parsed_count += 1

        return parsed_count / len(predictions) if predictions else 0.0

    @staticmethod
    def compute_metrics(
        predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute comprehensive metrics for math problem evaluation."""
        metrics = {}

        # Exact match accuracy
        metrics["exact_match"] = MathMetrics.exact_match_accuracy(
            predictions, references
        )

        # Parse rate
        metrics["parse_rate"] = MathMetrics.parse_rate(predictions)

        # String-based exact match (for debugging)
        str_matches = [
            pred.strip() == ref.strip() for pred, ref in zip(predictions, references)
        ]
        metrics["string_exact_match"] = (
            sum(str_matches) / len(str_matches) if str_matches else 0.0
        )

        logger.info(f"Metrics computed: {metrics}")
        return metrics


class CustomMetricsCallback:
    """Callback for computing custom metrics during evaluation."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_metrics(self, eval_preds):
        """Compute metrics for evaluation."""
        predictions, labels = eval_preds

        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Extract the solution part (after the "### Solution:" marker)
        pred_solutions = []
        label_solutions = []

        for pred, label in zip(decoded_preds, decoded_labels):
            # Find the solution part
            if "### Solution:" in pred:
                pred_solution = pred.split("### Solution:")[-1].strip()
            else:
                pred_solution = pred.strip()

            if "### Solution:" in label:
                label_solution = label.split("### Solution:")[-1].strip()
            else:
                label_solution = label.strip()

            pred_solutions.append(pred_solution)
            label_solutions.append(label_solution)

        # Compute metrics
        metrics = MathMetrics.compute_metrics(pred_solutions, label_solutions)

        return metrics
