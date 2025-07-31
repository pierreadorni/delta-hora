#!/usr/bin/env python3
"""
Evaluation script for the fine-tuned model.
"""
import argparse
import os
import sys
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from data.dataset_loader import MAWPSDatasetLoader
from utils.metrics import MathMetrics
import logging


def load_model_and_tokenizer(model_path: str, base_model_name: str = None):
    """Load the fine-tuned model and tokenizer."""
    logger = logging.getLogger(__name__)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if base_model_name:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

    return model, tokenizer


def generate_predictions(model, tokenizer, problems: list, config: dict):
    """Generate predictions for a list of problems."""
    predictions = []

    for i, problem in enumerate(problems):
        if i % 10 == 0:
            print(f"Processing {i}/{len(problems)}")

        prompt = f"Below is a math word problem. Solve it step by step.\n\n### Problem:\n{problem}\n\n### Solution:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=config.get("max_new_tokens", 256),
                temperature=config.get("temperature", 0.1),
                do_sample=config.get("do_sample", True),
                top_p=config.get("top_p", 0.9),
                repetition_penalty=config.get("repetition_penalty", 1.1),
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "### Solution:" in generated_text:
            solution = generated_text.split("### Solution:")[-1].strip()
        else:
            solution = generated_text.strip()

        predictions.append(solution)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Llama3 model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="Base model name (required if loading PEFT model)",
    )
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ChilleD/MAWPS",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--num_samples", type=int, help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument("--output_file", type=str, help="Save detailed results to file")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load config
    if args.config:
        config = Config.from_json(args.config)
        inference_config = config.inference.__dict__
    else:
        inference_config = {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        }

    # Load model
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset_loader = MAWPSDatasetLoader(args.dataset_name)
    dataset = dataset_loader.load_dataset()

    eval_data = dataset[args.split]
    if args.num_samples:
        eval_data = eval_data.select(range(min(args.num_samples, len(eval_data))))

    logger.info(f"Evaluating on {len(eval_data)} samples")

    # Extract problems and answers
    problems = []
    answers = []

    for example in eval_data:
        problem = example.get(
            "Question", example.get("question", example.get("Problem", ""))
        )
        answer = example.get(
            "Answer", example.get("answer", example.get("Solution", ""))
        )

        problems.append(problem)
        answers.append(str(answer))

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = generate_predictions(model, tokenizer, problems, inference_config)

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = MathMetrics.compute_metrics(predictions, answers)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Dataset: {args.dataset_name} ({args.split})")
    print(f"Samples: {len(eval_data)}")
    print(f"Model: {args.model_path}")
    print("-" * 30)

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("=" * 50)

    # Save detailed results
    if args.output_file:
        results = {
            "config": {
                "model_path": args.model_path,
                "base_model": args.base_model,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "num_samples": len(eval_data),
            },
            "metrics": metrics,
            "examples": [
                {"problem": prob, "expected": ans, "predicted": pred}
                for prob, ans, pred in zip(
                    problems[:10], answers[:10], predictions[:10]
                )  # Save first 10 examples
            ],
        }

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Detailed results saved to {args.output_file}")


if __name__ == "__main__":
    main()
