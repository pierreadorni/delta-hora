#!/usr/bin/env python3
"""
Inference script for the fine-tuned Llama3 model.
"""
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
import logging


def load_model_and_tokenizer(model_path: str, base_model_name: str = None):
    """Load the fine-tuned model and tokenizer."""
    logger = logging.getLogger(__name__)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    if base_model_name:
        logger.info(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float16, device_map="auto"
        )
        logger.info(f"Loading PEFT model from {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

    return model, tokenizer


def generate_solution(model, tokenizer, problem: str, config: dict):
    """Generate solution for a math word problem."""

    # Format the prompt
    prompt = f"Below is a math word problem. Solve it step by step.\n\n### Problem:\n{problem}\n\n### Solution:\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
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

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the solution part
    if "### Solution:" in generated_text:
        solution = generated_text.split("### Solution:")[-1].strip()
    else:
        solution = generated_text.strip()

    return solution


def interactive_mode(model, tokenizer, config: dict):
    """Run interactive inference mode."""
    print("\n" + "=" * 60)
    print("Interactive Math Problem Solver")
    print("Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            problem = input("Enter a math word problem: ").strip()

            if problem.lower() in ["quit", "exit", "q"]:
                break

            if not problem:
                continue

            print("\nGenerating solution...")
            solution = generate_solution(model, tokenizer, problem, config)

            print(f"\nProblem: {problem}")
            print(f"Solution: {solution}")
            print("-" * 60)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error generating solution: {e}")

    print("\nGoodbye!")


def batch_inference(model, tokenizer, problems: list, config: dict):
    """Run batch inference on multiple problems."""
    results = []

    for i, problem in enumerate(problems):
        print(f"Processing problem {i+1}/{len(problems)}")
        solution = generate_solution(model, tokenizer, problem, config)

        result = {"problem": problem, "solution": solution}
        results.append(result)

        print(f"Problem: {problem}")
        print(f"Solution: {solution}")
        print("-" * 40)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned Llama3 model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="Base model name (required if loading PEFT model)",
    )
    parser.add_argument("--config", type=str, help="Path to inference config file")
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--problems", type=str, nargs="+", help="Math problems to solve"
    )
    parser.add_argument(
        "--problems_file", type=str, help="JSON file containing problems to solve"
    )
    parser.add_argument("--output_file", type=str, help="Output file for batch results")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load inference config
    if args.config:
        config = Config.from_json(args.config)
        inference_config = config.inference.__dict__
    else:
        # Default inference config
        inference_config = {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        }

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    logger.info("Model loaded successfully")

    # Run inference
    if args.interactive:
        interactive_mode(model, tokenizer, inference_config)

    elif args.problems:
        results = batch_inference(model, tokenizer, args.problems, inference_config)

        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")

    elif args.problems_file:
        with open(args.problems_file, "r") as f:
            problems_data = json.load(f)

        # Handle different file formats
        if isinstance(problems_data, list):
            if isinstance(problems_data[0], str):
                problems = problems_data
            else:
                problems = [
                    item.get("problem", item.get("Question", ""))
                    for item in problems_data
                ]
        else:
            problems = problems_data.get("problems", [])

        results = batch_inference(model, tokenizer, problems, inference_config)

        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")

    else:
        # Default: interactive mode
        interactive_mode(model, tokenizer, inference_config)


if __name__ == "__main__":
    main()
