#!/usr/bin/env python3
"""
Simple example showing how to use the fine-tuning project.
"""
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    print("Llama3 MAWPS Fine-tuning Project")
    print("=" * 40)
    print()

    print("Available scripts:")
    print("1. Training (DialoGPT):  python scripts/train.py --config configs/default_config.json")
    print(
        "2. Quick test:           python scripts/train.py --config configs/quick_test_config.json --debug"
    )
    print(
        "3. Llama-2 (needs auth): python scripts/train.py --config configs/llama2_config.json"
    )
    print(
        "4. Inference:            python scripts/inference.py --model_path ./results/run_*/final_model --interactive"
    )
    print(
        "4. Evaluate:   python scripts/evaluate.py --model_path ./results/run_*/final_model"
    )
    print()

    print("Jupyter notebook:")
    print("  jupyter lab notebooks/llama3_mawps_finetuning.ipynb")
    print()

    print("Configuration files:")
    print("  configs/default_config.json      - Full training configuration")
    print("  configs/quick_test_config.json   - Quick test with smaller settings")
    print()

    print("Project structure:")
    print("  src/               - Source code modules")
    print("  scripts/           - Training and evaluation scripts")
    print("  configs/           - Configuration files")
    print("  notebooks/         - Jupyter notebook for exploration")
    print("  requirements.txt   - Python dependencies")
    print()

    print("To modify the project:")
    print("1. Edit configs/*.json for different hyperparameters")
    print("2. Modify src/ modules for custom functionality")
    print("3. Use different models by changing 'model.name' in config")
    print("4. Try different datasets by changing 'data.dataset_name' in config")
    print()

    print("Example math word problems:")
    examples = [
        "Sarah has 15 apples. She gives 7 to Tom. How many apples does Sarah have left?",
        "A school has 450 students. 180 are boys. How many are girls?",
        "Mike bought 3 packages of 12 pencils each. How many pencils total?",
    ]

    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")

    print()
    print("Happy fine-tuning! 🚀")


if __name__ == "__main__":
    main()
