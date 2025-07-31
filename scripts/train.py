#!/usr/bin/env python3
"""
Main training script for fine-tuning Llama3 7B on MAWPS dataset with LoRA.
"""
import argparse
import os
import sys
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import Config
from data.dataset_loader import MAWPSDatasetLoader
from data.preprocessor import MAWPSPreprocessor
from model.llama_model import LlamaModelSetup
from training.trainer import LlamaTrainer
from utils.logging import setup_logging, TrainingLogger
from utils.metrics import CustomMetricsCallback

import logging


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama3 7B on MAWPS dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.json",
        help="Path to configuration file",
    )
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument(
        "--wandb_project", type=str, help="Override Weights & Biases project name"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with smaller dataset"
    )

    args = parser.parse_args()

    # Load configuration
    config = Config.from_json(args.config)

    # Override config with command line arguments
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.wandb_project:
        config.logging.wandb_project = args.wandb_project

    # Update output directory with timestamp
    config.update_output_dir("./results")

    # Setup logging
    log_file = os.path.join(config.training.output_dir, "training.log")
    setup_logging(config.logging.log_level, log_file)

    logger = logging.getLogger(__name__)
    training_logger = TrainingLogger()

    logger.info("Starting Llama3 fine-tuning pipeline")
    training_logger.log_training_start(config)

    try:
        # Check GPU availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Training will be very slow on CPU.")
        else:
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
            logger.info(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        # 1. Load dataset
        logger.info("Loading MAWPS dataset...")
        dataset_loader = MAWPSDatasetLoader(config.data.dataset_name)
        dataset = dataset_loader.load_dataset()

        # Debug mode: use smaller dataset
        if args.debug:
            logger.info("Debug mode: Using smaller dataset")
            train_dataset = dataset[config.data.train_split].select(range(100))
            eval_dataset = dataset[config.data.test_split].select(range(50))
        else:
            train_dataset = dataset[config.data.train_split]
            eval_dataset = dataset[config.data.test_split]

        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Eval dataset size: {len(eval_dataset)}")

        # 2. Setup model and tokenizer
        logger.info("Setting up model and tokenizer...")
        model_setup = LlamaModelSetup(
            model_config=config.model.__dict__, lora_config=config.lora.__dict__
        )
        model, tokenizer = model_setup.get_model_and_tokenizer()

        training_logger.log_model_info(model)

        # 3. Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = MAWPSPreprocessor(tokenizer, config.data.max_length)

        train_dataset = preprocessor.preprocess_dataset(
            train_dataset, config.data.instruction_template
        )
        eval_dataset = preprocessor.preprocess_dataset(
            eval_dataset, config.data.instruction_template
        )

        data_collator = preprocessor.get_data_collator()

        # 4. Setup trainer
        logger.info("Setting up trainer...")
        trainer = LlamaTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            training_config=config.training.__dict__,
            logging_config=config.logging.__dict__,
        )

        # Add custom metrics
        metrics_callback = CustomMetricsCallback(tokenizer)
        trainer.trainer = trainer.create_trainer()
        trainer.trainer.compute_metrics = metrics_callback.compute_metrics

        # 5. Train the model
        logger.info("Starting training...")
        train_result = trainer.train()

        # 6. Evaluate the model
        logger.info("Evaluating model...")
        eval_result = trainer.evaluate()
        training_logger.log_evaluation(eval_result)

        # 7. Save the final model
        final_model_path = os.path.join(config.training.output_dir, "final_model")
        trainer.save_model(final_model_path)

        training_logger.log_training_end()
        logger.info(
            f"Training completed successfully. Model saved to: {final_model_path}"
        )

        return train_result, eval_result

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
