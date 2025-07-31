"""
Training utilities and main trainer class.
"""

import os
import torch
from transformers import Trainer, TrainingArguments
from peft import PeftModel
import logging
from typing import Dict, Any, Optional
import wandb

logger = logging.getLogger(__name__)


class LlamaTrainer:
    """Custom trainer for Llama fine-tuning with LoRA."""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        data_collator,
        training_config: Dict[str, Any],
        logging_config: Dict[str, Any],
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.training_config = training_config
        self.logging_config = logging_config

        self.trainer = None
        self.training_args = None

    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.logging_config.get("use_wandb", False):
            try:
                wandb.init(
                    project=self.logging_config.get(
                        "wandb_project", "llama-finetuning"
                    ),
                    entity=self.logging_config.get("wandb_entity"),
                    name=f"llama-mawps-{os.path.basename(self.training_config['output_dir'])}",
                    config={
                        "model_name": "llama-7b",
                        "dataset": "MAWPS",
                        "method": "LoRA",
                        **self.training_config,
                    },
                )
                logger.info("Weights & Biases initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Weights & Biases: {e}")
                self.logging_config["use_wandb"] = False

    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments."""

        # Ensure output directory exists
        os.makedirs(self.training_config["output_dir"], exist_ok=True)

        # Set report_to based on wandb configuration
        report_to = ["wandb"] if self.logging_config.get("use_wandb", False) else []

        self.training_args = TrainingArguments(
            output_dir=self.training_config["output_dir"],
            num_train_epochs=self.training_config["num_train_epochs"],
            per_device_train_batch_size=self.training_config[
                "per_device_train_batch_size"
            ],
            per_device_eval_batch_size=self.training_config[
                "per_device_eval_batch_size"
            ],
            gradient_accumulation_steps=self.training_config[
                "gradient_accumulation_steps"
            ],
            learning_rate=self.training_config["learning_rate"],
            max_grad_norm=self.training_config["max_grad_norm"],
            weight_decay=self.training_config["weight_decay"],
            warmup_ratio=self.training_config["warmup_ratio"],
            lr_scheduler_type=self.training_config["lr_scheduler_type"],
            logging_steps=self.training_config["logging_steps"],
            evaluation_strategy=self.training_config["evaluation_strategy"],
            eval_steps=self.training_config["eval_steps"],
            save_strategy=self.training_config["save_strategy"],
            save_steps=self.training_config["save_steps"],
            save_total_limit=self.training_config["save_total_limit"],
            load_best_model_at_end=self.training_config["load_best_model_at_end"],
            metric_for_best_model=self.training_config["metric_for_best_model"],
            greater_is_better=self.training_config["greater_is_better"],
            dataloader_num_workers=self.training_config["dataloader_num_workers"],
            remove_unused_columns=self.training_config["remove_unused_columns"],
            optim=self.training_config["optim"],
            # Additional settings for stability
            fp16=True,
            dataloader_pin_memory=False,
            group_by_length=True,
            # Reporting
            report_to=report_to,
            logging_dir=os.path.join(self.training_config["output_dir"], "logs"),
        )

        return self.training_args

    def create_trainer(self) -> Trainer:
        """Create the Hugging Face trainer."""
        if self.training_args is None:
            self.create_training_arguments()

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        return self.trainer

    def train(self):
        """Execute the training process."""
        logger.info("Starting training...")

        # Setup logging
        self.setup_wandb()

        # Create trainer if not already created
        if self.trainer is None:
            self.create_trainer()

        # Start training
        try:
            train_result = self.trainer.train()

            # Save the final model
            self.trainer.save_model()

            # Save training metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)

            logger.info("Training completed successfully")
            return train_result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        finally:
            if self.logging_config.get("use_wandb", False):
                wandb.finish()

    def evaluate(self):
        """Evaluate the model."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call create_trainer() first.")

        logger.info("Starting evaluation...")
        eval_result = self.trainer.evaluate()

        # Log evaluation metrics
        self.trainer.log_metrics("eval", eval_result)
        self.trainer.save_metrics("eval", eval_result)

        logger.info("Evaluation completed")
        return eval_result

    def save_model(self, save_path: str):
        """Save the fine-tuned model."""
        logger.info(f"Saving model to {save_path}")

        # Save the PEFT model
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(save_path)
        else:
            self.trainer.save_model(save_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

        logger.info("Model saved successfully")
