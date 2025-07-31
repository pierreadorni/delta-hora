"""
Logging utilities for the training process.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration."""

    # Configure logging level
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Reduce noise from other libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("tokenizers").setLevel(logging.WARNING)

    return root_logger


class TrainingLogger:
    """Custom logger for training metrics and progress."""

    def __init__(self, logger_name: str = "training"):
        self.logger = logging.getLogger(logger_name)

    def log_training_start(self, config):
        """Log training configuration."""
        self.logger.info("=" * 50)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 50)
        self.logger.info(f"Model: {config.model.name}")
        self.logger.info(f"Dataset: {config.data.dataset_name}")
        self.logger.info(f"Output dir: {config.training.output_dir}")
        self.logger.info(f"Epochs: {config.training.num_train_epochs}")
        self.logger.info(f"Learning rate: {config.training.learning_rate}")
        self.logger.info(f"Batch size: {config.training.per_device_train_batch_size}")
        self.logger.info(f"LoRA rank: {config.lora.r}")

    def log_training_end(self):
        """Log training completion."""
        self.logger.info("=" * 50)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 50)

    def log_evaluation(self, metrics):
        """Log evaluation metrics."""
        self.logger.info("Evaluation Results:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")

    def log_model_info(self, model):
        """Log model information."""
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

        # Log model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(
            f"Trainable percentage: {100 * trainable_params / total_params:.2f}%"
        )
