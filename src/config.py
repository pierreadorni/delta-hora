"""
Configuration management for the fine-tuning project.
"""

import json
import os
from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    name: str
    cache_dir: str
    quantization: Dict[str, Any]


@dataclass
class LoRAConfig:
    r: int
    lora_alpha: int
    target_modules: list
    lora_dropout: float
    bias: str
    task_type: str


@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_grad_norm: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    dataloader_num_workers: int
    remove_unused_columns: bool
    optim: str


@dataclass
class DataConfig:
    dataset_name: str
    max_length: int
    train_split: str
    test_split: str
    instruction_template: str
    response_template: str


@dataclass
class LoggingConfig:
    use_wandb: bool
    wandb_project: str
    wandb_entity: str
    log_level: str


@dataclass
class InferenceConfig:
    max_new_tokens: int
    temperature: float
    do_sample: bool
    top_p: float
    repetition_penalty: float


@dataclass
class Config:
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    data: DataConfig
    logging: LoggingConfig
    inference: InferenceConfig

    @classmethod
    def from_json(cls, config_path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        return cls(
            model=ModelConfig(**config_dict["model"]),
            lora=LoRAConfig(**config_dict["lora"]),
            training=TrainingConfig(**config_dict["training"]),
            data=DataConfig(**config_dict["data"]),
            logging=LoggingConfig(**config_dict["logging"]),
            inference=InferenceConfig(**config_dict["inference"]),
        )

    def to_json(self, config_path: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    def update_output_dir(self, base_dir: str):
        """Update output directory with timestamp."""
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training.output_dir = os.path.join(base_dir, f"run_{timestamp}")
