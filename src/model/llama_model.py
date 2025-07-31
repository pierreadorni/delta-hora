"""
Llama3 model setup and configuration.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class LlamaModelSetup:
    """Handles Llama3 model loading and LoRA setup."""

    def __init__(self, model_config: Dict[str, Any], lora_config: Dict[str, Any]):
        self.model_config = model_config
        self.lora_config = lora_config
        self.model = None
        self.tokenizer = None

    def load_tokenizer(self) -> AutoTokenizer:
        """Load and configure the tokenizer."""
        logger.info(f"Loading tokenizer: {self.model_config['name']}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["name"],
            cache_dir=self.model_config.get("cache_dir"),
            trust_remote_code=True,
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Set padding side to right for training
        self.tokenizer.padding_side = "right"

        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        return self.tokenizer

    def create_quantization_config(self) -> BitsAndBytesConfig:
        """Create quantization configuration."""
        quant_config = self.model_config.get("quantization", {})

        if quant_config.get("load_in_4bit", False):
            logger.info("Setting up 4-bit quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, quant_config.get("bnb_4bit_compute_dtype", "float16")
                ),
                bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quant_config.get(
                    "bnb_4bit_use_double_quant", True
                ),
            )
        elif quant_config.get("load_in_8bit", False):
            logger.info("Setting up 8-bit quantization")
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            logger.info("No quantization configured")
            return None

    def load_base_model(self) -> AutoModelForCausalLM:
        """Load the base Llama model."""
        logger.info(f"Loading base model: {self.model_config['name']}")

        quantization_config = self.create_quantization_config()

        # Model loading arguments
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_config["name"],
            "cache_dir": self.model_config.get("cache_dir"),
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        # Prepare model for k-bit training if using quantization
        if quantization_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        logger.info("Base model loaded successfully")
        return self.model

    def setup_lora(self) -> AutoModelForCausalLM:
        """Setup LoRA configuration and apply to model."""
        if self.model is None:
            raise ValueError("Base model must be loaded before setting up LoRA")

        logger.info("Setting up LoRA configuration")

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            target_modules=self.lora_config["target_modules"],
            lora_dropout=self.lora_config["lora_dropout"],
            bias=self.lora_config["bias"],
            task_type=self.lora_config["task_type"],
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.print_trainable_parameters()

        logger.info("LoRA setup complete")
        return self.model

    def print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_param = 0

        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )

    def get_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Complete model setup pipeline."""
        self.load_tokenizer()
        self.load_base_model()
        self.setup_lora()

        return self.model, self.tokenizer
