"""
Data preprocessing utilities for MAWPS dataset.
"""

from typing import Dict, Any, List, Callable
import logging
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class MAWPSPreprocessor:
    """Preprocesses MAWPS dataset for fine-tuning."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def format_instruction(self, example: Dict[str, Any], template: str) -> str:
        """Format a single example using the instruction template."""
        try:
            # Handle different possible field names in MAWPS dataset
            problem = example.get(
                "Question", example.get("question", example.get("Problem", ""))
            )
            solution = example.get(
                "Answer", example.get("answer", example.get("Solution", ""))
            )

            # Ensure solution is a string
            if isinstance(solution, (int, float)):
                solution = str(solution)

            formatted_text = template.format(problem=problem, solution=solution)
            return formatted_text

        except Exception as e:
            logger.warning(f"Failed to format example: {e}")
            logger.warning(f"Example keys: {list(example.keys())}")
            return ""

    def tokenize_function(
        self, examples: Dict[str, List], template: str
    ) -> Dict[str, List]:
        """Tokenize examples for training."""
        formatted_texts = []

        for i in range(len(examples[list(examples.keys())[0]])):
            example = {key: examples[key][i] for key in examples.keys()}
            formatted_text = self.format_instruction(example, template)
            formatted_texts.append(formatted_text)

        # Tokenize the formatted texts
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None,  # Return lists, not tensors
        )

        # For causal LM, input_ids and labels are the same
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def create_tokenize_function(self, template: str) -> Callable:
        """Create a tokenization function with the given template."""

        def tokenize_fn(examples):
            return self.tokenize_function(examples, template)

        return tokenize_fn

    def preprocess_dataset(self, dataset, template: str, num_proc: int = 4):
        """Preprocess entire dataset."""
        logger.info("Preprocessing dataset...")

        tokenize_fn = self.create_tokenize_function(template)

        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        logger.info(f"Preprocessing complete. Dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset

    def get_data_collator(self):
        """Get appropriate data collator for the tokenizer."""
        from transformers import DataCollatorForLanguageModeling

        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False  # We're doing causal LM, not masked LM
        )
