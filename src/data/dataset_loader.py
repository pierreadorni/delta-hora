"""
MAWPS dataset loading and management.
"""

from datasets import load_dataset, Dataset
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class MAWPSDatasetLoader:
    """Handles loading and basic processing of the MAWPS dataset."""

    def __init__(self, dataset_name: str = "ChilleD/MAWPS"):
        self.dataset_name = dataset_name
        self.dataset = None

    def load_dataset(self) -> Dict[str, Dataset]:
        """Load the MAWPS dataset from Hugging Face."""
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            self.dataset = load_dataset(self.dataset_name)

            # Log dataset info
            for split_name, split_data in self.dataset.items():
                logger.info(f"Split '{split_name}': {len(split_data)} examples")
                logger.info(f"Features: {split_data.features}")

            return self.dataset

        except Exception as e:
            logger.error(f"Failed to load dataset {self.dataset_name}: {e}")
            raise

    def get_sample_data(
        self, split: str = "train", num_samples: int = 5
    ) -> Dict[str, Any]:
        """Get sample data for inspection."""
        if self.dataset is None:
            self.load_dataset()

        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not found in dataset")

        return self.dataset[split].select(
            range(min(num_samples, len(self.dataset[split])))
        )

    def get_split(self, split: str) -> Dataset:
        """Get a specific split of the dataset."""
        if self.dataset is None:
            self.load_dataset()

        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not found in dataset")

        return self.dataset[split]

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        if self.dataset is None:
            self.load_dataset()

        info = {"name": self.dataset_name, "splits": {}, "features": {}}

        for split_name, split_data in self.dataset.items():
            info["splits"][split_name] = {
                "num_examples": len(split_data),
                "features": list(split_data.features.keys()),
            }

        # Get feature info from the first split
        first_split = next(iter(self.dataset.values()))
        info["features"] = {
            feature_name: str(feature_type)
            for feature_name, feature_type in first_split.features.items()
        }

        return info
