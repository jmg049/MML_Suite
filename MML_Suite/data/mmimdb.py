from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import h5py as h5
import torch
from data.base_dataset import MultimodalBaseDataset
from modalities import Modality
from experiment_utils.logging import get_logger

logger = get_logger()


class MMIMDb(MultimodalBaseDataset):
    """
    Dataset class for the MMIMDb dataset, stored in an HDF5 file.

    This class provides methods to load multimodal data for training and evaluation
    tasks, supporting dynamic missing modality patterns.

    For more details about the dataset, visit:
    https://github.com/johnarevalo/gmu-mmimdb
    """

    # Valid dataset splits
    VALID_SPLITS: list[Literal["train", "val", "test"]] = ["train", "val", "test"]
    # Number of output classes for the dataset
    NUM_CLASSES: int = 23
    # Available modalities and their mapping to Modality enum
    AVAILABLE_MODALITIES: Dict[str, Modality] = {"image": Modality.IMAGE, "text": Modality.TEXT}

    @staticmethod
    def get_full_modality() -> str:
        """
        Get the concatenated string representation of all available modalities.

        Returns:
            str: Sorted concatenation of the first letter of all available modality keys.
        """
        modality_keys = [k[0] for k in MMIMDb.AVAILABLE_MODALITIES.keys()]
        modality_keys.sort()
        return "".join(modality_keys)

    def __init__(
        self,
        data_fp: str | Path | PathLike,
        split: Literal["train", "val", "test"],
        target_modality: Modality = Modality.MULTIMODAL,
        *,
        missing_patterns: Optional[Dict[str, Dict[str, float]]] = None,
        selected_patterns: Optional[list[str]] = None,
        image_key: str = "vgg_features",
        text_key: str = "features",
        labels_key: str = "genres",
        imdb_ids_key: str = "imdb_ids",
        missing_strategy: Literal["noise", "zero"] = "zero",
        split_indices: Optional[List[int]] = None,
        _id: int = 1,
    ):
        """
        Initialize the MMIMDb dataset.

        Args:
            data_fp (PathLike): Path to the HDF5 file containing the dataset.
            split (str): Dataset split ("train", "val", "test").
            target_modality (Modality | str): Target modality for the task.
            missing_patterns (Optional[Dict[str, Dict[str, float]]]): Missing modality patterns.
            selected_patterns (Optional[list[str]]): Selected missing patterns for evaluation.
            image_key (str): Key for image features in the HDF5 file.
            text_key (str): Key for text features in the HDF5 file.
            labels_key (str): Key for labels in the HDF5 file.
            imdb_ids_key (str): Key for IMDb IDs in the HDF5 file.
        """
        m_patterns = missing_patterns or {
            "it": {Modality.IMAGE: 1.0, Modality.TEXT: 1.0},  # Both modalities present
            "i": {Modality.IMAGE: 1.0, Modality.TEXT: 0.0},  # Image only
            "t": {Modality.IMAGE: 0.0, Modality.TEXT: 1.0},  # Text only
        }

        modality_stats = {
            Modality.IMAGE: {"mean": 0.3381, "std": 0.9204},
            Modality.TEXT: {"mean": -0.0009, "std": 0.0405},
        }

        super().__init__(
            split=split,
            selected_patterns=selected_patterns,
            missing_patterns=m_patterns,
            _id=_id,
            missing_strategy=missing_strategy,
            modality_stats=modality_stats,
        )

        assert split in self.VALID_SPLITS, f"Invalid split provided, must be one of {self.VALID_SPLITS}"

        self.split = split

        data_fp = Path(data_fp)
        if not data_fp.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_fp}")
        self.data = h5.File(Path(data_fp), "r")

        assert isinstance(target_modality, Modality), "Invalid modality provided, must be a Modality or string"
        assert target_modality in [
            Modality.TEXT,
            Modality.IMAGE,
            Modality.MULTIMODAL,
        ], "Invalid modality provided, must be one of [text, image, multimodal]"
        self.target_modality = target_modality

        # Ensure required keys exist in the dataset
        assert imdb_ids_key in self.data.keys(), f"IMDb IDs key {imdb_ids_key} not found in the dataset"
        assert image_key in self.data.keys(), f"Image key {image_key} not found in the dataset"
        assert text_key in self.data.keys(), f"Text key {text_key} not found in the dataset"
        assert labels_key in self.data.keys(), f"Labels key {labels_key} not found in the dataset"

        self.imdb_ids = imdb_ids_key
        self.text_features = text_key
        self.image_features = image_key
        self.labels = labels_key

        self.num_samples = len(self.data[self.imdb_ids])
        # Set up pattern-specific indices for validation/test
        if split != "train":
            self.pattern_indices = {pattern: list(range(self.num_samples)) for pattern in self.selected_patterns}
        self.masks = self._initialise_missing_masks(self.missing_patterns, len(self))

        logger.info(
            f"Initialized MM-IMDb dataset:"
            f"\n  Split: {split}"
            f"\n  Target Modality: {target_modality}"
            f"\n  Samples: {self.num_samples}"
            f"\n  Patterns: {', '.join(self.selected_patterns)}"
        )

    def _load_image(self, idx: int) -> torch.Tensor:
        """
        Load image features for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            torch.Tensor: Image features tensor.
        """
        return torch.as_tensor(self.data[self.image_features][idx]).float()

    def _load_text(self, idx: int) -> torch.Tensor:
        """
        Load text features for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            torch.Tensor: Text features tensor.
        """
        return torch.as_tensor(self.data[self.text_features][idx]).float()

    def _load_label(self, idx: int) -> torch.Tensor:
        """
        Load labels for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            torch.Tensor: Label tensor.
        """
        return torch.as_tensor(self.data[self.labels][idx]).float()

    def _load_id(self, idx: int) -> str:
        """
        Load the IMDb ID for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            str: IMDb ID string.
        """
        return self.data[self.imdb_ids][idx].decode("utf-8")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a dataset sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, Any]: A dictionary containing sample data and metadata.
        """
        _data = super().__getitem__(idx=idx)
        pattern_name, idx = _data.pop("pattern"), _data.pop("sample_idx")

        self.current_pattern = pattern_name
        label = self._load_label(idx)
        sample = {
            "label": label,
            "pattern_name": pattern_name,
            "missing_mask": {},
            "sample_idx": idx,
            **_data,
        }
        modality_loaders = {
            "image": (lambda: self._load_image(idx), Modality.IMAGE),
            "text": (lambda: self._load_text(idx), Modality.TEXT),
        }

        sample = self.get_samples(sample, modality_loaders)
        return sample

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.num_samples if self.split == "train" else self.num_samples * len(self.selected_patterns)
