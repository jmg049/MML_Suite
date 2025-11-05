import pickle
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from data.base_dataset import MultimodalBaseDataset
from experiment_utils.logging import get_logger
from modalities import Modality, add_modality
from torch.nn.utils.rnn import pad_sequence
from experiment_utils.printing import get_console
console = get_console()
logger = get_logger()

add_modality("video")


class MultimodalSentimentDataset(MultimodalBaseDataset):
    """
    Base class for CMU-MOSI and CMU-MOSEI datasets with missing modality support.

    Supports audio, video, and text modalities with customizable missing patterns,
    aligned or unaligned data, and pattern-specific data loading.
    """

    VALID_SPLITS: List[str] = ["train", "valid", "test"]
    NUM_CLASSES: int = 3  # Assumes 3-way classification [positive, negative, neutral]
    AVAILABLE_MODALITIES: Dict[str, Modality] = {
        "audio": Modality.AUDIO,
        "video": Modality.VIDEO,
        "text": Modality.TEXT,
    }

    def __init__(
        self,
        data_fp: Path | PathLike,
        split: str,
        target_modality: Modality | str = Modality.MULTIMODAL,
        *,
        missing_patterns: Optional[Dict[str, Dict[str, float]]] = None,
        selected_patterns: Optional[List[str]] = None,
        missing_strategy: Literal["zero", "noise"] = "zero",
        labels_key: str = "classification_labels",
        aligned: bool = False,
        length: Optional[int] = None,
        num_classes: Optional[int] = None,
        batch_size: int = 1,
        **kwargs
    ) -> None:
        """
        Initialize the Multimodal Sentiment Dataset.

        Args:
            data_fp (PathLike): Path to the data file.
            split (str): Dataset split ('train', 'valid', 'test').
            target_modality (Modality | str): Target modality for the task.
            missing_patterns (Optional[Dict[str, Dict[str, float]]]): Dictionary of missing patterns.
            selected_patterns (Optional[List[str]]): Selected patterns for evaluation.
            labels_key (str): Key to access labels in the data.
            aligned (bool): Whether the data is aligned across modalities.
            length (Optional[int]): Length of aligned sequences.
            num_classes (Optional[int]): Number of output classes (overrides default).
        """
        # Set up missing patterns
        m_patterns = missing_patterns or {
            "atv": {Modality.AUDIO: 1.0, Modality.TEXT: 1.0, Modality.VIDEO: 1.0},
            "at": {Modality.AUDIO: 1.0, Modality.TEXT: 1.0, Modality.VIDEO: 0.0},
            "av": {Modality.AUDIO: 1.0, Modality.TEXT: 0.0, Modality.VIDEO: 1.0},
            "tv": {Modality.AUDIO: 0.0, Modality.TEXT: 1.0, Modality.VIDEO: 1.0},
            "a": {Modality.AUDIO: 1.0, Modality.TEXT: 0.0, Modality.VIDEO: 0.0},
            "t": {Modality.AUDIO: 0.0, Modality.TEXT: 1.0, Modality.VIDEO: 0.0},
            "v": {Modality.AUDIO: 0.0, Modality.TEXT: 0.0, Modality.VIDEO: 1.0},
        }

        modality_stats = {
            Modality.AUDIO: {"mean": 0.6085, "std": 11.3582},
            Modality.VIDEO: {"mean": -0.2222, "std": 1.3391},
            Modality.TEXT: {"mean": 0.059, "std": 0.3493},
        }

        # Override number of classes if specified
        if num_classes is not None:
            self.NUM_CLASSES = num_classes

        super().__init__(
            split=split,
            selected_patterns=selected_patterns,
            missing_patterns=m_patterns,
            batch_size=batch_size,
            modality_stats=modality_stats,
            missing_strategy=missing_strategy,
        )

        self.data_fp = Path(data_fp)
        self.aligned = aligned
        self.length = length if aligned else None
        self.labels_key = labels_key
        # Process target modality
        if isinstance(target_modality, str):
            target_modality = Modality.from_str(target_modality)
        assert isinstance(
            target_modality, Modality
        ), f"Invalid modality provided, must be a Modality instance, not {type(target_modality)}"
        assert (
            target_modality in self.AVAILABLE_MODALITIES.values() or target_modality == Modality.MULTIMODAL
        ), f"Invalid target modality provided, must be one of {list(self.AVAILABLE_MODALITIES.values())}"
        self.target_modality = target_modality

        # Load and validate data
        self.data = self._load_data(labels_key)
        self.num_samples = len(self.data["label"])
        self.labels = self.get_labels()
        # Set up pattern-specific indices for validation/test
        if split != "train":
            self.pattern_indices = {pattern: list(range(self.num_samples)) for pattern in self.selected_patterns}
        self.masks = self._initialise_missing_masks(self.missing_patterns, len(self))

        logger.info(
            f"Initialized {self.__class__.__name__} dataset:"
            f"\n  Split: {split}"
            f"\n  Target Modality: {target_modality}"
            f"\n  Samples: {self.num_samples}"
            f"\n  Patterns: {', '.join(self.selected_patterns)}"
        )

    def get_labels(self) -> np.ndarray:
        """
        Get the labels from the dataset.

        Returns:
            np.ndarray: Array of labels.
        """
        if "label" not in self.data:
            raise KeyError(f"Labels key 'labels' not found in data, available keys: {list(self.data.keys())}")
        return self.data["label"].numpy()

    def _load_data(self, labels_key: str) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess data from a pickle file.

        Args:
            labels_key (str): Key to access labels in the data.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of tensors for each modality and labels.
        """
        if not self.data_fp.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_fp}")

        with open(self.data_fp, "rb") as f:
            raw_data = pickle.load(f)

        if self.split not in raw_data:
            raise KeyError(f"Split '{self.split}' not found in data")

        split_data = raw_data[self.split]
        if labels_key not in split_data:
            raise KeyError(f"Labels key '{labels_key}' not found in data")

        core_data = {
            Modality.AUDIO: torch.tensor(split_data["audio"]).float(),
            Modality.VIDEO: torch.tensor(split_data["vision"]).float(),
            Modality.TEXT: torch.tensor(split_data["text"]).float(),
            "label": torch.tensor(
                split_data[labels_key], dtype=torch.float32 if "regression" in self.labels_key else torch.long
            ),
        }

        self.original_label_size = core_data["label"].size(0)

        return (
            core_data
            if self.aligned
            else core_data
            | {
                "audio_lengths": torch.tensor(split_data["audio_lengths"]).float(),
                "video_lengths": torch.tensor(split_data["vision_lengths"]).float(),
            }
        )

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        # Always use original patterns for length calculation to maintain DataLoader consistency
        # Temporary patterns don't change the dataset length - they just change what patterns are returned
        return self.num_samples if self.split == "train" else self.num_samples * len(self.selected_patterns)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a dataset sample by index with missing pattern applied.

        Args:
            idx (int): Dataset index.

        Returns:
            Dict[str, Any]: A dictionary containing the sample data and metadata.
        """

        _data = super().__getitem__(idx)

        pattern_name, sample_idx = _data.pop("pattern"), _data.pop("sample_idx")
        self.current_pattern = pattern_name

        sample = {
            "label": self.data["label"][sample_idx],
            "pattern_names": pattern_name,
            # "missing_index": {},
            "sample_idx": sample_idx,
            **_data,
        }

        if not self.aligned:
            sample["audio_length"] = self.data["audio_lengths"][sample_idx]
            sample["video_length"] = self.data["video_lengths"][sample_idx]

        modality_loaders = {
            "audio": (lambda: self.data[Modality.AUDIO][sample_idx], Modality.AUDIO),
            "video": (lambda: self.data[Modality.VIDEO][sample_idx], Modality.VIDEO),
            "text": (lambda: self.data[Modality.TEXT][sample_idx], Modality.TEXT),
        }

        sample = self.get_samples(sample=sample, modality_loaders=modality_loaders)
        return sample

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples with pattern-aware batching.

        Args:
            batch (List[Dict[str, Any]]): List of samples.

        Returns:
            Dict[str, Any]: Collated batch of samples.
        """
        return self._collate_train_batch(batch)

    def _collate_train_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate training batch with mixed patterns.

        Args:
            batch (List[Dict[str, Any]]): List of training samples.

        Returns:
            Dict[str, Any]: Collated batch.
        """
        collated = {
            "label": torch.stack([b["label"] for b in batch]),
            "pattern_names": [b["pattern_names"] for b in batch],
            # "missing_index": [b["missing_index"] for b in batch],
            "sample_idx": [b["sample_idx"] for b in batch],
        }

        for mod_enum in self.AVAILABLE_MODALITIES.values():
            sequences = [b.get(mod_enum) for b in batch if mod_enum in b]
            collated[mod_enum] = pad_sequence(sequences, batch_first=True, padding_value=0) if sequences else None
            collated[f"{mod_enum}_original"] = pad_sequence(
                [b.get(f"{mod_enum}_original") for b in batch if f"{mod_enum}_original" in b],
                batch_first=True,
                padding_value=0,
            ) if f"{mod_enum}_original" in batch[0] else None
            collated[f"{mod_enum}_reverse"] = pad_sequence(
                [b.get(f"{mod_enum}_reverse") for b in batch if f"{mod_enum}_reverse" in b],
                batch_first=True,
                padding_value=0,
            ) if f"{mod_enum}_reverse" in batch[0] else None
            collated[f"{mod_enum}_missing_index"] = torch.stack([b.get(f"{mod_enum}_missing_index") for b in batch if f"{mod_enum}_missing_index" in b])

        return collated

    def _collate_eval_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate evaluation batch with pattern-specific grouping.

        Args:
            batch (List[Dict[str, Any]]): List of evaluation samples.

        Returns:
            Dict[str, Any]: Collated batch grouped by patterns.
        """
        pattern_groups = {}
        for b in batch:
            pattern = b["pattern_names"]
            pattern_groups.setdefault(pattern, []).append(b)

        return {pattern: self._collate_train_batch(group) for pattern, group in pattern_groups.items()}

    @staticmethod
    def normalize_features(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalize features along the time dimension.

        Args:
            features (torch.Tensor): Input features.
            eps (float): Small value to prevent division by zero.

        Returns:
            torch.Tensor: Normalized features.
        """
        mean = torch.mean(features, dim=0, keepdim=True)
        std = torch.std(features, dim=0, keepdim=True).clamp(min=eps)
        return (features - mean) / std


class MOSEI(MultimodalSentimentDataset):
    """CMU-MOSEI dataset implementation."""

    @staticmethod
    def get_num_classes(is_classification: bool = True) -> int:
        """
        Get the number of classes for the task.

        Args:
            is_classification (bool): Whether the task is classification.

        Returns:
            int: Number of classes.
        """
        return 3 if is_classification else 1


class MOSI(MultimodalSentimentDataset):
    """CMU-MOSI dataset implementation."""

    @staticmethod
    def get_num_classes(is_classification: bool = True) -> int:
        """
        Get the number of classes for the task.

        Args:
            is_classification (bool): Whether the task is classification.

        Returns:
            int: Number of classes.
        """
        return 3 if is_classification else 1
