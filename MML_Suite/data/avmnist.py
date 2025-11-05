from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from data.base_dataset import MultimodalBaseDataset
from data.pattern import PatternSpecificDataset
from experiment_utils.logging import get_logger
from matplotlib import cm
from modalities import Modality
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import PILToTensor, ToDtype
from experiment_utils.utils import get_console

console  = get_console()


logger = get_logger()


class AVMNIST(MultimodalBaseDataset):
    """
    Dataset class for the audio-visual MNIST dataset with support for missing modality patterns.

    This dataset supports both training and evaluation splits with customizable missing patterns and
    dynamic masking for audio and image modalities.
    """

    NUM_CLASSES: int = 10
    VALID_SPLITS: List[Literal["train", "valid", "test"]] = ["train", "valid", "test"]
    AVAILABLE_MODALITIES: Dict[str, Modality] = {"audio": Modality.AUDIO, "image": Modality.IMAGE}

    @staticmethod
    def get_full_modality() -> str:
        """
        Get the concatenated string representation of all available modalities.

        Returns:
            str: Sorted concatenation of the first letters of all available modality keys.
        """
        modality_keys = [k[0] for k in AVMNIST.AVAILABLE_MODALITIES.keys()]
        modality_keys.sort()
        return "".join(modality_keys)

    def __init__(
        self,
        data_fp: Path | PathLike,
        split: str,
        target_modality: Modality | str = Modality.MULTIMODAL,
        *,
        missing_patterns: Optional[Dict[str, Dict[str, float]]] = None,
        selected_patterns: Optional[List[str]] = None,
        missing_strategy: Literal["zero", "noise"] = "zero",
        audio_column: str = "audio",
        image_column: str = "image",
        labels_column: str = "label",
        split_indices: Optional[List[int]] = None,
        _id: int = 1,
        **kwargs,
    ) -> None:
        """
        Initialize the AVMNIST dataset.

        Args:
            data_fp (PathLike): Path to the data CSV file.
            split (str): Dataset split ("train", "valid", or "test").
            target_modality (Modality | str): Target modality for the dataset.
            missing_patterns (Optional[Dict[str, Dict[str, float]]]): Dict of pattern configurations.
            selected_patterns (Optional[List[str]]): List of selected patterns for evaluation.
            audio_column (str): Name of the audio column in the CSV.
            image_column (str): Name of the image column in the CSV.
            labels_column (str): Name of the labels column in the CSV.
            split_indices (Optional[List[int]]): Optional indices for dataset splitting.
        """
        m_patterns = missing_patterns or {
            "ai": {Modality.AUDIO: 1.0, Modality.IMAGE: 1.0},  # Both modalities present
            "a": {Modality.AUDIO: 1.0, Modality.IMAGE: 0.0},  # Audio only
            "i": {Modality.AUDIO: 0.0, Modality.IMAGE: 1.0},  # Image only
        }

        assert split in AVMNIST.VALID_SPLITS, f"Invalid split provided, must be one of {AVMNIST.VALID_SPLITS}"

        # train
        #  IMAGE → mean = 0.0857,  std = 0.2487
        #  AUDIO → mean = 97562.3273,  std = 627670.5328

        # validation
        # IMAGE → mean = 0.0889,  std = 0.2532
        # AUDIO → mean = 97426.6131,  std = 624061.4419

        # test
        #  IMAGE → mean = 0.0876,  std = 0.2511
        #  AUDIO → mean = 91291.6249,  std = 603451.1890

        ## Only used for noisy training
        modality_stats = {
            Modality.AUDIO: {"mean": 97562.3273, "std": 627670.5328},
            Modality.IMAGE: {"mean": 0.0857, "std": 0.2487},
        }
    
        super().__init__(split=split, selected_patterns=selected_patterns, missing_patterns=m_patterns, _id=_id, missing_strategy=missing_strategy, modality_stats=modality_stats)



        self.data_fp = Path(data_fp)
        if not self.data_fp.exists():
            raise FileNotFoundError(f"Data file not found: {data_fp}")

        self.split = split
        self.audio_column = audio_column
        self.image_column = image_column
        self.labels_column = labels_column

        # Set up transforms
        self.transforms = {
            "pil_to_tensor": PILToTensor(),
            "scale": ToDtype(torch.float32, scale=True),
        }

        # Load and process data
        self._load_data(split_indices)


        self.num_samples = len(self.data)

        # Set up pattern-specific indices for validation/test
        if split != "trn":
            self.pattern_indices = {pattern: list(range(self.num_samples)) for pattern in self.selected_patterns}
        self.masks = self._initialise_missing_masks(self.missing_patterns, len(self))

        logger.info(
            f"Initialized AVMNIST dataset:"
            f"\n  Split: {split}"
            f"\n  Target Modality: {target_modality}"
            f"\n  Samples: {self.num_samples}"
            f"\n  Patterns: {', '.join(self.selected_patterns)}"
        )

        if isinstance(target_modality, str):
            target_modality = Modality.from_str(target_modality)
        assert isinstance(
            target_modality, Modality
        ), f"Invalid modality provided, must be a Modality Enum, not {type(target_modality)}"
        assert target_modality in [
            Modality.AUDIO,
            Modality.IMAGE,
            Modality.MULTIMODAL,
        ], "Invalid modality provided, must be one of [audio, image, multimodal]"
        self.target_modality = target_modality

        logger.info(
            f"Initialized AVMNIST dataset:"
            f"\n  Split: {split}"
            f"\n  Target Modality: {target_modality}"
            f"\n  Samples: {self.num_samples}"
            f"\n  Patterns: {', '.join(self.selected_patterns)}"
        )

    def _load_data(self, split_indices: Optional[List[int]] = None) -> None:
        """
        Load and validate dataset from a CSV file.

        Args:
            split_indices (Optional[List[int]]): Optional indices for filtering rows.
        """
        self.data = pd.read_csv(self.data_fp)
        if split_indices is not None:
            self.data = self.data.iloc[split_indices].reset_index(drop=True)

        # Validate required columns
        required_columns = [self.audio_column, self.image_column, self.labels_column]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def __len__(self) -> int:
        """
        Return the total length of the dataset.

        Returns:
            int: Total number of samples.
        """
        if self.split == "train":
            return self.num_samples
        else:
            return self.num_samples * len(self.selected_patterns)

    @lru_cache(maxsize=1000)
    def _load_audio(self, path: str) -> torch.Tensor:
        """
        Load audio data from a file with caching.

        Args:
            path (str): Path to the audio file.

        Returns:
            torch.Tensor: Loaded audio data as a tensor.
        """
        return torch.load(path, weights_only=True)

    @lru_cache(maxsize=1000)
    def _load_image(self, path: str) -> torch.Tensor:
        """
        Load and process image data from a file with caching.

        Args:
            path (str): Path to the image file.

        Returns:
            torch.Tensor: Processed image data as a tensor.
        """
        img_data = np.array(torch.load(path, weights_only=False))
        img = Image.fromarray(np.uint8(cm.gist_earth(img_data) * 255)).convert("L")
        img_tensor = self.transforms["pil_to_tensor"](img)
        return self.transforms["scale"](img_tensor)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a dataset sample by index with missing pattern applied.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, Any]: A dictionary containing the sample data and metadata.
        """

        _data = super().__getitem__(idx)
        pattern_name, idx = _data.pop("pattern"), _data.pop("sample_idx")

        self.current_pattern = pattern_name
        label = self.data[self.labels_column].iloc[idx]
        label = torch.tensor(label, dtype=torch.long)
        sample = {
            "labels": label,
            "pattern_name": pattern_name,
            "missing_mask": {},
            "sample_idx": idx,
            **_data,
        }

        modality_loaders = {
            "audio": (lambda: self._load_audio(self.data[self.audio_column].iloc[idx]), Modality.AUDIO),
            "image": (lambda: self._load_image(self.data[self.image_column].iloc[idx]), Modality.IMAGE),
        }

        sample = self.get_samples(sample, modality_loaders)
        return sample

    def get_pattern_batches(self, batch_size: int, **dataloader_kwargs) -> Dict[str, DataLoader]:
        """
        Get separate DataLoaders for each pattern.

        Args:
            batch_size (int): Batch size for the DataLoader.
            **dataloader_kwargs: Additional DataLoader keyword arguments.

        Returns:
            Dict[str, DataLoader]: A dictionary of DataLoaders for each pattern.
        """
        if self.split == "train":
            raise ValueError("Pattern-specific batches only available for validation/test")

        pattern_loaders = {}
        for pattern in self.selected_patterns:
            pattern_dataset = PatternSpecificDataset(self, pattern)
            pattern_loaders[pattern] = DataLoader(
                pattern_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn, **dataloader_kwargs
            )
        return pattern_loaders

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collation function for batching.

        Args:
            batch (List[Dict[str, Any]]): List of samples.

        Returns:
            Dict[str, Any]: Collated batch of samples.
        """
        device = batch[0]["labels"].device

        collated = {
            "labels": torch.stack([b["labels"] for b in batch]),
            "pattern_name": [b["pattern_name"] for b in batch],
            "missing_masks": {
                mod: torch.tensor([b["missing_mask"][mod] for b in batch], device=device)
                for mod in [Modality.AUDIO, Modality.IMAGE]
                if mod in batch[0]["missing_mask"]
            },
            "sample_idx": torch.tensor([b["sample_idx"] for b in batch], device=device),
        }

        if self.target_modality == Modality.MULTIMODAL:
            for mod in [Modality.AUDIO, Modality.IMAGE]:
                if mod in batch[0]:
                    collated[mod] = torch.stack([b[mod] for b in batch])
        else:
            collated[self.target_modality] = torch.stack([b[self.target_modality] for b in batch])

        return collated
