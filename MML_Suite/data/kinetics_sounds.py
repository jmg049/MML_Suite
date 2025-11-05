from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from modalities import Modality, add_modality
import pandas as pd
from torch import Tensor
import torch

from data.base_dataset import MultimodalBaseDataset

_ = add_modality("VIDEO")

class KineticsSounds(MultimodalBaseDataset):
    VALID_SPLITS: list[Literal["train", "val", "test"]] = ["train", "val", "test"]
    # Number of output classes for the dataset
    NUM_CLASSES: int = 26
    # Available modalities and their mapping to Modality enum
    AVAILABLE_MODALITIES: Dict[str, Modality] = {"audio": Modality.AUDIO, "video": Modality.VIDEO}

    @staticmethod
    def get_full_modality() -> str:
        modality_keys = [k[0] for k in KineticsSounds.AVAILABLE_MODALITIES.keys()]
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
        missing_strategy: Literal["noise", "zero"] = "zero",
        audio_key: str = "audio",
        video_key: str = "video",
        labels_key: str = "label",
        split_indices: Optional[List[int]] = None,
        _id: int = 1,
        **kwargs: Any
    ):
        m_patterns = missing_patterns or {
            "av": {Modality.AUDIO: 1.0, Modality.VIDEO: 1.0},
            "a": {Modality.AUDIO: 1.0, Modality.VIDEO: 0.0},
            "v": {Modality.AUDIO: 0.0, Modality.VIDEO: 1.0},
        }

        assert (
            split in KineticsSounds.VALID_SPLITS
        ), f"Invalid split: {split}, must be one of {KineticsSounds.VALID_SPLITS}"
        self.split = split
        data_fp = Path(data_fp)
        if not data_fp.exists():
            raise FileNotFoundError(f"File not found: {data_fp}")


        modality_stats = {
            Modality.AUDIO: {"mean": 0.0, "std": 3.1276},
            Modality.VIDEO: {"mean": 160719.6702, "std": 564025.9526}
        }

        super(KineticsSounds, self).__init__(
            split=split, selected_patterns=selected_patterns, missing_patterns=m_patterns, _id=_id, modality_stats=modality_stats,
            missing_strategy=missing_strategy        )

        self.data = pd.read_csv(data_fp)
        assert isinstance(target_modality, Modality), "Invalid modality provided, must be a Modality"
        assert target_modality in [Modality.AUDIO, Modality.VIDEO, Modality.MULTIMODAL], "Invalid target modality"

        self.target_modality = target_modality
        self.audio_key = audio_key
        self.video_key = video_key
        self.labels_key = labels_key

        assert self.audio_key in self.data.columns, f"Audio key not found in the dataset: {self.audio_key}"
        assert self.video_key in self.data.columns, f"Video key not found in the dataset: {self.video_key}"
        assert self.labels_key in self.data.columns, f"Labels key not found in the dataset: {self.labels_key}"

        self.num_samples = len(self.data)
        self.masks = self._initialise_missing_masks(self.missing_patterns, len(self))

    @staticmethod
    def _load_audio(fp: str | Path | PathLike) -> Tensor:
        return torch.load(fp, weights_only=True).float()

    @staticmethod
    def _load_video(fp: str | Path | PathLike) -> Tensor:
        return torch.load(fp, weights_only=True).float()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        _data = super().__getitem__(idx)
        pattern_name, idx = _data.pop("pattern"), _data.pop("sample_idx")

        self.current_pattern = pattern_name
        label = self.data[self.labels_key].iloc[idx]
        audio_fp = self.data[self.audio_key].iloc[idx]
        video_fp = self.data[self.video_key].iloc[idx]

        sample = {
            "labels": label,
            "pattern_name": pattern_name,
            "missing_mask": {},
            "sample_idx": idx,
            **_data,
        }
        modality_loaders = {
            "audio": (lambda: self._load_audio(audio_fp), Modality.AUDIO),
            "video": (lambda: self._load_video(video_fp), Modality.VIDEO),
        }

        sample = self.get_samples(sample, modality_loaders)
        return sample

    def __len__(self) -> int:
        return self.num_samples if self.split == "train" else self.num_samples * len(self.selected_patterns)
