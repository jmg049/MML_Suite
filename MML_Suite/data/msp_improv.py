from pathlib import Path
from typing import Optional, Any, Literal
import h5py
import torch
import numpy as np
import os
from modalities import Modality, add_modality
from data.base_dataset import MultimodalBaseDataset
from experiment_utils.logging import get_logger
from experiment_utils.printing import get_console
from torch.nn.utils.rnn import pad_sequence

_ = add_modality("video")

logger = get_logger()
console = get_console()


class MSACrossFoldDataset(MultimodalBaseDataset):
    VALID_SPLITS: list[Literal["train", "valid", "test"]] = ["train", "valid", "test"]
    NUM_CLASSES = 4
    AVAILABLE_MODALITIES: dict[str, Modality] = {
        "audio": Modality.AUDIO,
        "video": Modality.VIDEO,
        "text": Modality.TEXT,
    }

    def __init__(
        self,
        data_fp: str | Path,
        split: Literal["train", "valid", "test"],
        target_modality: Modality | str = Modality.MULTIMODAL,
        *,
        missing_patterns: Optional[dict[str, dict[str, float]]] = None,
        selected_patterns: Optional[list[str]] = None,
        labels_key: str = "classification_labels",
        num_classes: Optional[int] = None,
        batch_size: int = 1,
        cv_no: int = 1,
        A_type: Literal["comparE", "comparE_raw"] = "comparE_raw",
        V_type: Literal["denseface"] = "denseface",
        T_type: Literal["bert_large"] = "bert_large",
        name: Literal["MSP_IMPROV", "IEMOCAP"] = "MSP_IMPROV",
        norm_method: Literal["trn", "utt"] = "trn",
    ):
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

        self.name = name
        # Override number of classes if specified
        if num_classes is not None:
            self.NUM_CLASSES = num_classes

        super().__init__(
            split=split, selected_patterns=selected_patterns, missing_patterns=m_patterns, batch_size=batch_size
        )

        self.data_fp = Path(data_fp)
        self.labels_key = labels_key

        self.A_type = A_type
        self.V_type = V_type
        self.T_type = T_type

        self.norm_method = norm_method

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

        self.cv_no = cv_no
        self.data = self._load_data()
        self.num_samples = len(self.labels)

        if len(self.labels) == 0:
            raise ValueError(f"No samples found for split '{split}' in dataset at {self.data_fp}")

        # Set up pattern-specific indices for validation/test
        if split != "train":
            self.pattern_indices = {pattern: list(range(self.num_samples)) for pattern in self.selected_patterns}
        self.masks = self._initialise_missing_masks(self.missing_patterns, len(self))
        self.manual_collate = True
        logger.info(
            f"Initialized {self.__class__.__name__} dataset:"
            f"\n  Split: {split}"
            f"\n  Target Modality: {target_modality}"
            f"\n  Samples: {self.num_samples}"
            f"\n  Patterns: {', '.join(self.selected_patterns)}"
            f"\n  CV No: {self.cv_no}"
        )

        console.print(
            f"[bold green]Initialized {self.__class__.__name__} dataset:[/bold green]\n"
            f"  Split: {split}\n"
            f"  Target Modality: {target_modality}\n"
            f"  Samples: {self.num_samples}\n"
            f"  Patterns: {', '.join(self.selected_patterns)}\n"
            f"  CV No: {self.cv_no}"
        )

    def _load_data(self):
        self.all_A = h5py.File(os.path.join(self.data_fp, "A", f"{self.A_type}.h5"), "r")
        self.all_V = h5py.File(os.path.join(self.data_fp, "V", f"{self.V_type}.h5"), "r")
        self.all_T = h5py.File(os.path.join(self.data_fp, "L", f"{self.T_type}.h5"), "r")

        if self.A_type == "comparE":
            self.mean_std = h5py.File(os.path.join(self.data_fp, "A", "comparE_mean_std.h5"))

            self.mean = torch.from_numpy(self.mean_std[str(self.cv_no)]["mean"][()]).unsqueeze(0).float()
            self.std = torch.from_numpy(self.mean_std[str(self.cv_no)]["std"][()]).unsqueeze(0).float()
        elif self.A_type == "comparE_raw":
            self.mean, self.std = self.calc_mean_std()

        # load target
        label_path = os.path.join(self.data_fp / "target", f"{self.cv_no}", f"{self.split}_label.npy")

        int2name_path = os.path.join(self.data_fp / "target", f"{self.cv_no}", f"{self.split}_int2name.npy")
        self.labels = np.load(label_path)
        if self.name == "IEMOCAP":
            self.labels = np.argmax(self.labels, axis=1)

        self.int2name = np.load(int2name_path)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.num_samples if self.split == "train" else self.num_samples * len(self.selected_patterns)

    def __getitem__(self, idx):
        _data = super().__getitem__(idx)

        pattern_name, sample_idx = _data.pop("pattern"), _data.pop("sample_idx")
        self.current_pattern = pattern_name
        int2name = self.int2name[sample_idx]
        if self.name.lower()     == "IEMOCAP".lower():
            int2name = int2name[0].decode()

        if self.name.lower() == "iemocap":
            l = np.where(self.labels[sample_idx] == 1)
            l = l[0]
            label = torch.tensor(l)
        else:
            label = torch.tensor(self.labels[sample_idx])
        sample = {
            "label": label,
            "pattern_names": pattern_name,
            "sample_idx": sample_idx,
            **_data,
        }
        def _audio():
            audio = torch.from_numpy(self.all_A[int2name][()]).float()
            if self.A_type == "comparE" or self.A_type == "comparE_raw":
                audio = self.normalize_on_utt(audio) if self.norm_method == "utt" else self.normalize_on_trn(audio)

            return audio.float()

        modality_loaders = {
            "audio": (lambda: _audio(), Modality.AUDIO),
            "video": (lambda: torch.from_numpy(self.all_V[int2name][()]).float(), Modality.VIDEO),
            "text": (lambda: torch.from_numpy(self.all_T[int2name][()]).float(), Modality.TEXT),
        }

        sample = self.get_samples(sample=sample, modality_loaders=modality_loaders)
        return sample

    def calc_mean_std(self):
        utt_ids = [utt_id for utt_id in self.all_A.keys()]
        feats = np.array([self.all_A[utt_id] for utt_id in utt_ids])
        _feats = feats.reshape(-1, feats.shape[2])
        mean = np.mean(_feats, axis=0)
        std = np.std(_feats, axis=0)
        std[std == 0.0] = 1.0
        return mean, std

    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features

    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate a batch of samples with pattern-aware batching.

        Args:
            batch (List[Dict[str, Any]]): List of samples.

        Returns:
            Dict[str, Any]: Collated batch of samples.
        """
        batch = self._collate_train_batch(batch)
        return batch
        # return self._collate_eval_batch(batch) if self.split != "train" else self._collate_train_batch(batch)

    def _collate_train_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
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
            "sample_idx": torch.tensor([b["sample_idx"] for b in batch]),
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

    def _collate_eval_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate evaluation batch with pattern-specific grouping.

        Args:
            batch (List[Dict[str, Any]]): List of evaluation samples.

        Returns:
            Dict[str, Any]: Collated batch grouped by patterns.
        """
        pattern_groups = {}
        for b in batch:
            pattern = b["pattern_name"]
            pattern_groups.setdefault(pattern, []).append(b)

        x=  {pattern: self._collate_train_batch(group) for pattern, group in pattern_groups.items()}

        return x


class IEMOCAP(MSACrossFoldDataset):
    @staticmethod
    def get_num_classes(is_classification: bool = True) -> int:
        """
        Get the number of classes for the task.

        Args:
            is_classification (bool): Whether the task is classification.

        Returns:
            int: Number of classes.
        """

        return 4

    @staticmethod
    def get_no_cv() -> int:
        return 10


class MSP_IMPROV(MSACrossFoldDataset):
    @staticmethod
    def get_num_classes(is_classification: bool = True) -> int:
        """
        Get the number of classes for the task.

        Args:
            is_classification (bool): Whether the task is classification.

        Returns:
            int: Number of classes.
        """
        return 4

    @staticmethod
    def get_no_cv() -> int:
        return 12
