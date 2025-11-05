from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import chain, combinations
from pathlib import Path
import re
import traceback
from typing import Any, Dict, List, Optional, Set

from data.base_dataset import MultimodalBaseDataset
from config.base_config import BaseConfig
from experiment_utils.printing import get_console
from experiment_utils.logging import get_logger
from experiment_utils.utils import format_path_with_env
from torch.utils.data import DataLoader
from modalities import Modality
from inspect import signature
from .resolvers import resolve_dataset_name

logger = get_logger()
console = get_console()


@dataclass
class ModalityConfig:
    """Configuration for a single modality's missing rate"""

    missing_rate: float = 0.0
    apply_to: Optional[List[str]] = None

    def __post_init__(self):
        if not 0 <= self.missing_rate <= 1:
            raise ValueError(f"Missing rate must be between 0 and 1, got {self.missing_rate}")

    def should_apply_to(self, pattern: str) -> bool:
        """Check if the missing rate should be applied to the given pattern. Does not apply  by default"""
        if self.apply_to is None:
            return False

        return pattern in self.apply_to


@dataclass
class MissingPatternConfig:
    """Configuration for missing patterns"""

    modalities: Dict[Modality, ModalityConfig] = field(default_factory=OrderedDict)
    selected_patterns: Optional[List[str]] = None

    def __post_init__(self):
        ## Sort to ensure that the patterns are always in the same order
        if self.selected_patterns:
            self.selected_patterns = ["".join(sorted(p)) for p in self.selected_patterns]

    @property
    def available_modalities(self) -> Set[str]:
        """Get set of available modalities"""
        return set(self.modalities.keys()) | {"multimodal"}

    def generate_patterns(self) -> Dict[str, Dict[str, float]]:
        """Generate all possible missing patterns"""
        base_mods = set(self.modalities.keys())
        all_mods = base_mods

        # Generate powerset excluding empty set
        mod_combinations = chain.from_iterable(combinations(base_mods, r) for r in range(1, len(base_mods) + 1))

        mod_combinations = list(mod_combinations)
        mod_combinations = sorted(mod_combinations, key=lambda x: (len(x), x))
        full_combination = mod_combinations[-1]
        full_combination = sorted(full_combination)
        full_combination = "".join(str(m)[0] for m in full_combination)

        patterns = {}

        for combo in mod_combinations:
            combo = sorted(combo)
            pattern_name = "".join(str(m)[0] for m in combo)

            probs = {}
            for modality in all_mods:
                if modality in combo:
                    ## if the modality is in the apply_to list, apply the missing rate from the config
                    if self.modalities[modality].should_apply_to(pattern_name):
                        rate = self.modalities[modality].missing_rate
                        probs[modality] = round(1.0 - rate, 4)
                    ## Do not apply the missing rate, 100% chance of being selected
                    else:
                        probs[modality] = 1.0
                # Absent modality - always mask
                else:
                    probs[modality] = 0.0

            patterns[pattern_name] = probs

        # Add multimodal pattern
        multimodal_probs = {}
        for modality in all_mods:
            rate = self.modalities[modality].missing_rate
            multimodal_probs[modality] = round(1.0 - rate, 4)

        patterns[full_combination] = multimodal_probs

        # Filter patterns if selected_patterns is specified
        if self.selected_patterns:
            patterns = {k: v for k, v in patterns.items() if k in self.selected_patterns}

        return patterns


@dataclass
class DatasetConfig(BaseConfig):
    """Enhanced dataset configuration with missing pattern support"""

    # Existing fields
    dataset: str
    data_fp: str
    target_modality: str
    split: str
    batch_size: int
    client_batch_size: Optional[int] = None
    shuffle: bool = False
    pin_memory: bool = False
    drop_last: bool = False
    num_workers: int = 0
    selected_missing_types: Optional[List[str]] = None
    cv_no: Optional[int] = None
    use_collate_fn: bool = False
    kwargs: Dict[str, Any] = field(default_factory=dict)

    missing_patterns: Optional[MissingPatternConfig] = None

    def __post_init__(self):
        """Validate configuration on initialization."""
        self.data_fp = format_path_with_env(self.data_fp)
        self._validate_config()

        # Initialize default missing pattern config if none provided
        if self.missing_patterns is None:
            self.missing_patterns = MissingPatternConfig()

    def get_dataset_args(self) -> Dict[str, Any]:
        """Extract dataset specific arguments."""
        args = {
            "data_fp": self.data_fp,
            "split": self.split,
            "target_modality": self.target_modality,
            "cv_no": self.cv_no,
        }

        # Add missing pattern configuration if available
        if self.missing_patterns:
            args.update(
                {
                    "missing_patterns": self.missing_patterns.generate_patterns(),
                    "selected_patterns": self.missing_patterns.selected_patterns,
                }
            )

        # Add any additional kwargs
        args.update(self.kwargs)

        logger.info(f"Dataset arguments: {args}")
        return args

    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        data_loader_str = "\n".join(f"{key}={getattr(self, key)}" for key in self.get_dataloader_args())
        dataset_str = f"dataset={self.dataset}\ndata_fp={self.data_fp}\nsplit={self.split}\ntarget_modality={self.target_modality}"
        missing_str = f"missing_patterns={self.missing_patterns}" if self.missing_patterns else ""

        return f"{dataset_str}\n{data_loader_str}\n{missing_str}"

    def _validate_config(self) -> None:
        """Validate the dataset configuration."""
        try:
            # Validate data path
            data_path = Path(self.data_fp)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_fp}")

            # Validate dataset class

            self._dataset_cls: type[MultimodalBaseDataset] = resolve_dataset_name(self.dataset)

            logger.info(f"Validated dataset class: {self.dataset}")
            console.print(f"[green]✓[/] Dataset class verified: {self._dataset_cls.__name__}")

        except Exception as e:
            error_msg = f"Configuration validation failed: {str(e)}"
            logger.error(error_msg)
            console.print(f"[red]✗[/] {error_msg}")
            raise

    def get_dataloader_args(self) -> Dict[str, Any]:
        """Extract DataLoader specific arguments."""
        args = {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
            "use_collate_fn": self.use_collate_fn,
        }

        logger.debug(f"DataLoader arguments: {args}")

        return args

    def build_dataset(self, batch_size: int) -> MultimodalBaseDataset:
        """Build the dataset instance."""
        try:
            dataset_args = self.get_dataset_args()

            dataset_init_args = str(signature(self._dataset_cls.__init__))
            if "batch_size" in dataset_init_args:
                dataset_args["batch_size"] = batch_size
            
            dataset_args = dataset_args | self.kwargs
            
            dataset = self._dataset_cls(**dataset_args)
            logger.info(f"Created {self._dataset_cls.__name__} dataset for {self.split} split")
            console.print(f"[green]✓[/] Created dataset: {self._dataset_cls.__name__} ({len(dataset)} with missing) | ({len(dataset) /( len(dataset.selected_patterns) if not re.search("train|trn", dataset.split) else 1)} base number of samples.) ")
            return dataset
        except Exception as e:
            error_msg = f"Failed to create dataset: {traceback.format_exc()}"
            logger.error(error_msg)
            console.print(f"[red]✗[/] {error_msg}")
            raise e


@dataclass
class DataConfig(BaseConfig):
    """Enhanced data configuration with advanced builder patterns."""

    datasets: Dict[str, DatasetConfig]
    default_batch_size: int = 32
    use_collate_fn: bool = False

    def __post_init__(self):
        """Initialize and validate the configuration."""
        self._validate_configs()

    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        return "\n".join(str(config) for config in self.datasets.values())

    def _validate_configs(self) -> None:
        """Validate all dataset configurations."""
        if not self.datasets:
            raise ValueError("No datasets configured")

        for name, config in self.datasets.items():
            try:
                if not isinstance(config, DatasetConfig):
                    self.datasets[name] = DatasetConfig.from_dict(config)
                logger.info(f"Validated configuration for {name} dataset")
            except Exception as e:
                error_msg = f"Invalid configuration for {name} dataset: {str(e)}"
                logger.error(error_msg)
                console.print(f"[red]✗[/] {error_msg}")
                raise

    def build_dataloader(
        self,
        target_split: str,
    ) -> DataLoader:
        """
        Build a DataLoader for the specified split with enhanced error handling
        and logging.

        Args:
            target_split: The split to build the DataLoader for
            batch_size: Optional batch size override
            print_fn: Function to use for printing status messages

        Returns:
            DataLoader instance for the specified split
        """
        if target_split not in self.datasets:
            raise KeyError(f"Split '{target_split}' not found in configuration")

        try:
            dataset_config = self.datasets[target_split]
            dataloader_args = dataset_config.get_dataloader_args()
            batch_size = dataloader_args.get("batch_size", self.default_batch_size)
            # Build the dataset
            dataset = dataset_config.build_dataset(batch_size)

            console.print(f"[blue]Use collate_fn: {self.use_collate_fn}[/]")
            console.print(f"[blue]Has custom collate_fn: {hasattr(dataset, 'collate_fn')}[/]")

            if self.use_collate_fn and hasattr(dataset, "collate_fn"):
                console.print(f"[yellow]Using custom collate_fn for {target_split} split[/]")
                dataloader_args["collate_fn"] = dataset.collate_fn

            try:
                del dataloader_args["use_collate_fn"]
            except KeyError:
                pass

            # Create the DataLoader
            dataloader = DataLoader(dataset, **dataloader_args)

            # Log success
            logger.info(f"Created DataLoader for {target_split} split " f"(batch_size={dataloader_args['batch_size']})")
            console.print(f"[green]✓[/] Created DataLoader for {target_split} split")

            return dataloader

        except Exception as e:
            error_msg = f"Failed to build DataLoader for {target_split}: {str(e)}"
            logger.error(error_msg)
            console.print(f"[red]✗[/] {error_msg}")
            raise e

    def build_all_dataloaders(self, is_train: bool = True, is_test: bool = True) -> Dict[str, DataLoader]:
        """Build DataLoaders for all configured splits."""
        dataloaders = {}
        for split in self.datasets:
            if (split == "train" or split == "trn" or split == "validation") and not is_train:
                continue

            if split == "test" and not is_test:
                continue

            try:
                dataloaders[split] = self.build_dataloader(split)
            except Exception as e:
                logger.error(f"Failed to build DataLoader for {split}: {str(e)}")
                raise e
        return dataloaders
    

    def build_all_datasets(self) -> Dict[str, MultimodalBaseDataset]:
        """Build all datasets based on the configuration."""
        datasets = {}
        for name, config in self.datasets.items():
            try:
                datasets[name] = config.build_dataset(self.default_batch_size)
            except Exception as e:
                logger.error(f"Failed to build dataset for {name}: {str(e)}")
                raise e
        return datasets
