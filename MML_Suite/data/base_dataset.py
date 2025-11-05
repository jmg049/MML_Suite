import random
from contextlib import contextmanager
from itertools import combinations
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
from experiment_utils.printing import get_console, print_info
from experiment_utils.logging import get_logger
from experiment_utils.utils import NestedDictAccess
from modalities import Modality, create_missing_mask
from torch.utils.data import Dataset

console = get_console()
logger = get_logger()


class MultimodalBaseDataset(Dataset):
    """Base class for multimodal datasets with mi ssing modality support."""

    _ndict_accessor = NestedDictAccess(max_depth=5, logger=logger)

    def __init__(
        self,
        split: Literal["train", "valid", "test"],
        selected_patterns: Optional[List[str]] = None,
        missing_patterns: Optional[Dict[str, Dict[str, float]]] = None,
        batch_size: int = 1,
        _id: int = 1,
        missing_strategy: Literal["zero", "noise"] = "zero",
        modality_stats: Optional[Dict[Modality, Dict[str, float]]] = None,
    ) -> None:
        self.split = split.lower()
        assert split in self.VALID_SPLITS, f"Invalid split provided, must be one of {self.VALID_SPLITS}"

        self.missing_patterns = missing_patterns

        # Handle pattern selection
        if selected_patterns is not None:
            self.selected_patterns = self.validate_patterns(selected_patterns)
        else:
            self.selected_patterns = self.get_all_possible_patterns()
        self.pattern_indices = None
        self._batch_size = batch_size
        self.current_pattern = None
        
        # Dynamic pattern switching support
        self._original_selected_patterns = self.selected_patterns.copy()
        self._temp_patterns = None
        self._temp_masks = None

        self.missing_strategy = missing_strategy
        self.modality_stats = modality_stats if modality_stats else {}
        assert isinstance(_id, int), "ID must be an integer."
        self._id = _id

    def _initialise_missing_masks(self, missing_patterns, batch_size: int = 1):
        masks = {}
        if missing_patterns is not None:
            for pattern, modality_patterns in missing_patterns.items():
                _mask = create_missing_mask(
                    len(self.AVAILABLE_MODALITIES), batch_size, [1 - pct for _, pct in modality_patterns.items()]
                )
                # Create per-sample masks
                masks[pattern] = {}
                for i, (modality, _) in enumerate(modality_patterns.items()):
                    masks[pattern][modality] = _mask[:, i]  # Keep as tensor per sample
        return masks

    def get_samples(
        self,
        sample: Dict[str, Any],
        modality_loaders: Dict[str, Tuple[Callable, Modality]],
    ) -> Dict[str, Any]:
        """Load data for each modality."""
        for _mod_name, (loader_fn, mod_enum) in modality_loaders.items():
            if self.target_modality == Modality.MULTIMODAL or self.target_modality == mod_enum:
                orig = loader_fn()
                mask = sample[f"{mod_enum}_missing_index"]

                if self.missing_strategy == "zero":
                    filled = orig * mask
                else:  # Gaussian noise replacement
                    stats = self.modality_stats[mod_enum]
                    mean = stats["mean"]
                    std = stats["std"]

                    ## Create a replacement that is gaussian noise with the same mean and std
                    replacement = torch.normal(mean=mean, std=std, size=orig.shape).to(orig.device)

                    filled = orig * mask + replacement * (1 - mask)
                # assign back into the sample
                sample[f"{mod_enum}_original"] = orig
                sample[mod_enum] = filled
                # “reverse” (what you were doing before—keep it if you need it)
                sample[f"{mod_enum}_reverse"] = orig * -1 * (mask - 1)
        return sample

    def _get_pattern_and_sample_idx(self, idx: int) -> Tuple[str, int]:
        """
        Get the pattern and corresponding sample index for a given dataset index.

        Args:
            idx (int): Dataset index.

        Returns:
            Tuple[str, int]: Tuple containing the pattern name and sample index.
        """
        # Use temporary patterns if in dynamic pattern mode
        active_patterns = self._temp_patterns if self._temp_patterns is not None else self.selected_patterns
        
        if self.split == "train" or self.split == "trn":
            mp = random.choice(active_patterns)
            return mp, idx
        else:
            # For validation/test with temporary patterns, handle index mapping carefully
            if self._temp_patterns is not None:
                # When using temporary patterns, we want to return the temp pattern for all samples
                # This means that every sample will be transformed to the temporary pattern
                sample_idx = idx % self.num_samples
                return active_patterns[0], sample_idx  # Use first (and likely only) temporary pattern
            else:
                # Original logic for non-temporary patterns
                pattern_idx = idx // self.num_samples
                sample_idx = idx % self.num_samples
                return active_patterns[pattern_idx], sample_idx

    def set_pattern_indices(self, n_samples: int) -> None:
        # For validation/test, organize samples by pattern
        if self.split != "train":
            self.pattern_indices = {pattern: list(range(n_samples)) for pattern in self.selected_patterns}

    def get_split(self) -> str:
        return self.split

    def get_selected_patterns(self) -> List[str]:
        return self.selected_patterns
    
    def get_original_patterns(self) -> List[str]:
        """Get the original patterns set during initialization."""
        return self._original_selected_patterns

    def get_missing_patterns(self) -> Dict[str, Dict[str, float]]:
        return self.missing_patterns

    @staticmethod
    def get_full_modality() -> str:
        """Return the name of the full modality."""
        raise NotImplementedError("Method get_full_modality must be implemented in the derived class.")

    @classmethod
    def get_all_possible_patterns(cls) -> List[str]:
        """Generate all possible modality combinations excluding empty set."""
        modalities = list(cls.AVAILABLE_MODALITIES.keys())
        patterns = []
        for r in range(1, len(modalities) + 1):
            for combo in combinations(modalities, r):
                pattern_name = "".join(m[0] for m in sorted(combo))
                patterns.append(pattern_name)
        return sorted(patterns)

    def validate_patterns(self, patterns: List[str]) -> List[str]:
        """Validate and normalize pattern names."""
        # all_patterns = self.get_all_possible_patterns()
        # invalid_patterns = set(patterns) - set(all_patterns)
        # if invalid_patterns:
        #     raise ValueError(f"Invalid patterns: {invalid_patterns}\n" f"Valid patterns are: {all_patterns}")
        return patterns

    def set_selected_pattern(self, pattern: str) -> None:
        assert hasattr(self, "selected_pattern"), "Dataset must have attribute selected_pattern"
        assert pattern in self.get_all_possible_patterns(), "Invalid pattern"
        self.selected_pattern = pattern

    @contextmanager
    def temporary_patterns(self, patterns: List[str]):
        """
        Context manager for temporarily switching to different patterns.
        
        This allows C-MAM training to use specific patterns without permanently
        modifying the dataset configuration. The original patterns and masks
        are restored when exiting the context.
        
        Unlike the filtering approach, this actually allows the dataset to generate
        samples with the required missing patterns, even if they weren't in the
        original selected_patterns.
        
        Args:
            patterns: List of patterns to use temporarily
            
        Example:
            with dataset.temporary_patterns(["at"]):
                # Dataset now generates samples with "at" pattern from "atv" data
                for batch in dataloader:
                    # Train C-MAM that needs audio+text data
                    pass
            # Original patterns restored automatically
        """
        # print_info(console, f"Using temporary patterns: {patterns} - {type(patterns)}")
        # Validate patterns
        if isinstance(patterns, list) and isinstance(patterns[0], list):
            # print_info(console, "Flattening nested patterns list")
            patterns = patterns[0]  # Flatten the list of lists
            # print_info(console, f"Flattened patterns: {patterns} - {type(patterns)}")
        elif isinstance(patterns, list) and isinstance(patterns[0], str):
            # Patterns are already in the correct format
            pass
        elif isinstance(patterns, str):
            # print_info(console, "Converting single pattern string to list")
            patterns = [patterns]
        else:
            raise ValueError(f"Invalid patterns format: {patterns}. Must be a list of strings or a single string.")

            
        validated_patterns = self.validate_patterns(patterns)
        # print_info(console, f"Validated temporary patterns: {validated_patterns} - {type(validated_patterns)}")
        # Store current state
        original_patterns = self._temp_patterns
        original_masks = self._temp_masks
        
        try:
            # Set temporary patterns
            self._temp_patterns = validated_patterns
            # print_info(console, f"Using temporary patterns: {self._temp_patterns} - {type(self._temp_patterns)}")

            # Generate temporary masks for the new patterns
            # Key insight: Use original dataset length, not length based on temp patterns
            if self.missing_patterns is not None:
                # Create missing patterns dict that includes our temporary patterns
                temp_missing_patterns = {}
                for pattern in validated_patterns:
                    try:
                        if pattern in self.missing_patterns:
                            temp_missing_patterns[pattern] = self.missing_patterns[pattern]
                        else:
                            # Generate missing pattern configuration for new patterns
                            temp_missing_patterns[pattern] = self._generate_missing_pattern_config(pattern)
                    except Exception as e:
                        console.print(f"Error generating missing pattern config for {pattern}: {e}")
                        exit(0)
                # Use original dataset length to avoid index issues
                original_length = self.num_samples if self.split == "train" else self.num_samples * len(self.selected_patterns)
                self._temp_masks = self._initialise_missing_masks(temp_missing_patterns, original_length)
            else:
                self._temp_masks = None
                
            yield self
            
        finally:
            # Restore original state
            self._temp_patterns = original_patterns
            self._temp_masks = original_masks
    
    def _generate_missing_pattern_config(self, pattern: str) -> Dict[str, float]:
        """
        Generate missing pattern configuration for a pattern string.
        
        Args:
            pattern: Pattern string like "at", "av", etc.
            
        Returns:
            Dict mapping modality names to presence probabilities
        """
        config = {}
        
        # Convert pattern string to modality set
        pattern_modalities = set()
        for char in pattern.lower():
            for mod_name, mod_enum in self.AVAILABLE_MODALITIES.items():
                if mod_name.lower().startswith(char):
                    pattern_modalities.add(mod_enum)
                    break
        
        # Set probabilities: 1.0 for modalities in pattern, 0.0 for others
        for mod_name, mod_enum in self.AVAILABLE_MODALITIES.items():
            config[mod_enum] = 1.0 if mod_enum in pattern_modalities else 0.0
            
        return config

    def get_active_patterns(self) -> List[str]:
        """Get currently active patterns (temporary or original)."""
        return self._temp_patterns if self._temp_patterns is not None else self.selected_patterns

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pattern, sample_idx = self._get_pattern_and_sample_idx(idx)

        data = {
            "pattern": pattern,
            "sample_idx": sample_idx,
        }

        # Use temporary masks if in dynamic pattern mode
        active_masks = self._temp_masks if self._temp_masks is not None else self.masks

        for modality in self.AVAILABLE_MODALITIES.values():
            try:
                mask = MultimodalBaseDataset._ndict_accessor.get(active_masks, [pattern, modality, sample_idx])
            except Exception as e:
                console.error(f"Error accessing missing mask: {e}")
                exit(1)

            data[f"{str(modality)}_missing_index"] = torch.tensor(mask)

        return data
