import random
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from experiment_utils.printing import get_console
from experiment_utils.logging import get_logger
from modalities import Modality
from data.base_dataset import MultimodalBaseDataset

console = get_console()
logger = get_logger()


class ContrastiveMultimodalDataset(MultimodalBaseDataset):
    """
    Multimodal dataset with contrastive learning support while preserving missing modality functionality.
    
    Extends MultimodalBaseDataset to add label-aware batching and contrastive sample generation
    for improved class discrimination in C-MAM training.
    """

    def __init__(
        self,
        split: Literal["train", "valid", "test"],
        selected_patterns: Optional[List[str]] = None,
        missing_patterns: Optional[Dict[str, Dict[str, float]]] = None,
        batch_size: int = 1,
        _id: int = 1,
        missing_strategy: Literal["zero", "noise"] = "zero",
        modality_stats: Optional[Dict[Modality, Dict[str, float]]] = None,
        enable_contrastive: bool = True,
        min_samples_per_class: int = 2,
        **kwargs
    ) -> None:
        """
        Initialize the ContrastiveMultimodalDataset.
        
        Args:
            split: Dataset split ("train", "valid", or "test")
            selected_patterns: List of selected missing patterns
            missing_patterns: Dictionary of missing pattern configurations
            batch_size: Batch size for training
            _id: Dataset ID
            missing_strategy: Strategy for handling missing modalities
            modality_stats: Statistics for modality normalization
            enable_contrastive: Whether to enable contrastive learning features
            min_samples_per_class: Minimum samples per class for effective contrastive learning
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            split=split,
            selected_patterns=selected_patterns,
            missing_patterns=missing_patterns,
            batch_size=batch_size,
            _id=_id,
            missing_strategy=missing_strategy,
            modality_stats=modality_stats,
        )
        
        self.enable_contrastive = enable_contrastive
        self.min_samples_per_class = min_samples_per_class
        
        # Will be set by subclasses after loading data
        self.class_to_indices = None
        self.class_distribution = None
        
    def _analyze_class_distribution(self, labels: List[int]) -> None:
        """
        Analyze class distribution for contrastive learning.
        
        Args:
            labels: List of class labels for the dataset
        """
        if not self.enable_contrastive:
            return
            
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)
        
        self.class_distribution = {
            class_id: len(indices) 
            for class_id, indices in self.class_to_indices.items()
        }
        
        # Check if we have enough samples per class for effective contrastive learning
        insufficient_classes = [
            class_id for class_id, count in self.class_distribution.items() 
            if count < self.min_samples_per_class
        ]
        
        if insufficient_classes:
            console.warning(
                f"Classes {insufficient_classes} have fewer than {self.min_samples_per_class} samples. "
                "This may affect contrastive learning effectiveness."
            )
        
        logger.info(
            f"Class distribution for contrastive learning: {dict(self.class_distribution)}"
        )
    
    def get_class_distribution(self) -> Optional[Dict[int, int]]:
        """Get the class distribution for the dataset."""
        return self.class_distribution
    
    def get_contrastive_batch_info(self, batch_indices: List[int]) -> Dict[str, Any]:
        """
        Get information about a batch for contrastive learning.
        
        Args:
            batch_indices: Indices of samples in the batch
            
        Returns:
            Dictionary with batch contrastive learning information
        """
        if not self.enable_contrastive or not hasattr(self, '_get_label_for_index'):
            return {"contrastive_enabled": False}
        
        batch_labels = [self._get_label_for_index(idx) for idx in batch_indices]
        unique_classes = set(batch_labels)
        
        # Count samples per class in this batch
        class_counts = defaultdict(int)
        for label in batch_labels:
            class_counts[label] += 1
        
        # Check if we can form valid triplets
        # Need at least one class with 2+ samples (for anchor-positive pairs)
        # and at least one other class (for negatives)
        can_form_triplets = (
            len(unique_classes) >= 2 and 
            any(count >= 2 for count in class_counts.values())
        )
        
        return {
            "contrastive_enabled": True,
            "unique_classes": list(unique_classes),
            "class_counts": dict(class_counts),
            "can_form_triplets": can_form_triplets,
            "num_classes_in_batch": len(unique_classes),
            "total_samples": len(batch_indices),
        }
    
    def suggest_batch_composition(self, target_batch_size: int) -> Dict[str, Any]:
        """
        Suggest optimal batch composition for contrastive learning.
        
        Args:
            target_batch_size: Desired batch size
            
        Returns:
            Dictionary with suggested batch composition
        """
        if not self.enable_contrastive or not self.class_distribution:
            return {"suggestion": "Standard random sampling"}
        
        num_classes = len(self.class_distribution)
        
        if target_batch_size < 3:
            return {"suggestion": "Batch size too small for contrastive learning (need >= 3)"}
        
        if num_classes <= 1:
            return {"suggestion": "Only one class available, contrastive learning not effective"}
        
        # Suggest including multiple classes with multiple samples each
        suggested_classes = min(num_classes, max(2, target_batch_size // 3))
        samples_per_class = target_batch_size // suggested_classes
        
        return {
            "suggestion": f"Include {suggested_classes} classes with ~{samples_per_class} samples each",
            "recommended_classes": suggested_classes,
            "samples_per_class": samples_per_class,
            "total_classes_available": num_classes,
            "class_distribution": self.class_distribution,
        }
    
    def create_contrastive_aware_batch(self, batch_size: int) -> List[int]:
        """
        Create a batch that's optimized for contrastive learning.
        
        Args:
            batch_size: Size of the batch to create
            
        Returns:
            List of sample indices optimized for contrastive learning
        """
        if not self.enable_contrastive or not self.class_to_indices:
            # Fall back to random sampling
            return random.sample(range(self.num_samples), min(batch_size, self.num_samples))
        
        available_classes = [
            class_id for class_id, indices in self.class_to_indices.items()
            if len(indices) >= 2  # Need at least 2 samples for positive pairs
        ]
        
        if len(available_classes) < 2:
            # Not enough classes for contrastive learning
            return random.sample(range(self.num_samples), min(batch_size, self.num_samples))
        
        batch_indices = []
        
        # Strategy: Include multiple classes with multiple samples each
        num_classes_to_include = min(len(available_classes), max(2, batch_size // 3))
        samples_per_class = batch_size // num_classes_to_include
        remaining_samples = batch_size % num_classes_to_include
        
        selected_classes = random.sample(available_classes, num_classes_to_include)
        
        for i, class_id in enumerate(selected_classes):
            class_indices = self.class_to_indices[class_id]
            
            # Add extra sample to some classes if we have remainder
            num_samples_for_this_class = samples_per_class + (1 if i < remaining_samples else 0)
            num_samples_for_this_class = min(num_samples_for_this_class, len(class_indices))
            
            selected_indices = random.sample(class_indices, num_samples_for_this_class)
            batch_indices.extend(selected_indices)
        
        # If we still need more samples, add random ones
        while len(batch_indices) < batch_size:
            remaining_indices = [
                idx for idx in range(self.num_samples) 
                if idx not in batch_indices
            ]
            if not remaining_indices:
                break
            batch_indices.append(random.choice(remaining_indices))
        
        return batch_indices[:batch_size]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a dataset sample with contrastive learning metadata.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing sample data and contrastive metadata
        """
        # Get base sample from parent class
        sample = super().__getitem__(idx)
        
        # Add contrastive learning information if enabled
        if self.enable_contrastive:
            sample["contrastive_enabled"] = True
            
            # Add class information if available
            if hasattr(self, '_get_label_for_index'):
                sample["class_label"] = self._get_label_for_index(
                    sample.get("sample_idx", idx)
                )
        else:
            sample["contrastive_enabled"] = False
            
        return sample
    
    def _get_label_for_index(self, idx: int) -> int:
        """
        Get the class label for a given sample index.
        
        This method should be implemented by subclasses to provide
        access to class labels.
        
        Args:
            idx: Sample index
            
        Returns:
            Class label for the sample
        """
        raise NotImplementedError(
            "Subclasses must implement _get_label_for_index to support contrastive learning"
        )
    
    def get_contrastive_stats(self) -> Dict[str, Any]:
        """
        Get statistics about contrastive learning readiness.
        
        Returns:
            Dictionary with contrastive learning statistics
        """
        if not self.enable_contrastive:
            return {"contrastive_enabled": False}
        
        stats = {
            "contrastive_enabled": True,
            "num_classes": len(self.class_distribution) if self.class_distribution else 0,
            "min_samples_per_class": self.min_samples_per_class,
        }
        
        if self.class_distribution:
            stats.update({
                "class_distribution": self.class_distribution,
                "classes_with_sufficient_samples": sum(
                    1 for count in self.class_distribution.values() 
                    if count >= self.min_samples_per_class
                ),
                "min_class_size": min(self.class_distribution.values()),
                "max_class_size": max(self.class_distribution.values()),
                "mean_class_size": np.mean(list(self.class_distribution.values())),
            })
        
        return stats


class ContrastiveAVMNIST(ContrastiveMultimodalDataset):
    """
    Contrastive version of AVMNIST dataset.
    
    Example implementation showing how to extend ContrastiveMultimodalDataset
    for a specific dataset.
    """
    
    NUM_CLASSES: int = 10
    VALID_SPLITS: List[Literal["train", "valid", "test"]] = ["train", "valid", "test"]
    AVAILABLE_MODALITIES: Dict[str, Modality] = {"audio": Modality.AUDIO, "image": Modality.IMAGE}
    
    def __init__(
        self,
        data_fp,
        split: str,
        target_modality: Modality | str = Modality.MULTIMODAL,
        labels_column: str = "label",
        enable_contrastive: bool = True,
        **kwargs
    ):
        """
        Initialize ContrastiveAVMNIST dataset.
        
        Args:
            data_fp: Path to the dataset file
            split: Dataset split
            target_modality: Target modality for the task
            labels_column: Name of the labels column
            enable_contrastive: Whether to enable contrastive features
            **kwargs: Additional arguments
        """
        super().__init__(
            split=split,
            enable_contrastive=enable_contrastive,
            **kwargs
        )
        
        self.data_fp = data_fp
        self.labels_column = labels_column
        self.target_modality = target_modality
        
        # Load data would happen here
        # self._load_data()
        
        # After loading, analyze class distribution
        # if enable_contrastive and hasattr(self, 'labels'):
        #     self._analyze_class_distribution(self.labels.tolist())
    
    def _get_label_for_index(self, idx: int) -> int:
        """
        Get the class label for AVMNIST sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Class label (0-9 for AVMNIST)
        """
        # This would be implemented based on how labels are stored
        # return self.data[self.labels_column].iloc[idx]
        pass
    
    @staticmethod
    def get_full_modality() -> str:
        """Get full modality string for AVMNIST."""
        modality_keys = [k[0] for k in ContrastiveAVMNIST.AVAILABLE_MODALITIES.keys()]
        modality_keys.sort()
        return "".join(modality_keys)