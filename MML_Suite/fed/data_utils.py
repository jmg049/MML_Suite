from __future__ import annotations
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from fed import DataSplitType
from fed.dataset_split_metrics import DatasetSplitAnalyzer, DatasetSplitMetrics
from data.base_dataset import MultimodalBaseDataset
from experiment_utils.logging import get_logger
from experiment_utils.printing import get_console, print_debug, print_info
from torch.utils.data import Subset

console = get_console()
logger = get_logger()


class PatternAwareSubset(Subset):
    """
    A Subset that properly handles pattern expansion for validation/test splits.

    This subset understands that validation/test datasets expand their samples
    by the number of patterns, and correctly maps global indices to the
    expanded index space.
    """

    def __init__(
        self, dataset: MultimodalBaseDataset, indices: List[int], split: str, patterns: Optional[List[str]] = None
    ):
        """
        Args:
            dataset: The base multimodal dataset
            indices: Indices for the subset (in base sample space)
            split: Dataset split ("train", "valid", "test")
            patterns: Patterns to use (defaults to dataset's selected patterns)
        """
        self.split = split
        self.base_indices = indices
        self.patterns = patterns or dataset.get_selected_patterns()
        self.full_pattern = "".join(self.patterns)
        self.full_pattern = sorted(self.full_pattern)  # Ensure consistent order

        if split in ["valid", "test", "val"]:
            # For validation/test, expand indices to cover all patterns
            expanded_indices = []
            num_base_samples = dataset.num_samples

            for pattern_idx, pattern in enumerate(self.patterns):
                pattern_offset = pattern_idx * num_base_samples
                expanded_indices.extend([idx + pattern_offset for idx in indices])

            super().__init__(dataset, expanded_indices)
        else:
            # For training, use indices as-is
            super().__init__(dataset, indices)

    def __len__(self):
        if self.split in ["valid", "test", "val"]:
            return len(self.base_indices) * len(self.patterns)
        return len(self.base_indices)
    
    def temporary_patterns(self, patterns: List[str]):
        """
        Delegate temporary pattern switching to the underlying dataset.
        
        This allows PatternAwareSubset to work with the new context manager approach
        by passing the pattern switching request to the base dataset.
        
        Args:
            patterns: List of patterns to use temporarily
        """
        if hasattr(self.dataset, 'temporary_patterns'):
            return self.dataset.temporary_patterns(patterns)
        else:
            raise ValueError(
                f"Underlying dataset {type(self.dataset)} does not support temporary patterns. "
                f"Ensure the dataset inherits from MultimodalBaseDataset."
            )

class FederatedDataset:
    """
    This class properly handles pattern expansion for validation/test splits,
    ensuring that global and client datasets include all pattern variations.
    """
    

    def __init__(
        self,
        base_dataset: MultimodalBaseDataset,
        num_clients: int,
        distribution_strategy: Literal["iid", "non_iid"] = "iid",
        task_type: Literal["classification", "multilabel_classification"] = "classification",
        client_patterns: Optional[list[list[str]]] = None,
        global_fraction: float = 0.0,  # Fraction of data reserved for global model
        global_split_strategy: Literal["random", "stratified"] = "stratified",
        alpha: float = 0.5,  # For Dirichlet distribution in non-IID
        min_samples_per_client: int = 10,
        seed: int = 42,
        modality_distribution_method: Literal["congruent", "incongruent"] = "congruent",
    ) -> None:
        """
        Initialize FederatedDataset with proper pattern handling.

        Args:
            base_dataset: The underlying MultimodalBaseDataset instance
            num_clients: Number of federated clients
            distribution_strategy: How to distribute data among clients
            task_type: Type of learning task
            client_patterns: Optional list of patterns to assign to clients
            global_fraction: Fraction of data to reserve for global model (0.0 to 1.0)
            global_split_strategy: How to split global data ("random" or "stratified")
            alpha: Dirichlet distribution parameter for non-IID splits
            min_samples_per_client: Minimum samples each client should have
            seed: Random seed for reproducibility
            modality_distribution_method: Whether to use congruent or incongruent modality distribution
        """
        self.base_dataset = base_dataset
        self.num_clients = num_clients
        self.distribution_strategy = distribution_strategy
        self.task_type = task_type
        self.global_fraction = global_fraction
        self.global_split_strategy = global_split_strategy
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client
        self.split = base_dataset.split
        self.all_possible_patterns = base_dataset.get_all_possible_patterns()
        self.modality_distribution_method = modality_distribution_method

        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Handle client patterns assignment
        if client_patterns is not None:
            if len(client_patterns) != num_clients:
                raise ValueError(
                    f"Length of client_patterns ({len(client_patterns)}) must match num_clients ({num_clients})"
                )
            print_info(console, f"Assigning provided patterns to {num_clients} clients")
            self.client_patterns = client_patterns
            self.available_patterns = base_dataset.get_selected_patterns()
        else:
            # Randomly assign patterns to clients
            print_info(console, f"Assigning random patterns to {num_clients} clients")
            available_patterns = base_dataset.get_selected_patterns()
            self.available_patterns = available_patterns
            self.client_patterns = [[random.choice(available_patterns)] for _ in range(num_clients)]

        # Get labels for distribution
        self.labels = self._extract_labels()

        # Split data between global and client data (in base sample space)
        self.global_indices, self.client_pool_indices = self._split_global_client_data()

        # Create global dataset if needed
        self.global_dataset = None
        if len(self.global_indices) > 0:
            # Use PatternAwareSubset for proper pattern handling
            self.global_dataset = PatternAwareSubset(
                base_dataset, self.global_indices, split=self.split, patterns=base_dataset.get_selected_patterns()
            )

        # Assign client pool data to individual clients
        self.client_assignments = self._assign_client_data(num_clients, distribution_strategy, task_type)

        # Create client datasets as Subsets
        self.client_datasets = self._create_client_datasets()

        # Log distribution info
        self._log_distribution_info()

    def _extract_labels(self) -> np.ndarray:
        """Extract labels from the base dataset."""
        labels = []

        # Different datasets store labels differently
        if hasattr(self.base_dataset, "labels"):
            # For MSP_IMPROV/IEMOCAP style datasets
            labels = self.base_dataset.labels
        elif hasattr(self.base_dataset, "data") and hasattr(self.base_dataset.data, "columns"):
            # For AVMNIST style (pandas DataFrame)
            labels = self.base_dataset.data[self.base_dataset.labels_column].values
        elif hasattr(self.base_dataset, "data") and hasattr(self.base_dataset, "_load_label"):
            # For MMIMDb style (HDF5)
            labels = [self.base_dataset._load_label(i) for i in range(self.base_dataset.num_samples)]
            labels = np.array(labels)
        elif hasattr(self.base_dataset, "get_labels"):
            # For datasets with a get_labels method
            print_info(console, "Extracting labels using get_labels method")
            labels = self.base_dataset.get_labels()
        else:
            raise ValueError("Cannot extract labels from base dataset")

        return labels

    def _split_global_client_data(self) -> Tuple[List[int], List[int]]:
        """Split data between global model and client pool in base sample space."""
        # Use num_samples (base samples), not expanded length
        n_samples = self.base_dataset.num_samples
        all_indices = list(range(n_samples))

        if self.global_fraction == 0.0:
            return [], all_indices

        n_global = int(n_samples * self.global_fraction)

        if self.global_split_strategy == "random":
            # Random split
            random.shuffle(all_indices)
            global_indices = all_indices[:n_global]
            client_indices = all_indices[n_global:]
        else:  # stratified
            # Stratified split to maintain class distribution
            global_indices = []
            client_indices = []

            if self.task_type == "classification":
                labels = self.labels
                if len(labels.shape) > 1:
                    labels = np.argmax(labels, axis=1)

                # Group indices by class
                class_indices = defaultdict(list)
                for idx, label in enumerate(labels):
                    class_indices[label].append(idx)

                # Take proportional samples from each class
                for class_label, indices in class_indices.items():
                    np.random.shuffle(indices)
                    n_global_class = int(len(indices) * self.global_fraction)
                    global_indices.extend(indices[:n_global_class])
                    client_indices.extend(indices[n_global_class:])
            else:  # multilabel - use random for simplicity
                random.shuffle(all_indices)
                global_indices = all_indices[:n_global]
                client_indices = all_indices[n_global:]

        return global_indices, client_indices

    def _assign_client_data(
        self,
        num_clients: int,
        distribution_strategy: Literal["iid", "non_iid"],
        task_type: Literal["classification", "multilabel_classification"] ,
    ) -> List[List[int]]:
        """Assigns data samples to clients based on the specified distribution strategy."""
        if distribution_strategy == "iid":
            if task_type == "classification":
                return self._cls_assign_iid_data(num_clients)
            else:
                return self._ml_cls_assign_iid_data(num_clients)
        else:  # non-IID
            if task_type == "classification":
                return self._cls_assign_non_iid_data(num_clients)
            else:
                return self._ml_cls_assign_non_iid_data(num_clients)

    def _cls_assign_iid_data(self, num_clients: int) -> List[List[int]]:
        """IID distribution for classification tasks."""
        # Use only client pool indices
        indices = self.client_pool_indices.copy()
        random.shuffle(indices)

        # Split indices evenly among clients
        n_samples = len(indices)
        samples_per_client = n_samples // num_clients
        client_assignments = []

        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < num_clients - 1 else n_samples
            client_assignments.append(indices[start_idx:end_idx])

        return client_assignments

    def _ml_cls_assign_iid_data(self, num_clients: int) -> List[List[int]]:
        """IID distribution for multilabel classification tasks."""
        # For multilabel, we treat it similarly to single-label IID
        return self._cls_assign_iid_data(num_clients)

    def _cls_assign_non_iid_data(self, num_clients: int) -> List[List[int]]:
        """Non-IID distribution for classification using Dirichlet distribution."""
        # Use only client pool indices
        client_pool_labels = self.labels[self.client_pool_indices]

        if len(client_pool_labels.shape) > 1:
            # If labels are one-hot encoded, convert to class indices
            client_pool_labels = np.argmax(client_pool_labels, axis=1)

        # Get unique classes
        unique_classes = np.unique(client_pool_labels)

        # Group indices by class (using indices from client pool)
        class_indices = defaultdict(list)
        for i, idx in enumerate(self.client_pool_indices):
            label = client_pool_labels[i]
            class_indices[label].append(idx)

        # Use Dirichlet distribution to create non-IID splits
        client_assignments = [[] for _ in range(num_clients)]

        for class_label in unique_classes:
            class_idxs = np.array(class_indices[class_label])
            np.random.shuffle(class_idxs)
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(self.alpha, num_clients))
            proportions = (proportions * len(class_idxs)).astype(int)
            proportions[-1] = len(class_idxs) - proportions[:-1].sum()

            # Assign samples to clients
            start = 0
            for client_id, num_samples in enumerate(proportions):
                if num_samples > 0:
                    client_assignments[client_id].extend(class_idxs[start : start + num_samples].tolist())
                    start += num_samples

        # Ensure minimum samples per client
        for i, assignment in enumerate(client_assignments):
            if len(assignment) < self.min_samples_per_client:
                # Steal some samples from the client with most samples
                max_client = max(range(num_clients), key=lambda x: len(client_assignments[x]))
                if max_client != i and len(client_assignments[max_client]) > self.min_samples_per_client:
                    steal_count = min(
                        self.min_samples_per_client - len(assignment),
                        len(client_assignments[max_client]) - self.min_samples_per_client,
                    )
                    if steal_count > 0:
                        client_assignments[i].extend(client_assignments[max_client][-steal_count:])
                        client_assignments[max_client] = client_assignments[max_client][:-steal_count]

        return client_assignments

    def _ml_cls_assign_non_iid_data(self, num_clients: int) -> List[List[int]]:
        """Non-IID distribution for multilabel classification."""
        # Use only client pool indices
        client_pool_labels = self.labels[self.client_pool_indices]

        # For multilabel, we can create non-IID by:
        # 1. Clustering based on label combinations
        # 2. Assigning clusters to clients non-uniformly

        # Convert multilabel to label combination strings for grouping
        label_combinations = ["".join(map(str, label.astype(int))) for label in client_pool_labels]
        unique_combinations = list(set(label_combinations))

        # Group indices by label combination
        combination_indices = defaultdict(list)
        for i, (idx, combo) in enumerate(zip(self.client_pool_indices, label_combinations)):
            combination_indices[combo].append(idx)

        # Distribute combinations among clients (similar to class-based approach)
        client_assignments = [[] for _ in range(num_clients)]

        for combo in unique_combinations:
            combo_idxs = np.array(combination_indices[combo])
            np.random.shuffle(combo_idxs)

            # Use Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(self.alpha, num_clients))
            proportions = (proportions * len(combo_idxs)).astype(int)
            proportions[-1] = len(combo_idxs) - proportions[:-1].sum()

            start = 0
            for client_id, num_samples in enumerate(proportions):
                if num_samples > 0:
                    client_assignments[client_id].extend(combo_idxs[start : start + num_samples].tolist())
                    start += num_samples

        return client_assignments

    def _create_client_datasets(self) -> List["ClientSubset"]:
        """Create Subset datasets for each client."""
        client_datasets = []

        for client_id, indices in enumerate(self.client_assignments,):
            # Create a ClientSubset that maintains pattern information
            # For congruent training, clients need access to all patterns for C-MAM training
            # For incongruent training, clients are limited to their assigned patterns
            
            # Debug information
            print_debug(console, f"Creating client {client_id+1}: split='{self.split}', method='{self.modality_distribution_method}'")
            print_debug(console, f"  Base dataset patterns: {self.base_dataset.get_selected_patterns()}")
            print_debug(console, f"  Client assigned patterns: {self.client_patterns[client_id] if client_id < len(self.client_patterns) else 'None'}")
            
            if self.split == DataSplitType.TRAIN.value and self.modality_distribution_method == "congruent":
                # Congruent training: clients have access to all patterns for C-MAM training
                client_patterns = self.base_dataset.get_selected_patterns()
                print_debug(console, f"  Using congruent patterns: {client_patterns}")
            elif self.split == DataSplitType.TRAIN.value:
                # Incongruent training: clients limited to assigned patterns
                client_patterns = self.client_patterns[client_id]
                print_debug(console, f"  Using incongruent patterns: {client_patterns}")
            else:
                # Validation/test: for incongruent, use client's assigned pattern; for congruent, use all patterns
                if self.modality_distribution_method == "incongruent":
                    # For incongruent training, validation should use the same pattern as training
                    client_patterns = self.client_patterns[client_id]
                    print_debug(console, f"  Using incongruent validation pattern: {client_patterns}")
                else:
                    # For congruent training, use all patterns for comprehensive evaluation
                    client_patterns = self.base_dataset.get_selected_patterns()
                    print_debug(console, f"  Using congruent validation patterns: {client_patterns}")
                
            client_subset = ClientSubset(
                self.base_dataset,
                indices,
                patterns=client_patterns,
                client_id=client_id,
                split=self.split,
            )
            client_datasets.append(client_subset)

        return client_datasets

    def get_client_dataset(self, client_id: int) -> ClientSubset:
        """Get the dataset for a specific client."""
        if client_id >= self.num_clients:
            raise ValueError(f"Client ID {client_id} out of range (0-{self.num_clients-1})")
        return self.client_datasets[client_id]

    def get_client_dataloader(self, client_id: int, batch_size: int, **kwargs):
        """Get a DataLoader for a specific client."""
        from torch.utils.data import DataLoader

        client_dataset = self.get_client_dataset(client_id)

        # Use the base dataset's collate_fn if it has one
        collate_fn = getattr(self.base_dataset, "collate_fn", None)

        return DataLoader(client_dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs)

    def get_global_dataset(self) -> Optional[PatternAwareSubset]:
        """Get the global dataset (if any was reserved)."""
        return self.global_dataset

    def get_global_dataloader(self, batch_size: int, **kwargs):
        """Get a DataLoader for the global dataset."""
        if self.global_dataset is None:
            raise ValueError("No global dataset was created (global_fraction=0)")

        from torch.utils.data import DataLoader

        # Use the base dataset's collate_fn if it has one
        collate_fn = getattr(self.base_dataset, "collate_fn", None)

        return DataLoader(self.global_dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs)

    def _log_distribution_info(self):
        """Log information about the data distribution."""
        logger.info(
            f"Created FederatedDataset with {self.num_clients} clients using {self.distribution_strategy} distribution"
        )

        # Log sample counts per client
        for i, assignment in enumerate(self.client_assignments, start=1):
            logger.info(f"Client {i}: {len(assignment)} samples, pattern: {self.client_patterns[i - 1]}")

        # Log class distribution if classification task
        if self.task_type == "classification":
            self._log_class_distribution()

    def _log_class_distribution(self):
        """Log the class distribution across clients."""
        labels = self.labels
        if len(labels.shape) > 1:
            labels = np.argmax(labels, axis=1)

        unique_classes = np.unique(labels)

        console.print("\n[bold]Class Distribution Across Clients:[/]")
        for client_id, indices in enumerate(self.client_assignments):
            client_labels = labels[indices]
            class_counts = {cls: 0 for cls in unique_classes}
            unique, counts = np.unique(client_labels, return_counts=True)
            for cls, count in zip(unique, counts):
                class_counts[cls] = count

            console.print(f"Client {client_id} (pattern: {self.client_patterns[client_id]}): {dict(class_counts)}")

    def print_summary(self):
        """Print a summary of the federated dataset."""
        console.print("\n[bold]Federated Dataset Summary:[/]")
        console.print(f"[bold]Base Dataset:[/] {self.base_dataset.__class__.__name__}")
        console.print(f"[bold]Split:[/] {self.split}")
        console.print(f"[bold]Number of Clients:[/] {self.num_clients}")
        console.print(f"[bold]Distribution Strategy:[/] {self.distribution_strategy.strategy}")
        console.print(f"[bold]Task Type:[/] {self.task_type}")

        if self.distribution_strategy == "non_iid":
            console.print(f"[bold]Dirichlet Alpha:[/] {self.alpha}")

        # Global dataset info
        if self.global_dataset is not None:
            console.print("\n[bold]Global Dataset:[/]")
            console.print(f"  Base samples: {len(self.global_indices)} ({self.global_fraction*100:.1f}% of total)")
            if self.split in ["valid", "test", "val"]:
                console.print(
                    f"  Expanded samples: {len(self.global_dataset)} (with {len(self.base_dataset.get_selected_patterns())} patterns)"
                )
            console.print(f"  Strategy: {self.global_split_strategy}")

        console.print("\n[bold]Client Data Pool:[/]")
        console.print(f"  Base samples: {len(self.client_pool_indices)} ({(1-self.global_fraction)*100:.1f}% of total)")

        console.print("\n[bold]Samples per Client:[/]")
        total_client_samples = 0
        for i, assignment in enumerate(self.client_assignments):
            n_samples = len(assignment)
            total_client_samples += n_samples
            console.print(f"  Client {i} ({self.client_patterns[i]}): {n_samples} base samples")
            if self.split in ["valid", "test", "val"]:
                console.print(f"    (Expands to {n_samples * len([self.client_patterns[i]])} with pattern)")

        console.print(f"\n[bold]Total Base Samples:[/] {self.base_dataset.num_samples}")
        console.print(f"[bold]Average Samples per Client:[/] {total_client_samples / self.num_clients:.1f}")

        # Calculate standard deviation of sample counts
        sample_counts = [len(assignment) for assignment in self.client_assignments]
        std_dev = np.std(sample_counts)
        console.print(f"[bold]Std Dev of Sample Counts:[/] {std_dev:.1f}")

    def compute_split_metrics(self) -> DatasetSplitMetrics:
        """
        Compute comprehensive dataset split metrics for IID/Non-IID analysis.
        
        Returns:
            DatasetSplitMetrics object containing detailed split analysis
        """
        # Determine task type from dataset
        task_type = getattr(self.base_dataset, 'task_type', 'classification')
        if hasattr(self.base_dataset, 'is_multilabel') and self.base_dataset.is_multilabel:
            task_type = 'multilabel_classification'
        
        # Initialize analyzer
        analyzer = DatasetSplitAnalyzer(task_type=task_type)
        
        # Get labels for analysis
        if hasattr(self.base_dataset, 'labels') and self.base_dataset.labels is not None:
            labels = self.base_dataset.labels
        elif hasattr(self.base_dataset, 'targets') and self.base_dataset.targets is not None:
            labels = self.base_dataset.targets
        else:
            # Fallback: extract labels from dataset
            print_info(console, "Extracting labels from dataset for split analysis...")
            labels = []
            for i in range(len(self.base_dataset)):
                try:
                    sample = self.base_dataset[i]
                    labels.append(sample["labels"])
                except Exception as e:
                    print_debug(console, f"Error extracting label for sample {i}: {e}.")
                    raise ValueError(
                        "Failed to extract labels from dataset. Ensure dataset has 'labels' or 'targets' attribute."
                    )
            labels = np.array(labels)
        
        # Analyze the split
        metrics = analyzer.analyze_split(
            client_assignments=self.client_assignments,
            labels=labels,
            distribution_strategy=self.distribution_strategy,
            alpha=self.alpha,
        )
        
        return metrics
    
    def save_split_metrics(self, output_dir: Path, experiment_name: str = "federated_experiment") -> Path:
        """
        Compute and save dataset split metrics to file.
        
        Args:
            output_dir: Directory to save metrics
            experiment_name: Name of the experiment for file naming
            
        Returns:
            Path to the saved metrics file
        """
        metrics = self.compute_split_metrics()
        
        # Create output path
        output_path = output_dir / f"{experiment_name}_dataset_split_metrics.json"
        
        # Save metrics
        analyzer = DatasetSplitAnalyzer()
        analyzer.save_metrics(metrics, output_path)
        
        # Print summary
        self._print_split_metrics_summary(metrics)
        
        return output_path
    
    def _print_split_metrics_summary(self, metrics: DatasetSplitMetrics) -> None:
        """Print a summary of dataset split metrics to console."""
        console.print("\n[bold blue]Dataset Split Analysis Summary[/]")
        console.print("=" * 50)
        
        console.print(f"[bold]Distribution Strategy:[/] {metrics.distribution_strategy}")
        if metrics.alpha is not None:
            console.print(f"[bold]Alpha (Dirichlet):[/] {metrics.alpha}")
        
        console.print(f"[bold]Total Clients:[/] {metrics.num_clients}")
        console.print(f"[bold]Total Samples:[/] {metrics.total_samples}")
        
        # Sample distribution
        console.print("\n[bold cyan]Sample Distribution:[/]")
        stats = metrics.sample_distribution_stats
        console.print(f"  Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
        console.print(f"  Range: [{stats['min']:.0f}, {stats['max']:.0f}]")
        console.print(f"  Coefficient of Variation: {stats['cv']:.3f}")
        
        # Heterogeneity metrics
        console.print("\n[bold cyan]Data Heterogeneity:[/]")
        console.print(f"  Heterogeneity Score: {metrics.heterogeneity_score:.3f} (0=IID, 1=Non-IID)")
        console.print(f"  Gini Coefficient: {metrics.gini_coefficient:.3f}")
        
        # Label distribution
        console.print("\n[bold cyan]Label Distribution:[/]")
        js_mean = metrics.label_distribution_divergence['js_divergence_mean']
        console.print(f"  Average JS Divergence: {js_mean:.3f}")
        console.print(f"  Label Skew Coefficient: {metrics.label_skew_coefficient:.3f}")
        
        # Inter-client similarity
        console.print("\n[bold cyan]Inter-Client Similarity:[/]")
        sim = metrics.inter_client_similarity
        console.print(f"  Mean: {sim['mean']:.3f}, Std: {sim['std']:.3f}")
        
        # Class balance
        console.print("\n[bold cyan]Class Balance:[/]")
        balance = metrics.class_balance_metrics
        console.print(f"  Global Balance: {balance['global_balance']:.3f}")
        console.print(f"  Client Balance (avg): {balance['client_balance_mean']:.3f}")
        
        console.print("=" * 50)


class ClientSubset(PatternAwareSubset):
    """
    A Subset that maintains pattern information and client ID.

    For validation/test splits, this only includes samples for the client's
    assigned pattern, not all patterns.
    """

    def __init__(self, dataset: MultimodalBaseDataset, indices: List[int], patterns: list[str], client_id: int, split: str):
        self.pattern = patterns
        self.client_id = client_id
        self.selected_patterns = patterns  # Compatibility with base dataset interface

        # Store the dataset's pattern temporarily and restore after each access
        self._original_patterns = dataset.get_selected_patterns()

        # For clients, only use their assigned pattern
        super().__init__(dataset, indices, split, patterns=patterns)

    def __getitem__(self, idx):
        # Temporarily set the pattern for this client
        if hasattr(self.dataset, "selected_patterns"):
            original = self.dataset.selected_patterns
            self.dataset.selected_patterns = self.pattern

        # Get the item using parent class method
        item = super().__getitem__(idx)

        # Restore original patterns
        if hasattr(self.dataset, "selected_patterns"):
            self.dataset.selected_patterns = original

        # Add client information
        item["client_id"] = self.client_id

        return item

    def get_selected_patterns(self) -> List[str]:
        """Return the pattern assigned to this client."""
        return self.selected_patterns

    def temporary_patterns(self, patterns: List[str]):
        """
        Delegate temporary pattern switching to the underlying dataset.
        
        This allows ClientSubset to work with the new context manager approach
        by passing the pattern switching request to the base dataset.
        
        Args:
            patterns: List of patterns to use temporarily
        """
        if hasattr(self.dataset, 'temporary_patterns'):
            return self.dataset.temporary_patterns(patterns)
        else:
            raise ValueError(
                f"Underlying dataset {type(self.dataset)} does not support temporary patterns. "
                f"Ensure the dataset inherits from MultimodalBaseDataset."
            )

    @property
    def num_samples(self) -> int:
        """Number of samples in this client's dataset."""
        return len(self.indices)


def get_cmam_required_pattern(input_modalities: list, target_modality) -> str:
    """
    Determine the required pattern for a C-MAM based on its input and target modalities.
    
    Args:
        input_modalities: List of input modalities for the C-MAM
        target_modality: Target modality that the C-MAM reconstructs
        
    Returns:
        Required pattern string (e.g., "ai", "at", "avt")
        
    Example:
        Audio→Image C-MAM: input=[Audio], target=Image → "ai"
        Audio+Video→Text C-MAM: input=[Audio,Video], target=Text → "atv"
    """
    # Get first letter of each modality (lowercase)
    modality_letters = []
    
    # Add input modalities
    for modality in input_modalities:
        modality_letters.append(str(modality)[0].lower())
    
    if target_modality is not None:
        # Add target modality  
        modality_letters.append(str(target_modality)[0].lower())
    
    # Sort alphabetically for consistent pattern naming
    modality_letters.sort()
    
    return ''.join(modality_letters)



def create_cmam_context_dataloader(base_dataloader, input_modalities: list, target_modality=None, required_pattern: Optional[str] = None, **dataloader_kwargs):
    """
    Create a DataLoader that uses the temporary patterns approach for C-MAM training.
    
    This approach allows the dataset to dynamically generate the required missing patterns
    from the available data, enabling C-MAMs to train on specific patterns even when
    the base dataset was configured with different patterns.
    
    Args:
        base_dataloader: The base DataLoader (from client or global dataset)
        input_modalities: List of input modalities for the C-MAM
        target_modality: Target modality that the C-MAM reconstructs
        **dataloader_kwargs: Additional kwargs to pass to DataLoader constructor
        
    Returns:
        Context manager that yields a DataLoader with the required pattern
        
    Example:
        with create_cmam_context_dataloader(base_dataloader, [Modality.AUDIO], Modality.TEXT) as cmam_loader:
            for batch in cmam_loader:
                # Train C-MAM with "at" pattern data generated from "atv" samples
                pass
    """
    from contextlib import contextmanager
    from torch.utils.data import DataLoader
    
    # Determine required pattern for this C-MAM
    required_pattern = required_pattern if required_pattern else get_cmam_required_pattern(input_modalities, target_modality=target_modality)

    # if target_modality:
    #     console.print(
    #         f"[bold yellow]Note:[/] C-MAM with target modality {target_modality} will generate data for pattern '{required_pattern}' "
    #         f"from available input modalities {input_modalities}."
    #     )
    
    @contextmanager
    def cmam_dataloader_context():
        # Get the base dataset from the dataloader
        base_dataset = base_dataloader.dataset
        
        # Check if the dataset supports temporary patterns
        if not hasattr(base_dataset, 'temporary_patterns'):
            raise ValueError(
                f"Dataset {type(base_dataset)} does not support temporary patterns. "
                f"Ensure the dataset inherits from MultimodalBaseDataset or use create_cmam_filtered_dataloader."
            )
        
        # Use the dataset's context manager to temporarily switch patterns
        with base_dataset.temporary_patterns([required_pattern]):
            # Create new DataLoader with same settings but now the dataset generates required pattern
            dataloader_config = {
                'batch_size': base_dataloader.batch_size,
                'shuffle': False,  # Don't shuffle to maintain consistency
                'num_workers': getattr(base_dataloader, 'num_workers', 0),
                'pin_memory': getattr(base_dataloader, 'pin_memory', False),
                'drop_last': getattr(base_dataloader, 'drop_last', False),
                'collate_fn': getattr(base_dataloader, 'collate_fn', None),
            }
            dataloader_config.update(dataloader_kwargs)
            
            yield DataLoader(base_dataset, **dataloader_config)
    
    return cmam_dataloader_context()


def split_global_client(
    base_dataset: MultimodalBaseDataset,
    num_clients: int,
    global_fraction: float = 0.1,
    distribution_strategy: Literal["iid", "non_iid"] = "iid",
    task_type: Literal["classification", "multilabel_classification"] = "classification",
    client_patterns: Optional[list[list[str]]] = None,
    global_split_strategy: Literal["random", "stratified"] = "stratified",
    modality_distribution_method: Literal["congruent", "incongruent"] = "congruent",
    **kwargs,
) -> Tuple[Optional[PatternAwareSubset], FederatedDataset]:
    """
    Convenience function to split a dataset into global and federated client datasets.

    This function properly handles pattern expansion for validation/test splits,
    ensuring the global dataset includes all patterns.

    Args:
        base_dataset: The base multimodal dataset to split
        num_clients: Number of federated clients
        global_fraction: Fraction of data to reserve for global model (0.0 to 1.0)
        distribution_strategy: How to distribute client data ("iid" or "non_iid")
        task_type: Type of learning task
        client_patterns: Optional list of patterns to assign to clients
        global_split_strategy: How to split global data ("random" or "stratified")
        modality_distribution_method: Whether to use congruent or incongruent modality distribution
        **kwargs: Additional arguments passed to FederatedDataset

    Returns:
        Tuple of (global_dataset, federated_dataset)
        - global_dataset: PatternAwareSubset for the global model (None if global_fraction=0)
        - federated_dataset: FederatedDataset instance managing client data

    Example:
        >>> base_dataset = AVMNIST(data_fp="data.csv", split="test")
        >>> global_data, client_data = split_global_client(
        ...     base_dataset,
        ...     num_clients=10,
        ...     global_fraction=0.2,
        ...     distribution_strategy="non_iid"
        ... )
        >>> # Global dataset will properly iterate through all patterns
        >>> print(f"Global samples: {len(global_data)}")
    """
 
    fed_dataset = FederatedDataset(
        base_dataset=base_dataset,
        num_clients=num_clients,
        distribution_strategy=distribution_strategy,
        task_type=task_type,
        client_patterns=client_patterns,
        global_fraction=global_fraction,
        global_split_strategy=global_split_strategy,
        modality_distribution_method=modality_distribution_method,
        **kwargs,
    )

    return fed_dataset.get_global_dataset(), fed_dataset
