"""
Dataset Split Metrics for Federated Learning

This module provides comprehensive tracking and analysis of IID/Non-IID data distributions
across federated clients, including label distribution analysis, heterogeneity quantification,
and statistical measures of data distribution fairness.
"""
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


@dataclass
class ClientLabelDistribution:
    """Label distribution for a single client."""
    client_id: int
    sample_count: int
    label_counts: Dict[Union[int, str], int]
    label_proportions: Dict[Union[int, str], float]
    num_classes_present: int
    dominant_class: Union[int, str]
    dominant_class_proportion: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "sample_count": self.sample_count,
            "label_counts": {str(k): v for k, v in self.label_counts.items()},
            "label_proportions": {str(k): float(v) for k, v in self.label_proportions.items()},
            "num_classes_present": self.num_classes_present,
            "dominant_class": str(self.dominant_class),
            "dominant_class_proportion": float(self.dominant_class_proportion),
        }


@dataclass
class DatasetSplitMetrics:
    """Comprehensive metrics for analyzing federated dataset splits."""
    
    # Basic statistics
    num_clients: int
    total_samples: int
    distribution_strategy: str
    alpha: Optional[float]  # Dirichlet parameter for non-IID
    
    # Sample distribution
    samples_per_client: List[int]
    sample_distribution_stats: Dict[str, float]  # mean, std, min, max, cv
    
    # Label distribution per client
    client_distributions: List[ClientLabelDistribution]
    
    # Global label distribution
    global_label_counts: Dict[Union[int, str], int]
    global_label_proportions: Dict[Union[int, str], float]
    
    # Heterogeneity metrics
    label_distribution_divergence: Dict[str, float]  # JS divergence, KL divergence metrics
    heterogeneity_score: float  # Overall heterogeneity score (0=IID, 1=highly non-IID)
    gini_coefficient: float  # Sample distribution inequality
    
    # Statistical measures
    inter_client_similarity: Dict[str, float]  # mean, std, min, max similarity scores
    class_balance_metrics: Dict[str, float]  # Global and per-client class balance
    
    # Advanced metrics
    effective_num_clients_per_class: Dict[Union[int, str], float]
    label_skew_coefficient: float
    participation_inequality: float  # How unequally clients participate in each class
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "num_clients": self.num_clients,
            "total_samples": self.total_samples,
            "distribution_strategy": self.distribution_strategy,
            "alpha": self.alpha,
            "samples_per_client": self.samples_per_client,
            "sample_distribution_stats": {k: float(v) for k, v in self.sample_distribution_stats.items()},
            "client_distributions": [client.to_dict() for client in self.client_distributions],
            "global_label_counts": {str(k): v for k, v in self.global_label_counts.items()},
            "global_label_proportions": {str(k): float(v) for k, v in self.global_label_proportions.items()},
            "label_distribution_divergence": {k: float(v) for k, v in self.label_distribution_divergence.items()},
            "heterogeneity_score": float(self.heterogeneity_score),
            "gini_coefficient": float(self.gini_coefficient),
            "inter_client_similarity": {k: float(v) for k, v in self.inter_client_similarity.items()},
            "class_balance_metrics": {k: float(v) for k, v in self.class_balance_metrics.items()},
            "effective_num_clients_per_class": {str(k): float(v) for k, v in self.effective_num_clients_per_class.items()},
            "label_skew_coefficient": float(self.label_skew_coefficient),
            "participation_inequality": float(self.participation_inequality),
        }


class DatasetSplitAnalyzer:
    """Analyzer for computing comprehensive dataset split metrics."""
    
    def __init__(self, task_type: str = "classification"):
        """
        Initialize the analyzer.
        
        Args:
            task_type: Type of task ("classification" or "multilabel_classification")
        """
        self.task_type = task_type
    
    def analyze_split(
        self,
        client_assignments: List[List[int]],
        labels: np.ndarray,
        distribution_strategy: str,
        alpha: Optional[float] = None,
    ) -> DatasetSplitMetrics:
        """
        Analyze a federated dataset split and compute comprehensive metrics.
        
        Args:
            client_assignments: List of sample indices for each client
            labels: Array of labels for all samples
            distribution_strategy: "iid" or "non_iid"
            alpha: Dirichlet parameter for non-IID splits
            
        Returns:
            DatasetSplitMetrics object with comprehensive analysis
        """
        num_clients = len(client_assignments)
        total_samples = len(labels)
        
        # Basic sample distribution
        samples_per_client = [len(assignment) for assignment in client_assignments]
        sample_stats = self._compute_sample_distribution_stats(samples_per_client)
        
        # Process labels based on task type
        if self.task_type == "multilabel_classification":
            # Convert multilabel to string combinations for analysis
            processed_labels = self._process_multilabel_labels(labels)
        else:
            # Handle potential one-hot encoding
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                processed_labels = np.argmax(labels, axis=1)
            else:
                processed_labels = labels.flatten()
        
        # Compute client label distributions
        client_distributions = self._compute_client_distributions(
            client_assignments, processed_labels
        )
        
        # Global label distribution
        global_label_counts, global_label_proportions = self._compute_global_distribution(
            processed_labels
        )
        
        # Heterogeneity metrics
        divergence_metrics = self._compute_label_divergence_metrics(
            client_distributions, global_label_proportions
        )
        
        heterogeneity_score = self._compute_heterogeneity_score(
            client_distributions, global_label_proportions
        )
        
        gini_coefficient = self._compute_gini_coefficient(samples_per_client)
        
        # Inter-client similarity
        similarity_metrics = self._compute_inter_client_similarity(client_distributions)
        
        # Class balance metrics
        balance_metrics = self._compute_class_balance_metrics(
            client_distributions, global_label_proportions
        )
        
        # Advanced metrics
        effective_clients = self._compute_effective_clients_per_class(client_distributions)
        skew_coefficient = self._compute_label_skew_coefficient(client_distributions)
        participation_inequality = self._compute_participation_inequality(client_distributions)
        
        return DatasetSplitMetrics(
            num_clients=num_clients,
            total_samples=total_samples,
            distribution_strategy=distribution_strategy,
            alpha=alpha,
            samples_per_client=samples_per_client,
            sample_distribution_stats=sample_stats,
            client_distributions=client_distributions,
            global_label_counts=global_label_counts,
            global_label_proportions=global_label_proportions,
            label_distribution_divergence=divergence_metrics,
            heterogeneity_score=heterogeneity_score,
            gini_coefficient=gini_coefficient,
            inter_client_similarity=similarity_metrics,
            class_balance_metrics=balance_metrics,
            effective_num_clients_per_class=effective_clients,
            label_skew_coefficient=skew_coefficient,
            participation_inequality=participation_inequality,
        )
    
    def _process_multilabel_labels(self, labels: np.ndarray) -> np.ndarray:
        """Convert multilabel arrays to string combinations for analysis."""
        if len(labels.shape) == 1:
            return labels
        return np.array([
            "".join(map(str, label.astype(int))) 
            for label in labels
        ])
    
    def _compute_sample_distribution_stats(self, samples_per_client: List[int]) -> Dict[str, float]:
        """Compute statistics for sample distribution across clients."""
        samples = np.array(samples_per_client)
        return {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
            "cv": float(np.std(samples) / np.mean(samples)) if np.mean(samples) > 0 else 0.0,
        }
    
    def _compute_client_distributions(
        self, client_assignments: List[List[int]], labels: np.ndarray
    ) -> List[ClientLabelDistribution]:
        """Compute label distribution for each client."""
        client_distributions = []
        
        for client_id, assignment in enumerate(client_assignments):
            if not assignment:
                # Handle empty assignments
                client_distributions.append(ClientLabelDistribution(
                    client_id=client_id,
                    sample_count=0,
                    label_counts={},
                    label_proportions={},
                    num_classes_present=0,
                    dominant_class="None",
                    dominant_class_proportion=0.0,
                ))
                continue
            
            client_labels = labels[assignment]
            label_counts = Counter(client_labels)
            sample_count = len(assignment)
            
            label_proportions = {
                label: count / sample_count 
                for label, count in label_counts.items()
            }
            
            dominant_class = max(label_counts, key=label_counts.get)
            dominant_proportion = label_proportions[dominant_class]
            
            client_distributions.append(ClientLabelDistribution(
                client_id=client_id,
                sample_count=sample_count,
                label_counts=dict(label_counts),
                label_proportions=label_proportions,
                num_classes_present=len(label_counts),
                dominant_class=dominant_class,
                dominant_class_proportion=dominant_proportion,
            ))
        
        return client_distributions
    
    def _compute_global_distribution(
        self, labels: np.ndarray
    ) -> Tuple[Dict[Union[int, str], int], Dict[Union[int, str], float]]:
        """Compute global label distribution."""
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        label_proportions = {
            label: count / total_samples 
            for label, count in label_counts.items()
        }
        
        return dict(label_counts), label_proportions
    
    def _compute_label_divergence_metrics(
        self, 
        client_distributions: List[ClientLabelDistribution],
        global_proportions: Dict[Union[int, str], float]
    ) -> Dict[str, float]:
        """Compute label distribution divergence metrics."""
        all_labels = list(global_proportions.keys())
        
        # Prepare distributions for comparison
        global_dist = np.array([global_proportions.get(label, 0.0) for label in all_labels])
        
        js_divergences = []
        kl_divergences = []
        
        for client_dist in client_distributions:
            if client_dist.sample_count == 0:
                continue
                
            client_dist_array = np.array([
                client_dist.label_proportions.get(label, 0.0) 
                for label in all_labels
            ])
            
            # Add small epsilon to avoid log(0) in KL divergence
            epsilon = 1e-10
            client_dist_array = client_dist_array + epsilon
            global_dist_smooth = global_dist + epsilon
            
            # Normalize after adding epsilon
            client_dist_array = client_dist_array / client_dist_array.sum()
            global_dist_smooth = global_dist_smooth / global_dist_smooth.sum()
            
            # Jensen-Shannon divergence
            js_div = jensenshannon(client_dist_array, global_dist_smooth)
            js_divergences.append(js_div)
            
            # KL divergence (client || global)
            kl_div = entropy(client_dist_array, global_dist_smooth)
            kl_divergences.append(kl_div)
        
        return {
            "js_divergence_mean": float(np.mean(js_divergences)) if js_divergences else 0.0,
            "js_divergence_std": float(np.std(js_divergences)) if js_divergences else 0.0,
            "js_divergence_max": float(np.max(js_divergences)) if js_divergences else 0.0,
            "kl_divergence_mean": float(np.mean(kl_divergences)) if kl_divergences else 0.0,
            "kl_divergence_std": float(np.std(kl_divergences)) if kl_divergences else 0.0,
            "kl_divergence_max": float(np.max(kl_divergences)) if kl_divergences else 0.0,
        }
    
    def _compute_heterogeneity_score(
        self,
        client_distributions: List[ClientLabelDistribution],
        global_proportions: Dict[Union[int, str], float]
    ) -> float:
        """
        Compute overall heterogeneity score (0=IID, 1=highly non-IID).
        
        This combines multiple factors:
        - Label distribution divergence from global
        - Class participation inequality
        - Sample distribution variance
        """
        if not client_distributions or not global_proportions:
            return 0.0
        
        all_labels = list(global_proportions.keys())
        
        # Factor 1: Average JS divergence from global distribution
        js_divergences = []
        for client_dist in client_distributions:
            if client_dist.sample_count == 0:
                continue
                
            client_array = np.array([
                client_dist.label_proportions.get(label, 0.0) 
                for label in all_labels
            ])
            global_array = np.array([global_proportions[label] for label in all_labels])
            
            # Add epsilon and normalize
            epsilon = 1e-10
            client_array = (client_array + epsilon) / (client_array + epsilon).sum()
            global_array = (global_array + epsilon) / (global_array + epsilon).sum()
            
            js_div = jensenshannon(client_array, global_array)
            js_divergences.append(js_div)
        
        avg_js_divergence = np.mean(js_divergences) if js_divergences else 0.0
        
        # Factor 2: Class participation inequality (how many clients have each class)
        class_participation = defaultdict(int)
        active_clients = sum(1 for c in client_distributions if c.sample_count > 0)
        
        for client_dist in client_distributions:
            for label in client_dist.label_counts:
                if client_dist.label_counts[label] > 0:
                    class_participation[label] += 1
        
        if class_participation and active_clients > 0:
            participation_rates = [count / active_clients for count in class_participation.values()]
            participation_std = np.std(participation_rates)
        else:
            participation_std = 0.0
        
        # Factor 3: Sample distribution coefficient of variation
        samples = [c.sample_count for c in client_distributions if c.sample_count > 0]
        sample_cv = np.std(samples) / np.mean(samples) if samples and np.mean(samples) > 0 else 0.0
        
        # Combine factors (weighted average)
        heterogeneity_score = (
            0.6 * min(avg_js_divergence, 1.0) +  # JS divergence (main factor)
            0.3 * min(participation_std, 1.0) +   # Class participation inequality  
            0.1 * min(sample_cv, 1.0)             # Sample distribution inequality
        )
        
        return float(heterogeneity_score)
    
    def _compute_gini_coefficient(self, samples_per_client: List[int]) -> float:
        """Compute Gini coefficient for sample distribution inequality."""
        if not samples_per_client or all(s == 0 for s in samples_per_client):
            return 0.0
        
        # Filter out zero values and sort
        samples = sorted([s for s in samples_per_client if s > 0])
        n = len(samples)
        
        if n == 0:
            return 0.0
        
        # Compute Gini coefficient
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * samples)) / (n * np.sum(samples)) - (n + 1) / n
        
        return float(gini)
    
    def _compute_inter_client_similarity(
        self, client_distributions: List[ClientLabelDistribution]
    ) -> Dict[str, float]:
        """Compute inter-client label distribution similarity metrics."""
        active_clients = [c for c in client_distributions if c.sample_count > 0]
        
        if len(active_clients) < 2:
            return {
                "mean": 1.0,
                "std": 0.0,
                "min": 1.0,
                "max": 1.0,
            }
        
        # Get all unique labels
        all_labels = set()
        for client in active_clients:
            all_labels.update(client.label_proportions.keys())
        all_labels = sorted(list(all_labels))
        
        # Compute pairwise similarities (1 - JS divergence)
        similarities = []
        
        for i in range(len(active_clients)):
            for j in range(i + 1, len(active_clients)):
                client_i = active_clients[i]
                client_j = active_clients[j]
                
                dist_i = np.array([client_i.label_proportions.get(label, 0.0) for label in all_labels])
                dist_j = np.array([client_j.label_proportions.get(label, 0.0) for label in all_labels])
                
                # Add epsilon and normalize
                epsilon = 1e-10
                dist_i = (dist_i + epsilon) / (dist_i + epsilon).sum()
                dist_j = (dist_j + epsilon) / (dist_j + epsilon).sum()
                
                js_div = jensenshannon(dist_i, dist_j)
                similarity = 1.0 - js_div
                similarities.append(similarity)
        
        return {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
        }
    
    def _compute_class_balance_metrics(
        self,
        client_distributions: List[ClientLabelDistribution],
        global_proportions: Dict[Union[int, str], float]
    ) -> Dict[str, float]:
        """Compute class balance metrics."""
        # Global class balance (entropy-based)
        global_probs = np.array(list(global_proportions.values()))
        global_probs = global_probs[global_probs > 0]  # Remove zero probabilities
        global_entropy = entropy(global_probs, base=2)
        max_entropy = np.log2(len(global_probs)) if len(global_probs) > 1 else 0
        global_balance = global_entropy / max_entropy if max_entropy > 0 else 1.0
        
        # Per-client class balance
        client_balances = []
        for client in client_distributions:
            if client.sample_count == 0 or not client.label_proportions:
                continue
                
            client_probs = np.array(list(client.label_proportions.values()))
            client_probs = client_probs[client_probs > 0]
            
            if len(client_probs) > 1:
                client_entropy = entropy(client_probs, base=2)
                max_client_entropy = np.log2(len(client_probs))
                client_balance = client_entropy / max_client_entropy
            else:
                client_balance = 0.0  # Only one class = perfectly imbalanced
                
            client_balances.append(client_balance)
        
        return {
            "global_balance": float(global_balance),
            "client_balance_mean": float(np.mean(client_balances)) if client_balances else 0.0,
            "client_balance_std": float(np.std(client_balances)) if client_balances else 0.0,
            "client_balance_min": float(np.min(client_balances)) if client_balances else 0.0,
            "client_balance_max": float(np.max(client_balances)) if client_balances else 0.0,
        }
    
    def _compute_effective_clients_per_class(
        self, client_distributions: List[ClientLabelDistribution]
    ) -> Dict[Union[int, str], float]:
        """Compute effective number of clients per class (based on Shannon entropy)."""
        # Collect all labels
        all_labels = set()
        for client in client_distributions:
            all_labels.update(client.label_proportions.keys())
        
        effective_clients = {}
        
        for label in all_labels:
            # Get proportion of this label for each client
            proportions = []
            total_label_samples = 0
            
            for client in client_distributions:
                label_count = client.label_counts.get(label, 0)
                total_label_samples += label_count
            
            if total_label_samples == 0:
                effective_clients[label] = 0.0
                continue
            
            # Calculate each client's share of this label
            for client in client_distributions:
                label_count = client.label_counts.get(label, 0)
                proportion = label_count / total_label_samples if total_label_samples > 0 else 0.0
                if proportion > 0:
                    proportions.append(proportion)
            
            # Calculate effective number using Shannon entropy
            if proportions:
                entropy_val = entropy(proportions, base=2)
                effective_clients[label] = 2 ** entropy_val
            else:
                effective_clients[label] = 0.0
        
        return effective_clients
    
    def _compute_label_skew_coefficient(
        self, client_distributions: List[ClientLabelDistribution]
    ) -> float:
        """Compute label skew coefficient across all clients."""
        # Collect dominant class proportions
        dominant_proportions = [
            client.dominant_class_proportion 
            for client in client_distributions 
            if client.sample_count > 0
        ]
        
        if not dominant_proportions:
            return 0.0
        
        # Higher standard deviation = more skewed
        return float(np.std(dominant_proportions))
    
    def _compute_participation_inequality(
        self, client_distributions: List[ClientLabelDistribution]
    ) -> float:
        """Compute how unequally clients participate in each class."""
        # Get all labels
        all_labels = set()
        for client in client_distributions:
            all_labels.update(client.label_counts.keys())
        
        if not all_labels:
            return 0.0
        
        active_clients = sum(1 for c in client_distributions if c.sample_count > 0)
        if active_clients == 0:
            return 0.0
        
        # For each label, calculate participation rate
        participation_rates = []
        
        for label in all_labels:
            clients_with_label = sum(
                1 for c in client_distributions 
                if c.label_counts.get(label, 0) > 0
            )
            participation_rate = clients_with_label / active_clients
            participation_rates.append(participation_rate)
        
        # Standard deviation of participation rates
        return float(np.std(participation_rates))

    def save_metrics(self, metrics: DatasetSplitMetrics, output_path: Path) -> None:
        """Save dataset split metrics to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        print(f"Dataset split metrics saved to: {output_path}")