#!/usr/bin/env python3
"""
Dataset Split Metrics Demo

This script demonstrates how to use the new dataset split metrics functionality
for analyzing IID and Non-IID federated data distributions.
"""

from pathlib import Path
import numpy as np
from fed.dataset_split_metrics import DatasetSplitAnalyzer


def create_demo_labels(n_samples: int = 1000, n_classes: int = 10) -> np.ndarray:
    """Create demo labels for testing."""
    return np.random.randint(0, n_classes, size=n_samples)


def create_iid_assignments(labels: np.ndarray, n_clients: int = 10) -> list[list[int]]:
    """Create IID client assignments."""
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    
    assignments = []
    samples_per_client = n_samples // n_clients
    
    for i in range(n_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < n_clients - 1 else n_samples
        assignments.append(indices[start:end].tolist())
    
    return assignments


def create_non_iid_assignments(labels: np.ndarray, n_clients: int = 10, alpha: float = 0.1) -> list[list[int]]:
    """Create Non-IID client assignments using Dirichlet distribution."""
    n_classes = len(np.unique(labels))
    
    # Group by class
    class_indices = {c: np.where(labels == c)[0] for c in range(n_classes)}
    
    assignments = [[] for _ in range(n_clients)]
    
    for class_id, indices in class_indices.items():
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        proportions = (proportions * len(indices)).astype(int)
        proportions[-1] = len(indices) - proportions[:-1].sum()
        
        start = 0
        for client_id, n_samples in enumerate(proportions):
            if n_samples > 0:
                assignments[client_id].extend(indices[start:start + n_samples].tolist())
                start += n_samples
    
    return assignments


def demo_metrics_analysis():
    """Demonstrate metrics analysis for different data distributions."""
    print("Dataset Split Metrics Demo")
    print("=" * 50)
    
    # Create synthetic data
    labels = create_demo_labels(n_samples=2000, n_classes=10)
    analyzer = DatasetSplitAnalyzer(task_type="classification")
    
    print(f"Dataset: {len(labels)} samples, {len(np.unique(labels))} classes")
    print(f"Global label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # Analyze IID split
    print("\n" + "="*30 + " IID Analysis " + "="*30)
    iid_assignments = create_iid_assignments(labels, n_clients=20)
    iid_metrics = analyzer.analyze_split(
        client_assignments=iid_assignments,
        labels=labels,
        distribution_strategy="iid",
        alpha=None
    )
    
    print(f"Heterogeneity Score: {iid_metrics.heterogeneity_score:.3f}")
    print(f"JS Divergence (mean): {iid_metrics.label_distribution_divergence['js_divergence_mean']:.3f}")
    print(f"Client Similarity (mean): {iid_metrics.inter_client_similarity['mean']:.3f}")
    print(f"Gini Coefficient: {iid_metrics.gini_coefficient:.3f}")
    print(f"Label Skew: {iid_metrics.label_skew_coefficient:.3f}")
    
    # Analyze different levels of Non-IID
    alpha_values = [1.0, 0.5, 0.1, 0.01]
    print("\n" + "="*25 + " Non-IID Analysis (Different α) " + "="*25)
    
    for alpha in alpha_values:
        non_iid_assignments = create_non_iid_assignments(labels, n_clients=20, alpha=alpha)
        non_iid_metrics = analyzer.analyze_split(
            client_assignments=non_iid_assignments,
            labels=labels,
            distribution_strategy="non_iid",
            alpha=alpha
        )
        
        print(f"\nα = {alpha}")
        print(f"  Heterogeneity Score: {non_iid_metrics.heterogeneity_score:.3f}")
        print(f"  JS Divergence (mean): {non_iid_metrics.label_distribution_divergence['js_divergence_mean']:.3f}")
        print(f"  Client Similarity (mean): {non_iid_metrics.inter_client_similarity['mean']:.3f}")
        print(f"  Gini Coefficient: {non_iid_metrics.gini_coefficient:.3f}")
        print(f"  Label Skew: {non_iid_metrics.label_skew_coefficient:.3f}")
    
    # Save detailed metrics
    output_dir = Path("./metrics_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*25 + " Saving Detailed Metrics " + "="*25)
    
    # Save IID metrics
    iid_path = output_dir / "demo_iid_metrics.json"
    analyzer.save_metrics(iid_metrics, iid_path)
    print(f"IID metrics saved to: {iid_path}")
    
    # Save highly Non-IID metrics
    highly_non_iid_assignments = create_non_iid_assignments(labels, n_clients=20, alpha=0.01)
    highly_non_iid_metrics = analyzer.analyze_split(
        client_assignments=highly_non_iid_assignments,
        labels=labels,
        distribution_strategy="non_iid",
        alpha=0.01
    )
    
    non_iid_path = output_dir / "demo_highly_non_iid_metrics.json"
    analyzer.save_metrics(highly_non_iid_metrics, non_iid_path)
    print(f"Highly Non-IID metrics saved to: {non_iid_path}")
    
    print("\n" + "="*25 + " Summary Comparison " + "="*25)
    print(f"{'Metric':<25} {'IID':<12} {'Non-IID (α=0.01)':<18}")
    print("-" * 60)
    print(f"{'Heterogeneity Score':<25} {iid_metrics.heterogeneity_score:<12.3f} {highly_non_iid_metrics.heterogeneity_score:<18.3f}")
    print(f"{'JS Divergence':<25} {iid_metrics.label_distribution_divergence['js_divergence_mean']:<12.3f} {highly_non_iid_metrics.label_distribution_divergence['js_divergence_mean']:<18.3f}")
    print(f"{'Client Similarity':<25} {iid_metrics.inter_client_similarity['mean']:<12.3f} {highly_non_iid_metrics.inter_client_similarity['mean']:<18.3f}")
    print(f"{'Gini Coefficient':<25} {iid_metrics.gini_coefficient:<12.3f} {highly_non_iid_metrics.gini_coefficient:<18.3f}")
    print(f"{'Label Skew':<25} {iid_metrics.label_skew_coefficient:<12.3f} {highly_non_iid_metrics.label_skew_coefficient:<18.3f}")
    
    print("\n✅ Demo completed! Check the generated JSON files for detailed metrics.")


def demo_federated_dataset_integration():
    """Demonstrate how the metrics are integrated into FederatedDataset."""
    print("\n" + "="*20 + " FederatedDataset Integration Demo " + "="*20)
    print("This is how the metrics are automatically computed during federated training:")
    print("""
# In your federated training script:
from fed.data_utils import FederatedDataset

# Create federated dataset (this will automatically compute and save metrics)
fed_dataset = FederatedDataset(
    base_dataset=your_dataset,
    num_clients=10,
    distribution_strategy="non_iid",  # or "iid"
    alpha=0.1,  # for non-IID
    # ... other parameters
)

# Metrics are automatically saved to: ./federated_metrics/federated_train_non_iid_alpha0.1_dataset_split_metrics.json
""")
    
    print("The metrics file will contain:")
    print("- Comprehensive client label distributions")
    print("- Heterogeneity scores and statistical measures")
    print("- Inter-client similarity analysis")
    print("- Class balance and participation metrics")
    print("- Effective number of clients per class")
    print("- And much more for thorough analysis!")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    demo_metrics_analysis()
    demo_federated_dataset_integration()