import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from experiment_utils.printing import get_console, print_info, print_success, print_warning
from experiment_utils.utils import ensure_dir, prepare_metrics_for_json

from fed.client import ClientMetricsResult

console = get_console()


@dataclass
class GroupMetrics:
    """Metrics for a specific client group (modality assignment)."""
    group_name: str
    client_ids: List[int]
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {mean, std, min, max}
    client_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_name": self.group_name,
            "client_ids": self.client_ids,
            "metrics": {
                metric_name: {
                    stat_name: float(stat_value) 
                    for stat_name, stat_value in metric_stats.items()
                }
                for metric_name, metric_stats in self.metrics.items()
            },
            "client_count": self.client_count,
        }


@dataclass
class RoundMetrics:
    """Comprehensive metrics for a single federated round."""
    round_number: int
    timestamp: float
    
    # Global metrics - now supports multiple metrics
    global_metrics: Dict[str, float]
    
    # Group-based metrics (for incongruent analysis)
    group_metrics: Dict[str, GroupMetrics]
    
    # Client-level metrics - now supports multiple metrics per client
    client_metrics: Dict[int, Dict[str, float]]
    client_assignments: Dict[int, str]
    
    # Communication metrics
    total_bytes_sent: int
    total_bytes_received: int
    model_bytes_per_client: int
    cmam_bytes_per_client: int
    
    # Convergence metrics - computed for primary accuracy metric
    primary_accuracy_key: str
    accuracy_variance: float
    inter_group_variance: float
    intra_group_variance: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "timestamp": self.timestamp,
            "global_metrics": {k: float(v) for k, v in self.global_metrics.items()},
            "group_metrics": {k: v.to_dict() for k, v in self.group_metrics.items()},
            "client_metrics": {
                str(client_id): {metric_name: float(metric_value) for metric_name, metric_value in metrics.items()}
                for client_id, metrics in self.client_metrics.items()
            },
            "client_assignments": {str(k): v for k, v in self.client_assignments.items()},
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "model_bytes_per_client": self.model_bytes_per_client,
            "cmam_bytes_per_client": self.cmam_bytes_per_client,
            "primary_accuracy_key": self.primary_accuracy_key,
            "accuracy_variance": float(self.accuracy_variance),
            "inter_group_variance": float(self.inter_group_variance),
            "intra_group_variance": float(self.intra_group_variance),
        }

    @property
    def global_accuracy(self) -> float:
        """Backward compatibility: return primary accuracy metric."""
        return self.global_metrics.get(self.primary_accuracy_key, 0.0)
    
    @property
    def global_loss(self) -> float:
        """Backward compatibility: return loss metric."""
        return self.global_metrics.get("loss", 0.0)


@dataclass
class FederatedExperimentMetrics:
    """Tracks metrics across all rounds of a federated experiment."""
    
    round_metrics: List[RoundMetrics] = field(default_factory=list)
    experiment_start_time: float = field(default_factory=time.time)
    
    def add_round_metrics(self, round_metrics: RoundMetrics) -> None:
        """Add metrics for a completed round."""
        self.round_metrics.append(round_metrics)
        print_success(console, f"Round {round_metrics.round_number} metrics recorded")
    
    def get_convergence_rate(self, window_size: int = 5, metric_key: Optional[str] = None) -> Optional[float]:
        """Calculate convergence rate over recent rounds for a specific metric."""
        if len(self.round_metrics) < window_size:
            return None
        
        recent_rounds = self.round_metrics[-window_size:]
        
        # Use provided metric key or fall back to primary accuracy
        if metric_key is None:
            metric_key = recent_rounds[0].primary_accuracy_key
        
        metric_values = [r.global_metrics.get(metric_key, 0.0) for r in recent_rounds]
        
        # Simple convergence rate: change in metric over time
        if len(metric_values) < 2:
            return None
        
        return (metric_values[-1] - metric_values[0]) / (len(metric_values) - 1)
    
    def get_time_to_convergence(self, threshold: float = 0.001, window_size: int = 5, 
                               metric_key: Optional[str] = None) -> Optional[int]:
        """Find round where convergence was achieved (metric change < threshold)."""
        if len(self.round_metrics) < window_size + 1:
            return None
        
        # Use provided metric key or fall back to primary accuracy
        if metric_key is None:
            metric_key = self.round_metrics[0].primary_accuracy_key
        
        for i in range(window_size, len(self.round_metrics)):
            window = self.round_metrics[i-window_size:i]
            metric_values = [r.global_metrics.get(metric_key, 0.0) for r in window]
            
            if len(metric_values) >= 2:
                changes = [abs(metric_values[j] - metric_values[j-1]) for j in range(1, len(metric_values))]
                if all(change < threshold for change in changes):
                    return i + 1  # Return round number (1-indexed)
        
        return None
    
    def save_to_json(self, file_path: Path) -> None:
        """Save all experiment metrics to JSON file."""
        ensure_dir(file_path.parent)
        
        data = {
            "experiment_start_time": self.experiment_start_time,
            "total_rounds": len(self.round_metrics),
            "experiment_duration_hours": (time.time() - self.experiment_start_time) / 3600,
            "convergence_rates": {
                # Calculate convergence for all available metrics
                metric_key: self.get_convergence_rate(metric_key=metric_key)
                for metric_key in (self.round_metrics[0].global_metrics.keys() if self.round_metrics else [])
            },
            "time_to_convergence": {
                # Calculate time to convergence for all available metrics
                metric_key: self.get_time_to_convergence(metric_key=metric_key)
                for metric_key in (self.round_metrics[0].global_metrics.keys() if self.round_metrics else [])
            },
            "round_metrics": [r.to_dict() for r in self.round_metrics],
        }
        
        prepared_data = prepare_metrics_for_json([data])[0]
        
        with open(file_path, 'w') as f:
            json.dump(prepared_data, f, indent=4)
        
        print_success(console, f"Federated experiment metrics saved to {file_path}")


class FederatedMetricsAnalyzer:
    """Analyzes client metrics and computes federated-specific statistics."""
    
    @staticmethod
    def analyze_round_metrics(
        client_metrics: List[tuple[ClientMetricsResult, Optional[ClientMetricsResult]]],
        client_assignments: Dict[int, str],
        round_number: int,
        communication_data: Optional[Dict[str, Any]] = None
    ) -> RoundMetrics:
        """
        Analyze metrics from all clients for a single round.
        
        Args:
            client_metrics: List of (model_metrics, cmam_metrics) tuples from clients
            client_assignments: Mapping of client_id -> modality_assignment
            round_number: Current round number
            communication_data: Optional communication statistics
        
        Returns:
            RoundMetrics object with comprehensive round analysis
        """
        print_info(console, f"Analyzing metrics for round {round_number}")
        
        # Extract client-level metrics with support for multiple metrics per client
        client_all_metrics = {}  # client_id -> {metric_name: value}
        
        for i, (model_result, cmam_result) in enumerate(client_metrics):
            # Handle cases where only one type of result is available
            if model_result is not None:
                client_id = model_result.client_id
                val_metrics = model_result.val_results
            elif cmam_result is not None:
                if isinstance(cmam_result, ClientMetricsResult):
                    client_id = cmam_result.client_id
                elif isinstance(cmam_result, dict):
                    client_id = cmam_result["client_id"]
                val_metrics = cmam_result.val_results
            else:
                continue
            
            # Store all metrics for this client
            client_all_metrics[client_id] = dict(val_metrics)
        
        if not client_all_metrics:
            raise ValueError("No valid client metrics found")
        
        # Identify all available metric keys across all clients
        all_metric_keys = set()
        for client_metrics_dict in client_all_metrics.values():
            all_metric_keys.update(client_metrics_dict.keys())
        
        # Identify accuracy keys and select primary accuracy metric
        accuracy_keys = [k for k in all_metric_keys if re.match(r"^accuracy", k, re.IGNORECASE)]
        primary_accuracy_key = FederatedMetricsAnalyzer._select_primary_accuracy_key(accuracy_keys)
        
        if not primary_accuracy_key:
            print_warning(console, "No accuracy metrics found, using 'loss' as primary metric")
            primary_accuracy_key = "loss"
        
        print_info(console, f"Found {len(all_metric_keys)} total metrics, {len(accuracy_keys)} accuracy metrics")
        print_info(console, f"Primary accuracy metric: {primary_accuracy_key}")
        
        # Calculate global metrics (mean across all clients for each metric)
        global_metrics = {}
        for metric_key in all_metric_keys:
            values = []
            for client_id, client_metrics_dict in client_all_metrics.items():
                if metric_key in client_metrics_dict:
                    values.append(client_metrics_dict[metric_key])
            
            if values:
                global_metrics[metric_key] = np.mean(values)
            else:
                global_metrics[metric_key] = 0.0
        
        # Group clients by modality assignment
        groups = defaultdict(list)
        for client_id, assignment in client_assignments.items():
            if client_id in client_all_metrics:
                groups[assignment].append(client_id)
        
        # Calculate group metrics
        group_metrics = {}
        group_primary_accuracies = {}  # For variance calculations
        
        for group_name, client_ids in groups.items():
            group_metric_stats = {}
            
            # Calculate stats for each metric within this group
            for metric_key in all_metric_keys:
                values = []
                for client_id in client_ids:
                    if metric_key in client_all_metrics[client_id]:
                        values.append(client_all_metrics[client_id][metric_key])
                
                if values:
                    group_metric_stats[metric_key] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values)
                    }
                else:
                    group_metric_stats[metric_key] = {
                        "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0
                    }
            
            group_metrics[group_name] = GroupMetrics(
                group_name=group_name,
                client_ids=client_ids,
                metrics=group_metric_stats,
                client_count=len(client_ids)
            )
            
            # Store primary accuracy values for variance calculation
            primary_acc_values = []
            for client_id in client_ids:
                if primary_accuracy_key in client_all_metrics[client_id]:
                    primary_acc_values.append(client_all_metrics[client_id][primary_accuracy_key])
            group_primary_accuracies[group_name] = primary_acc_values
        
        # Calculate variance metrics for drift analysis (using primary accuracy metric)
        primary_accuracy_values = []
        for client_metrics_dict in client_all_metrics.values():
            if primary_accuracy_key in client_metrics_dict:
                primary_accuracy_values.append(client_metrics_dict[primary_accuracy_key])
        
        accuracy_variance = np.var(primary_accuracy_values) if primary_accuracy_values else 0.0
        
        # Inter-group variance (between groups)
        group_means = []
        for group_name, group_metric in group_metrics.items():
            if primary_accuracy_key in group_metric.metrics:
                group_means.append(group_metric.metrics[primary_accuracy_key]["mean"])
        
        inter_group_variance = np.var(group_means) if len(group_means) > 1 else 0.0
        
        # Intra-group variance (within groups)
        intra_group_variances = []
        for group_name, group_accs in group_primary_accuracies.items():
            if len(group_accs) > 1:
                intra_group_variances.append(np.var(group_accs))
        
        intra_group_variance = np.mean(intra_group_variances) if intra_group_variances else 0.0
        
        # Communication metrics (placeholder for now)
        comm_data = communication_data or {}
        
        round_metrics = RoundMetrics(
            round_number=round_number,
            timestamp=time.time(),
            global_metrics=global_metrics,
            group_metrics=group_metrics,
            client_metrics=client_all_metrics,
            client_assignments=client_assignments,
            total_bytes_sent=comm_data.get("total_bytes_sent", 0),
            total_bytes_received=comm_data.get("total_bytes_received", 0),
            model_bytes_per_client=comm_data.get("model_bytes_per_client", 0),
            cmam_bytes_per_client=comm_data.get("cmam_bytes_per_client", 0),
            primary_accuracy_key=primary_accuracy_key,
            accuracy_variance=accuracy_variance,
            inter_group_variance=inter_group_variance,
            intra_group_variance=intra_group_variance,
        )
        
        # Print summary
        FederatedMetricsAnalyzer._print_round_summary(round_metrics)
        
        return round_metrics
    
    @staticmethod
    def _select_primary_accuracy_key(accuracy_keys: List[str]) -> Optional[str]:
        """
        Select the primary accuracy metric from available accuracy keys.
        Prioritizes more general accuracy metrics over specific ones.
        """
        if not accuracy_keys:
            return None
        
        # Priority order: prefer simpler/more general accuracy metrics
        priorities = [
            "accuracy",           # Most general
            "val_accuracy",       # Validation accuracy
            "test_accuracy",      # Test accuracy
            "train_accuracy",     # Training accuracy
        ]
        
        # Check for exact matches first
        for priority in priorities:
            if priority in accuracy_keys:
                return priority
        
        # If no exact match, look for partial matches
        for priority in priorities:
            for key in accuracy_keys:
                if priority.lower() in key.lower():
                    return key
        
        # Fall back to the first accuracy key found
        return accuracy_keys[0]
    
    @staticmethod
    def _print_round_summary(round_metrics: RoundMetrics) -> None:
        """Print a summary of round metrics to console."""
        print_info(console, f"=== Round {round_metrics.round_number} Summary ===")
        
        # Print global metrics
        print_info(console, "Global Metrics:")
        for metric_name, value in round_metrics.global_metrics.items():
            print_info(console, f"  {metric_name}: {value:.4f}")
        
        print_info(console, f"Primary Accuracy Variance: {round_metrics.accuracy_variance:.6f}")
        
        if len(round_metrics.group_metrics) > 1:
            print_info(console, f"Inter-group Variance: {round_metrics.inter_group_variance:.6f}")
            print_info(console, f"Intra-group Variance: {round_metrics.intra_group_variance:.6f}")
            
            print_info(console, "Group Performance:")
            for group_name, group_metrics in round_metrics.group_metrics.items():
                primary_metric = group_metrics.metrics.get(round_metrics.primary_accuracy_key, {})
                mean_val = primary_metric.get("mean", 0.0)
                std_val = primary_metric.get("std", 0.0)
                print_info(console, 
                    f"  {group_name}: {mean_val:.4f} ± {std_val:.4f} "
                    f"({group_metrics.client_count} clients)"
                )
        
        print_info(console, "=" * 40)
    
    @staticmethod
    def detect_client_drift(
        current_metrics: RoundMetrics,
        previous_metrics: List[RoundMetrics],
        drift_threshold: float = 0.05,
        metric_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect client drift by comparing current performance to historical performance.
        
        Args:
            current_metrics: Current round metrics
            previous_metrics: List of previous round metrics (recent history)
            drift_threshold: Threshold for detecting significant drift
            metric_key: Specific metric to analyze (uses primary accuracy if None)
        
        Returns:
            Dictionary with drift analysis results
        """
        if not previous_metrics:
            return {"drift_detected": False, "reason": "No historical data"}
        
        # Use specified metric or fall back to primary accuracy
        if metric_key is None:
            metric_key = current_metrics.primary_accuracy_key
        
        # Calculate baseline performance from previous rounds
        baseline_values = []
        for rm in previous_metrics:
            if metric_key in rm.global_metrics:
                baseline_values.append(rm.global_metrics[metric_key])
        
        if not baseline_values:
            return {"drift_detected": False, "reason": f"No historical data for metric {metric_key}"}
        
        baseline_metric = np.mean(baseline_values)
        current_metric = current_metrics.global_metrics.get(metric_key, 0.0)
        
        # Global drift detection
        global_drift = abs(current_metric - baseline_metric)
        global_drift_detected = global_drift > drift_threshold
        
        # Group-level drift detection
        group_drift_info = {}
        if len(current_metrics.group_metrics) > 1:
            for group_name, current_group in current_metrics.group_metrics.items():
                # Find historical performance for this group
                historical_values = []
                for prev_round in previous_metrics:
                    if (group_name in prev_round.group_metrics and 
                        metric_key in prev_round.group_metrics[group_name].metrics):
                        historical_values.append(
                            prev_round.group_metrics[group_name].metrics[metric_key]["mean"]
                        )
                
                if historical_values:
                    baseline_group_metric = np.mean(historical_values)
                    current_group_metric = current_group.metrics.get(metric_key, {}).get("mean", 0.0)
                    group_drift = abs(current_group_metric - baseline_group_metric)
                    
                    group_drift_info[group_name] = {
                        "drift_magnitude": group_drift,
                        "drift_detected": group_drift > drift_threshold,
                        "current_metric": current_group_metric,
                        "baseline_metric": baseline_group_metric
                    }
        
        return {
            "drift_detected": global_drift_detected or any(
                info["drift_detected"] for info in group_drift_info.values()
            ),
            "metric_analyzed": metric_key,
            "global_drift": {
                "magnitude": global_drift,
                "detected": global_drift_detected,
                "current_metric": current_metric,
                "baseline_metric": baseline_metric
            },
            "group_drift": group_drift_info,
            "drift_threshold": drift_threshold
        }