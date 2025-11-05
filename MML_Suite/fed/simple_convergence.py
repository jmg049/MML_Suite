"""
Simple convergence monitoring for federated learning.
Designed to work with actual metric structure from the codebase.
"""
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from experiment_utils.printing import get_console, print_info, print_success, print_warning
from experiment_utils.utils import prepare_metrics_for_json

console = get_console()


class SimpleConvergenceMonitor:
    """Simple convergence monitoring that works with actual metric structure."""
    
    def __init__(self, metrics_dir: Path, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize convergence monitor.
        
        Args:
            metrics_dir: Directory to save convergence metrics
            patience: Number of rounds to wait for improvement
            min_delta: Minimum improvement to reset patience counter
        """
        self.metrics_dir = metrics_dir
        self.patience = patience
        self.min_delta = min_delta
        
        # Track metrics over rounds
        self.round_history: List[Dict[str, Any]] = []
        self.global_history: List[Dict[str, float]] = []
        
        # Convergence state
        self.best_metric = None
        self.wait_count = 0
        self.converged = False
        self.convergence_round = None
        
        # Primary metrics to track (will be auto-detected)
        self.primary_accuracy_key = None
        self.primary_loss_key = "loss"
        
    def add_round_metrics(self, round_num: int, global_metrics: Dict[str, float], 
                         client_metrics: List[Tuple[Any, Any]], 
                         communication_metrics: Dict[str, Any] = None,
                         pre_aggregation_metrics: Dict[str, float] = None,
                         phase: str = "base_model") -> Dict[str, Any]:
        """
        Add metrics for a completed round and check for convergence.
        
        Args:
            round_num: Current round number
            global_metrics: Global model test metrics (post-aggregation)
            client_metrics: List of (base_metrics, cmam_metrics) from clients
            communication_metrics: Communication overhead metrics from all clients
            pre_aggregation_metrics: Global model test metrics before aggregation
            phase: Training phase ("base_model", "cmam")
            
        Returns:
            Dictionary with convergence analysis
        """
        # Extract meaningful metrics from the actual structure
        round_data = {
            "round": round_num,
            "phase": phase,
            "timestamp": time.time(),
            "global_metrics": global_metrics.copy(),
            "client_count": len(client_metrics)
        }
        
        # Add pre-aggregation metrics if provided
        if pre_aggregation_metrics:
            round_data["pre_aggregation_metrics"] = pre_aggregation_metrics.copy()
        
        # Add communication metrics if provided
        if communication_metrics:
            round_data["communication_metrics"] = communication_metrics
        
        # Auto-detect primary accuracy metric on first round
        if self.primary_accuracy_key is None:
            self.primary_accuracy_key = self._detect_primary_accuracy_key(global_metrics)
            print_info(console, f"Primary accuracy metric detected: {self.primary_accuracy_key}")
        
        # Calculate client-level statistics
        client_stats = self._analyze_client_metrics(client_metrics)
        round_data.update(client_stats)
        
        # Store round data
        self.round_history.append(round_data)
        self.global_history.append(global_metrics)
        
        # Check convergence
        convergence_info = self._check_convergence(global_metrics)
        round_data.update(convergence_info)
        
        # Save round metrics
        self._save_round_metrics(round_data)
        
        # Print convergence summary
        self._print_convergence_summary(round_data)
        
        return round_data
    
    def _detect_primary_accuracy_key(self, metrics: Dict[str, float]) -> str:
        """Detect the primary accuracy metric from available metrics."""
        # Look for common accuracy patterns
        accuracy_keys = [k for k in metrics.keys() if 'accuracy' in k.lower()]
        
        # Priority order for accuracy metrics
        priorities = [
            "accuracy_AI",    # Audio+Image (AVMNIST complete)
            "accuracy_TV",    # Text+Video (MOSEI complete)  
            "accuracy",       # Generic accuracy
        ]
        
        # Check for exact matches first
        for priority in priorities:
            if priority in accuracy_keys:
                return priority
        
        # If no exact match, take first accuracy metric found
        if accuracy_keys:
            return accuracy_keys[0]
        
        # Fallback to loss
        print_warning(console, "No accuracy metric found, using loss for convergence")
        return "loss"
    
    def _analyze_client_metrics(self, client_metrics: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        """Analyze client metrics to extract statistics."""
        if not client_metrics:
            return {"client_variance": 0.0, "client_mean": 0.0}
        
        # Extract client accuracies/losses
        client_values = []
        for base_result, cmam_result in client_metrics:
            if base_result is not None:
                # Check if result has val_results attribute
                if hasattr(base_result, 'val_results'):
                    val_metrics = base_result.val_results
                    if self.primary_accuracy_key in val_metrics:
                        client_values.append(val_metrics[self.primary_accuracy_key])
                    elif "loss" in val_metrics:
                        client_values.append(val_metrics["loss"])
                # If it's a dictionary (like C-MAM results)
                elif isinstance(base_result, dict):
                    if self.primary_accuracy_key in base_result:
                        client_values.append(base_result[self.primary_accuracy_key])
                    elif "loss" in base_result:
                        client_values.append(base_result["loss"])
            elif cmam_result is not None:
                # Try to extract from C-MAM results if base result is None
                if isinstance(cmam_result, dict):
                    if self.primary_accuracy_key in cmam_result:
                        client_values.append(cmam_result[self.primary_accuracy_key])
                    elif "loss" in cmam_result:
                        client_values.append(cmam_result["loss"])
        
        if not client_values:
            print_warning(console, f"No client values found for analysis. Primary key: {self.primary_accuracy_key}")
            return {"client_variance": 0.0, "client_mean": 0.0}
        
        return {
            "client_variance": float(np.var(client_values)),
            "client_mean": float(np.mean(client_values)),
            "client_std": float(np.std(client_values)),
            "client_min": float(np.min(client_values)),
            "client_max": float(np.max(client_values)),
            "client_count": len(client_values)
        }
    
    def _check_convergence(self, global_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if the model has converged."""
        if not global_metrics or self.primary_accuracy_key not in global_metrics:
            return {"converged": False, "improvement": False, "wait_count": self.wait_count}
        
        current_metric = global_metrics[self.primary_accuracy_key]
        
        # Determine if this is an improvement
        improvement = False
        if self.best_metric is None:
            improvement = True
            self.best_metric = current_metric
            self.wait_count = 0
        else:
            # For accuracy metrics, higher is better; for loss, lower is better
            if "loss" in self.primary_accuracy_key.lower():
                improvement = (self.best_metric - current_metric) > self.min_delta
            else:
                improvement = (current_metric - self.best_metric) > self.min_delta
            
            if improvement:
                self.best_metric = current_metric
                self.wait_count = 0
            else:
                self.wait_count += 1
        
        # Check for convergence
        if self.wait_count >= self.patience and not self.converged:
            self.converged = True
            self.convergence_round = len(self.round_history)
            print_success(console, f"Convergence detected at round {self.convergence_round}!")
        
        return {
            "converged": self.converged,
            "improvement": improvement,
            "wait_count": self.wait_count,
            "best_metric": self.best_metric,
            "current_metric": current_metric,
            "convergence_round": self.convergence_round
        }
    
    def _save_round_metrics(self, round_data: Dict[str, Any]) -> None:
        """Save round metrics to file."""
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual round metrics
        round_file = self.metrics_dir / f"convergence_round_{round_data['round']:03d}.json"
        prepared_data = prepare_metrics_for_json([round_data])[0]
        
        with open(round_file, 'w') as f:
            json.dump(prepared_data, f, indent=4)
    
    def _print_convergence_summary(self, round_data: Dict[str, Any]) -> None:
        """Print convergence summary to console."""
        round_num = round_data["round"]
        current = round_data.get("current_metric", 0.0)
        best = round_data.get("best_metric", 0.0)
        wait = round_data.get("wait_count", 0)
        improved = round_data.get("improvement", False)
        
        status = "↑" if improved else "→"
        color = "green" if improved else "yellow" if wait < self.patience else "red"
        
        console.print(f"[{color}]Round {round_num:3d}: {self.primary_accuracy_key}={current:.4f} {status} "
                     f"(best={best:.4f}, wait={wait}/{self.patience})[/]")
        
        if round_data.get("converged", False):
            console.print(f"[bold green]🎯 Converged! No improvement for {self.patience} rounds[/]")
    
    def get_convergence_rate(self, window_size: int = 5) -> Optional[float]:
        """Calculate convergence rate over recent rounds."""
        if len(self.global_history) < window_size:
            return None
        
        recent_metrics = []
        for metrics in self.global_history[-window_size:]:
            if self.primary_accuracy_key in metrics:
                recent_metrics.append(metrics[self.primary_accuracy_key])
        
        if len(recent_metrics) < 2:
            return None
        
        return (recent_metrics[-1] - recent_metrics[0]) / (len(recent_metrics) - 1)
    
    def save_final_summary(self) -> None:
        """Save final convergence summary."""
        if not self.round_history:
            return
        
        summary = {
            "total_rounds": len(self.round_history),
            "converged": self.converged,
            "convergence_round": self.convergence_round,
            "primary_metric": self.primary_accuracy_key,
            "best_metric_value": self.best_metric,
            "final_metric_value": self.round_history[-1].get("current_metric"),
            "convergence_rate": self.get_convergence_rate(),
            "patience_used": self.patience,
            "min_delta": self.min_delta,
            "experiment_duration_hours": (time.time() - self.round_history[0]["timestamp"]) / 3600
        }
        
        # Add final round metrics
        if self.round_history:
            summary["final_global_metrics"] = self.round_history[-1]["global_metrics"]
        
        summary_file = self.metrics_dir / "convergence_summary.json"
        prepared_summary = prepare_metrics_for_json([summary])[0]
        
        with open(summary_file, 'w') as f:
            json.dump(prepared_summary, f, indent=4)
        
        print_success(console, f"Convergence summary saved to {summary_file}")
        
        # Print final summary
        console.rule("[bold green]Convergence Summary")
        console.print(f"Total Rounds: {summary['total_rounds']}")
        console.print(f"Converged: {'Yes' if summary['converged'] else 'No'}")
        if summary['converged']:
            console.print(f"Convergence Round: {summary['convergence_round']}")
        console.print(f"Primary Metric: {summary['primary_metric']}")
        console.print(f"Best Value: {summary['best_metric_value']:.4f}")
        console.print(f"Final Value: {summary['final_metric_value']:.4f}")
        if summary['convergence_rate']:
            console.print(f"Convergence Rate: {summary['convergence_rate']:.6f}")
        console.print(f"Duration: {summary['experiment_duration_hours']:.2f} hours")