from __future__ import annotations

import importlib
import re
from collections import OrderedDict, defaultdict
from functools import partial
from os import PathLike
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from config.metric_config import MetricConfig
from numpy import ndarray
from torch import Tensor

from .logging import get_logger
from .printing import get_console
from .utils import safe_detach

logger = get_logger()
console = get_console()


class MetricRecorder:
    """
    A flexible metric recording system that tracks predictions and ground truths per modality
    and calculates metrics at epoch end.

    This recorder is designed to handle multi-modal data where different modalities might
    require separate metric calculations. It stores predictions and ground truths throughout
    an epoch and computes all metrics at the end.

    Example:
        ```python
        # Initialize with config
        config = MetricConfig.from_yaml('metrics_config.yaml')
        recorder = MetricRecorder(config)

        # During training/evaluation
        for batch in dataloader:
            predictions = model(batch)

            # For each modality type
            for m_type in modalities:
                mask = modalities == m_type
                recorder.update(
                    predictions=predictions[mask],
                    targets=targets[mask],
                    modality=m_type
                )

        # Calculate metrics at epoch end
        metrics = recorder.calculate_metrics()
        print(f"Results: {metrics}")

        # Reset for next epoch
        recorder.reset()
        ```

    Attributes:
        config (MetricConfig): Configuration object containing metric definitions
        metrics (OrderedDict[str, Callable]): Mapping of metric names to their functions
        modality_data (DefaultDict[Any, List[Tuple[ndarray, ndarray]]]): Stored predictions and targets per modality
        current_results (Dict[str, float]): Most recently calculated metric results
    """

    def __init__(
        self,
        config: MetricConfig,
        tensorboard_path: Optional[PathLike] = None,
        tb_record_only: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the MetricRecorder.

        Args:
            config: MetricConfig object containing metric definitions and parameters
        """
        self._validate_config(config)
        self.config = config
        self.metrics: OrderedDict[str, Callable] = self._load_metrics()
        self.modality_data: DefaultDict[Any, List[Tuple[ndarray, ndarray]]] = defaultdict(list)
        self.current_results: Dict[str, float] = {}
        self.tensorboard_path = tensorboard_path

        if self.tensorboard_path:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.tensorboard_path)
                logger.info(f"Logging metrics to TensorBoard at {self.tensorboard_path}")
                self.tb_record_only = tb_record_only
                logger.info(f"Only logging metrics: {self.tb_record_only}")
            except ImportError:
                console.print("[bold red]Error[/]: TensorBoardX not installed. Cannot log to TensorBoard.")
                self.writer = None

    def _validate_config(self, config: MetricConfig) -> None:
        """
        Validate the provided configuration.

        Args:
            config: MetricConfig object to validate

        Raises:
            ValueError: If config is invalid or missing required fields
        """
        if not isinstance(config, MetricConfig):
            raise ValueError("Config must be an instance of MetricConfig")
        if not config.metrics:
            raise ValueError("Config must contain at least one metric definition")

    def _load_metrics(self) -> OrderedDict[str, Callable]:
        """
        Load and initialize metric functions from configuration.

        Returns:
            OrderedDict mapping metric names to their initialized functions

        Raises:
            ImportError: If a metric function cannot be imported
            AttributeError: If a metric function is not found in its module
        """
        metrics = OrderedDict()
        for metric_name, metric_info in self.config.metrics.items():
            try:
                module_name, func_name = metric_info["function"].rsplit(".", 1)
                module = importlib.import_module(module_name)
                metric_func = getattr(module, func_name)
                metrics_kwargs = metric_info.get("kwargs", {})
                metrics[metric_name] = partial(metric_func, **metrics_kwargs)
            except (ImportError, AttributeError) as e:
                raise type(e)(f"Error loading metric '{metric_name}': {str(e)}")
        return metrics

    def update(self, predictions: Tensor | ndarray, targets: Tensor | ndarray, modality: str) -> None:
        """
        Store predictions and targets for later metric calculation.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            modality: Optional identifier for grouping metrics (e.g., modality type)

        Raises:
            ValueError: If predictions and targets have mismatched shapes
        """
        predictions = safe_detach(predictions, to_np=True)
        targets = safe_detach(targets, to_np=True)

        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch between predictions {predictions.shape} and " f"targets {targets.shape}")

        self.modality_data[modality].append((predictions, targets))

    def update_all(self, predictions: Tensor | ndarray, targets: Tensor | ndarray, m_types: Set[str]) -> None:
        """
        Store predictions and targets for later metric calculation. Applies the mask here instead of in the model code.

        """
        if not isinstance(m_types, set):
            try:
                m_types = set(m_types)
            except Exception as e:
                raise TypeError(f"m_types must be a set or set-like object: {str(e)}")

        for m_type in m_types:
            mask = m_types == m_type
            mask_preds = predictions[mask]
            mask_labels = targets[mask]
            self.update(predictions=mask_preds, targets=mask_labels, modality=m_type)

    def calculate_metrics(
        self, metric_group: Optional[str] = None, epoch: Optional[int] = None, loss: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics using stored predictions and targets.

        Returns:
            Dictionary mapping metric names to their computed values

        Note:
            Metric names are formatted as {metric_name}_{modality} for modality-specific metrics
        """
        results = {"loss": loss} if loss is not None else {}

        for modality, data in self.modality_data.items():
            if not data:
                continue

            try:
                all_preds = np.concatenate([p for p, _ in data], axis=0)
                all_targets = np.concatenate([t for _, t in data], axis=0)
            except ValueError as e:
                logger.error(f"Error concatenating data for modality {modality}: {str(e)}")
                continue

            for metric_name, metric_func in self.metrics.items():
                try:
                    value = metric_func(all_targets, all_preds)
                    metric_key = f"{metric_name}"

                    _modality = f"{modality.replace('z', '').upper()}" if modality else metric_name

                    if isinstance(value, dict):
                        for k, v in value.items():
                            results[f"{metric_key}_{k}_{_modality}"] = v
                    else:
                        results[f"{metric_key}_{_modality}"] = value
                except Exception as e:
                    console.print(f"[red]Error calculating metric {metric_name}: {str(e)}")
                    logger.error(f"Metric calculation error - {metric_name}: {str(e)}")

        if self.writer:
            logger.debug(f"Writer is not None {self.tensorboard_path}")
            for metric_name, value in results.items():
                ## want to match the metric name with the tb_record_only list using regex
                if self.tb_record_only:
                    for pattern in self.tb_record_only:
                        if re.match(pattern, metric_name):
                            self.writer.add_scalar(f"{metric_group}_{metric_name}", value, epoch or 0)
                            logger.info(f"Logged metric '{metric_name}' with value {value} to TensorBoard")
                        else:
                            logger.info(
                                f"Skipping metric '{metric_name}' with value {value} from logging to TensorBoard"
                            )
                else:
                    self.writer.add_scalar(f"{metric_group}_{metric_name}", value, epoch or 0)
                    logger.debug(f"Logged metric '{metric_name}' with value {value} to TensorBoard")
        self.current_results = results

        return results

    def get(self, metric_name: str, default: Any = None) -> Any:
        """
        Get the value of a specific metric from most recent calculation.

        Args:
            metric_name: Name of the metric to retrieve
            default: Value to return if metric not found

        Returns:
            Value of the requested metric or default if not found
        """
        return self.current_results.get(metric_name, default)

    def reset(self) -> None:
        """Reset all stored data and results."""
        self.modality_data.clear()
        self.current_results.clear()

    def clone(self) -> MetricRecorder:
        """
        Create a new independent instance with the same configuration.

        Returns:
            New MetricRecorder instance
        """
        return MetricRecorder(self.config)

    def __str__(self) -> str:
        """
        String representation showing configuration and current results.

        Returns:
            Formatted string with recorder state
        """
        metrics_info = [f"  {name}: {func.func.__module__}.{func.func.__name__}" for name, func in self.metrics.items()]
        metrics_str = "\n".join(metrics_info)

        results_str = "\n".join(f"  {metric}: {value:.4f}" for metric, value in self.current_results.items())

        return f"MetricRecorder:\n" f"Configured Metrics:\n{metrics_str}\n" f"Current Results:\n{results_str}"

    @classmethod
    def create_instances(cls, config: MetricConfig, *names: str) -> Dict[str, MetricRecorder]:
        """
        Create multiple recorder instances with the same configuration.

        Args:
            config: MetricConfig object to use for all instances
            *names: Names for the different instances (e.g., 'train', 'val', 'test')

        Returns:
            Dictionary mapping names to MetricRecorder instances
        """
        return {name: cls(config) for name in names}

    @classmethod
    def from_yaml(cls, yaml_path: str) -> MetricRecorder:
        """
        Create a recorder instance from a YAML configuration file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Initialized MetricRecorder instance
        """
        config = MetricConfig.from_yaml(yaml_path)
        return cls(config)

    @classmethod
    def from_yaml_multi(cls, yaml_path: str, *names: str) -> Dict[str, MetricRecorder]:
        """
        Create multiple recorder instances from a YAML configuration file.

        Args:
            yaml_path: Path to YAML configuration file
            *names: Names for the different instances

        Returns:
            Dictionary mapping names to MetricRecorder instances
        """
        config = MetricConfig.from_yaml(yaml_path)
        return cls.create_instances(config, *names)
