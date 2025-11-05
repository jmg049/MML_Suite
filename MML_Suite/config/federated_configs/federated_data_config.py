from dataclasses import dataclass, field
from typing import Literal, Optional

from config.data_config import DataConfig
from fed import DistributionStrategy


@dataclass
class FedDataConfig(DataConfig):
    """
    Configuration for federated data handling.
    This configuration is used to set up the data loading and preprocessing
    for federated learning tasks.
    """

    distribution_strategy: Literal["iid", "non_iid"] = "iid"
    task_type: Literal["classification", "multilabel_classification"] = "classification"
    global_fraction: float = 0.05
    alpha: Optional[float] = None