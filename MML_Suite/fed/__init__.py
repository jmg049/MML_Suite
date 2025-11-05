from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

# Export dataset split metrics functionality
from .dataset_split_metrics import DatasetSplitAnalyzer, DatasetSplitMetrics, ClientLabelDistribution

class DataSplitType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    
class FedDataCondition(Enum):
    IID = "iid"
    NON_IID = "non_iid"
    
class FedLearningType(Enum):
    CONGRUENT = "congruent"
    INCONGRUENT = "incongruent"



@dataclass
class ClientSelectionStrategy:
    strategy: Literal["random", "round_robin"]
    parameters: dict = None

    def __post_init__(self):
        if not isinstance(self.strategy, str):
            raise ValueError("Strategy must be a string.")
        if self.parameters is not None and not isinstance(self.parameters, dict) :
            raise ValueError("Parameters must be a dictionary.")
        
    @classmethod
    def from_dict(cls, data) -> ClientSelectionStrategy:
        """Create ClientSelectionStrategy from a dictionary."""
        return data
        
@dataclass
class Aggregation:
    method: Literal["fedavg"]
    parameters: dict

    def __post_init__(self):
        if not isinstance(self.method, str):
            raise ValueError("Method must be a string.")
        if not isinstance(self.parameters, dict):
            raise ValueError("Parameters must be a dictionary.")
    
    @classmethod
    def from_dict(cls, data) -> Aggregation:
        """Create Aggregation from a dictionary."""
        return data
        
@dataclass
class DistributionStrategy:
    strategy: Literal["iid", "non_iid"] = "iid"
    dirchlet_alpha: Optional[float] = None

    def __eq__(self, other: Literal["iid", "non_iid"]) -> bool:
        return self.strategy == other
    
