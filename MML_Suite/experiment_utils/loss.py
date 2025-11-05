from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from torch import Tensor, Type
from torch.nn import (
    Module,
    CrossEntropyLoss,
    NLLLoss,
    MSELoss,
    BCELoss,
    BCEWithLogitsLoss,
    L1Loss,
    SmoothL1Loss,
    KLDivLoss,
    HuberLoss,
    TripletMarginLoss,
    CosineEmbeddingLoss,
    MarginRankingLoss,
    MultiMarginLoss,
    SoftMarginLoss,
    MultiLabelMarginLoss,
    HingeEmbeddingLoss,
    PoissonNLLLoss,
    GaussianNLLLoss,
    CTCLoss,
)
from experiment_utils.logging import get_logger
from experiment_utils.printing import get_console
from cmam_loss import CMAMLoss
from loss_functions.contrastive_loss import ContrastiveTripletLoss, CombinedMSEContrastiveLoss

logger = get_logger()
console = get_console()


def resolve_criterion(criterion_name: str) -> Type[Module]:
    """
    Resolve loss criterion class from string name.

    Args:
        criterion_name: Name of the criterion (case-insensitive)

    Returns:
        Criterion class
    """
    criterion_map = {
        "cross_entropy": CrossEntropyLoss,
        "nll": NLLLoss,
        "mse": MSELoss,
        "bce": BCELoss,
        "bce_with_logits": BCEWithLogitsLoss,
        "l1": L1Loss,
        "smooth_l1": SmoothL1Loss,
        "kl_div": KLDivLoss,
        "huber": HuberLoss,
        "triplet": TripletMarginLoss,
        "cosine": CosineEmbeddingLoss,
        "margin_ranking": MarginRankingLoss,
        "multi_margin": MultiMarginLoss,
        "soft_margin": SoftMarginLoss,
        "multi_label_margin": MultiLabelMarginLoss,
        "hinge_embedding": HingeEmbeddingLoss,
        "poisson_nll": PoissonNLLLoss,
        "gaussian_nll": GaussianNLLLoss,
        "ctc": CTCLoss,
        "cmam": CMAMLoss,
        "contrastive_triplet": ContrastiveTripletLoss,
        "combined_mse_contrastive": CombinedMSEContrastiveLoss,
        "na": lambda x: x,
        "cycle": MSELoss,
    }

    criterion_name = criterion_name.lower()

    if criterion_name not in criterion_map:
        error_msg = f"Unknown criterion: {criterion_name}. Available criteria: {list(criterion_map.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Resolved criterion: {criterion_name}")
    return criterion_map[criterion_name]


@dataclass
class WeightedLossTerm:
    loss_fn: Module
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WeightedLossTerm:
        print(data)
        loss_name = data["loss_name"]
        loss_kwargs = data.get("loss_kwargs", {})
        weight = data.get("weight", 1.0)

        loss_fn_type = resolve_criterion(loss_name)

        l =  cls(loss_fn=loss_fn_type(**loss_kwargs), weight=weight)
        return l
    
    def __str__(self):
        return f"WeightedLossTerm: Weight={self.weight}, loss_fn = {self.loss_fn}"

    def __call__(self, inputs, targets, override_weight_with: Optional[float] = None, *args, **kwargs) -> Tensor:
        
        loss_value = self.loss_fn(inputs, targets, *args, **kwargs)
        if isinstance(loss_value, dict):
            loss_value = {
                k: v * self.weight if override_weight_with is None else v * override_weight_with
                for k, v in loss_value.items()
            }
            assert all(isinstance(v, Tensor) for v in loss_value.values()), "Loss values must be Tensors, got: {}".format(
                {k: type(v) for k, v in loss_value.items()}
            )
        else:
            loss_value = {
                "total_loss": (
                    loss_value * self.weight if override_weight_with is None else loss_value * override_weight_with
                )
            }
            assert isinstance(loss_value["total_loss"], Tensor), "Loss value must be a Tensor, got: {}".format(type(loss_value["total_loss"]))
        
        return loss_value


class LossFunctionGroup(Dict[str, WeightedLossTerm]):
    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> LossFunctionGroup:
        l_group = cls({key: WeightedLossTerm.from_dict(value) for key, value in data.items()})
        console.print(f"Created LossFunctionGroup with keys: {list(l_group.keys())}")

        return l_group

    def __str__(self):
        for k, v in self.items():
            print(f" - {k} - {v}")
        

    def __call__(
        self,
        inputs,
        targets,
        key: Optional[str | Set[str]] = None,
        override_weight_with: Optional[float] = None,
        return_key: bool = False,
        **kwargs,
    ) -> Tensor:
        losses = defaultdict(float)

        # update one specific loss term
        if key is not None:
            for loss_term, weighted_loss_term in self.items():
                if loss_term == key:
                    loss_value = weighted_loss_term(inputs, targets, override_weight_with, **kwargs)
                    if isinstance(loss_value, dict):
                        assert all(isinstance(v, Tensor) for v in loss_value.values()), "Loss values must be Tensors, got: {}".format(
                            {k: type(v) for k, v in loss_value.items()}
                        )
                        # console.print(f"I passed a single loss term: {loss_term} with keys: {list(loss_value.keys())}")
                    else:
                        assert isinstance(loss_value, Tensor), "Loss value must be a Tensor, got: {}".format(type(loss_value))
                        # console.print("I passed a single loss term")
                    for k, v in loss_value.items():
                        losses[k] += v
        # update all loss terms
        else:
            for loss_term, weighted_loss_term in self.items():
                loss_value = weighted_loss_term(inputs, targets, override_weight_with, **kwargs)
                for k, v in loss_value.items():
                    losses[k] += v

        return losses[key] if return_key and key else losses

    def __str__(self) -> str:
        return f"LossFunctionGroup({list(self.keys())})"
