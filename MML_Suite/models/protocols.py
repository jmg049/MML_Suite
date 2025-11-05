from __future__ import annotations
from typing import Any, Dict, Protocol, Tuple

import numpy as np
import torch
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from modalities import Modality
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class MultimodalModelProtocol(Protocol):
    def get_encoder(self, modality: Modality) -> Module: ...

    def flatten_parameters(self) -> None: ...

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: Optimizer,
        criterion: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        **kwargs: Any,
    ) -> Dict[str, Any]: ...

    def validation_step(
        self,
        batch: Dict[str, Any],
        criterion: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        return_test_info: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]: ...

    def get_embeddings(
        self, dataloader: DataLoader, device: torch.device
    ) -> Tuple[Dict[Modality, np.ndarray], Dict[Modality, np.ndarray]] | Dict[Modality, np.ndarray]: ...


    def to(self, device: torch.device) -> MultimodalModelProtocol: ...