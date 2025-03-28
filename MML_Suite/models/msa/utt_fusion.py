from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from experiment_utils.global_state import get_current_exp_name, get_current_run_id
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.printing import get_console
from experiment_utils.utils import SafeDict, format_path_with_env, safe_detach
from modalities import Modality
from models.mixins import MultimodalMonitoringMixin
from models.msa.networks.classifier import FcClassifier
from models.msa.networks.lstm import LSTMEncoder
from models.msa.networks.textcnn import TextCNN
from models.protocols import MultimodalModelProtocol
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

console = get_console()


class UttFusionModel(Module, MultimodalMonitoringMixin, MultimodalModelProtocol):
    """
    Fusion model for multimodal sentiment analysis using LSTM and TextCNN encoders
    with a fully connected classifier for prediction.

    This model supports audio, video, and text modalities, allowing for modular
    replacement and missing data handling.
    """

    def __init__(
        self,
        netA: LSTMEncoder,
        netV: LSTMEncoder,
        netT: TextCNN,
        netC: FcClassifier,
        *,
        clip: Optional[float] = None,
        pretrained_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the UttFusionModel.

        Args:
            netA (LSTMEncoder): LSTM encoder for audio modality.
            netV (LSTMEncoder): LSTM encoder for video modality.
            netT (TextCNN): TextCNN encoder for text modality.
            netC (FcClassifier): Fully connected classifier for fused features.
            clip (Optional[float]): Gradient clipping value (if specified).
            pretrained_path (Optional[str]): Path to pretrained weights (if any).
        """
        super().__init__()
        self.netA = netA
        self.netV = netV
        self.netT = netT
        self.netC = netC
        self.clip = clip
        self.pretrained_path = pretrained_path

    def load_pretrained(self) -> None:
        """
        Load pretrained weights into the model.
        """

        if self.pretrained_path is not None:
            self.pretrained_path = format_path_with_env(self.pretrained_path)
            self.pretrained_path = self.pretrained_path.format_map(
                SafeDict(run_id=get_current_run_id(), exp_name=get_current_exp_name())
            )

            console.print(f"Loading pretrained weights from {self.pretrained_path}")
            state_dict = torch.load(self.pretrained_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict=state_dict["model_state_dict"])
        else:
            console.print("[bold red] WARNING: No pretrained weights loaded.[/]")
            raise ValueError("No pretrained weights loaded.")

    def get_encoder(self, modality: Modality | str) -> Module:
        """
        Get the encoder module for a specific modality.

        Args:
            modality (Modality | str): Modality identifier.

        Returns:
            Module: Corresponding encoder module.

        Raises:
            ValueError: If the modality is invalid or unsupported.
        """
        if isinstance(modality, str):
            modality = Modality.from_str(modality)
        match modality:
            case Modality.AUDIO:
                return self.netA
            case Modality.VIDEO:
                return self.netV
            case Modality.TEXT:
                return self.netT
            case _:
                raise ValueError(f"Unknown modality: {modality}")

    def forward(
        self,
        A: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
        *,
        is_embd_A: bool = False,
        is_embd_V: bool = False,
        is_embd_T: bool = False,
    ) -> torch.Tensor:
        """
        Perform a forward pass of the fusion model.

        Args:
            A (Optional[torch.Tensor]): Audio features or embeddings.
            V (Optional[torch.Tensor]): Video features or embeddings.
            T (Optional[torch.Tensor]): Text features or embeddings.
            is_embd_A (bool): Whether the audio input is pre-embedded.
            is_embd_V (bool): Whether the video input is pre-embedded.
            is_embd_T (bool): Whether the text input is pre-embedded.

        Returns:
            torch.Tensor: Prediction logits.

        Raises:
            AssertionError: If no input is provided or all inputs are pre-embedded.
        """
        assert not all((A is None, V is None, T is None)), "At least one of A, V, T must be provided"
        assert not all([is_embd_A, is_embd_V, is_embd_T]), "Cannot have all embeddings as True"

        a_embd = self.netA(A) if not is_embd_A and A is not None else A
        v_embd = self.netV(V) if not is_embd_V and V is not None else V
        t_embd = self.netT(T) if not is_embd_T and T is not None else T
        #
        # console.print(f"A Embd Shape: {a_embd.shape}")
        # console.print(f"V Embd Shape: {v_embd.shape}")
        # console.print("T Embd Shape: {t_embd.shape}")
        #
        fused = torch.cat([embd for embd in [a_embd, v_embd, t_embd] if embd is not None], dim=-1)
        logits = self.netC(fused)
        return logits

    def flatten_parameters(self) -> None:
        """
        Flatten parameters for RNN layers to optimize training performance.
        """
        self.netA.rnn.flatten_parameters()
        self.netV.rnn.flatten_parameters()

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: Optimizer,
        loss_functions: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform a single training step.

        Args:
            batch (Dict[str, Any]): Batch of input data and labels.
            optimizer (Optimizer): Optimizer for the model.
            loss_functions (LossFunctionGroup): Loss function group.
            device (torch.device): Computation device.
            metric_recorder (MetricRecorder): Metric recorder for evaluation.

        Returns:
            Dict[str, Any]: Training results including loss.
        """
        A, V, T, labels, _miss_type = (
            batch[Modality.AUDIO].to(device).float(),
            batch[Modality.VIDEO].to(device).float(),
            batch[Modality.TEXT].to(device).float(),
            batch["label"].to(device),
            batch["pattern_name"],
        )

        self.train()
        logits = self.forward(A, V, T)

        optimizer.zero_grad()
        loss = loss_functions(logits.squeeze(), labels.squeeze())["total_loss"]
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        optimizer.step()

        predictions = safe_detach(F.softmax(logits, dim=-1).argmax(dim=-1).squeeze())
        labels = safe_detach(labels.squeeze())

        metric_recorder.update_group_all(
            "classification", predictions=predictions, targets=labels, m_types=np.array(_miss_type)
        )
        return {"loss": loss.item()}

    def validation_step(
        self,
        batch: Dict[str, Any],
        loss_functions: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        return_test_info: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform a single validation step.

        Args:
            batch (Dict[str, Any]): Batch of input data and labels.
            loss_functions (LossFunctionGroup): Loss function group.
            device (torch.device): Computation device.
            metric_recorder (MetricRecorder): Metric recorder for evaluation.
            return_test_info (bool): Whether to return detailed test information.

        Returns:
            Dict[str, Any]: Validation results including loss and optional test info.
        """
        self.eval()

        all_predictions, all_labels, all_miss_types = [], [], []

        with torch.no_grad():
            A, V, T, labels, miss_type = (
                batch[Modality.AUDIO].to(device).float(),
                batch[Modality.VIDEO].to(device).float(),
                batch[Modality.TEXT].to(device).float(),
                batch["label"].to(device),
                batch["pattern_name"],
            )

            logits = self.forward(A, V, T)

            miss_types = np.array(miss_type)

            loss = loss_functions(logits.squeeze(), labels)["total_loss"]
            predictions = safe_detach(F.softmax(logits, dim=-1).argmax(dim=-1).squeeze())
            labels = safe_detach(labels.squeeze())

            metric_recorder.update_group_all(
                "classification", predictions=predictions, targets=labels, m_types=miss_types
            )

            if return_test_info:
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_miss_types.append(miss_type)

        self.train()

        if return_test_info:
            return {
                "loss": loss.item(),
                "predictions": safe_detach(predictions),
                "labels": labels,
                "miss_type": miss_types,
                "targets": labels,
                "logits": safe_detach(logits),
            }
        return {
            "loss": loss.item(),
            "predictions": safe_detach(predictions),
            "labels": labels,
            "miss_type": miss_types,
            "targets": labels,
            "logits": safe_detach(logits),
        }

    # def get_embeddings(self, dataloader: DataLoader, device: torch.device) -> Dict[Modality, np.ndarray]:
    #     """
    #     Get embeddings for all samples in a dataloader.

    #     Args:
    #         dataloader (DataLoader): DataLoader for the dataset.
    #         device (torch.device): Computation device.

    #     Returns:
    #         Dict[Modality, np.ndarray]: Dictionary of embeddings for each modality.
    #     """
    #     self.eval()
    #     embeddings = defaultdict(list)
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             A, V, T = (
    #                 batch[Modality.AUDIO].to(device).float(),
    #                 batch[Modality.VIDEO].to(device).float(),
    #                 batch[Modality.TEXT].to(device).float(),
    #             )
    #             a_embd = self.netA(A)
    #             v_embd = self.netV(V)
    #             t_embd = self.netT(T)

    #             for mod, embd in zip([Modality.AUDIO, Modality.VIDEO, Modality.TEXT], [a_embd, v_embd, t_embd]):
    #                 if embd is not None:
    #                     embeddings[mod].append(safe_detach(embd))
    #             embeddings["label"] += batch["label"]

    #     # embeddings: Dict[Modality, np.ndarray] = {mod: np.concatenate(embds) for mod, embds in embeddings.items()}

    #     return embeddings
