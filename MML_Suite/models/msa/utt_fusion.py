from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from experiment_utils.global_state import get_current_exp_name, get_current_run_id
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.printing import get_console, print_info, print_warning
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
        self.load_pretrained()

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
            state_dict = torch.load(self.pretrained_path, map_location="cpu", weights_only=True)["model_state_dict"]
            netA_state_dict = {k.replace("netA.", ""): v for k, v in state_dict.items() if k.startswith("netA.")}
            netV_state_dict = {k.replace("netV.", ""): v for k, v in state_dict.items() if k.startswith("netV.")}
            netT_state_dict = {k.replace("netT.", ""): v for k, v in state_dict.items() if k.startswith("netT.")}

            self.netA.load_state_dict(netA_state_dict)
            self.netV.load_state_dict(netV_state_dict)
            self.netT.load_state_dict(netT_state_dict)
            console.print("[bold green] Pretrained weights loaded successfully.[/]")


        else:
            console.print("[bold red] WARNING: No pretrained weights loaded.[/]")
            # raise ValueError("No pretrained weights loaded.")

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

    def to(self, device):
        """
        Move the model to a specified device.

        Args:
            device: The target device (e.g., 'cpu' or 'cuda').
        """
        super().to(device)
        self.netA.to(device)
        self.netV.to(device)
        self.netT.to(device)
        self.netC.to(device)


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

        """

        a_embd = self.netA(A) if not is_embd_A and A is not None else A
        v_embd = self.netV(V) if not is_embd_V and V is not None else V
        t_embd = self.netT(T) if not is_embd_T and T is not None else T

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
            batch["pattern_names"],
        )

        if labels.numel() == 0:
            print_warning(console, "Empty labels encountered in training step.")
            raise ValueError("Empty labels encountered in training step.")

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
        if metric_recorder is not None:
            try:
                metric_recorder.update_group_all(
                "classification", predictions=predictions, targets=labels, m_types=np.array(_miss_type)
            )
            except Exception as e:
                print_warning(console,f"Failed to update metric recorder: {e}")
                print_warning(console, f"Predictions: {predictions}\nTargets: {labels}, m_types: {_miss_type}")
                raise e
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

        all_predictions, all_labels, all_miss_types, all_sample_ids = [], [], [], []


        with torch.no_grad():
            A, V, T, labels, miss_type, sample_ids = (
                batch[Modality.AUDIO].to(device).float(),
                batch[Modality.VIDEO].to(device).float(),
                batch[Modality.TEXT].to(device).float(),
                batch["label"].to(device),
                batch["pattern_names"],
                batch["sample_idx"],
            )

            ground_truth_embeddings = {
                Modality.AUDIO: self.netA(A) if A is not None else None,
                Modality.VIDEO: self.netV(V) if V is not None else None,
                Modality.TEXT: self.netT(T) if T is not None else None,
            }


            logits = self.forward(A, V, T)

            miss_types = np.array(miss_type)


            loss = loss_functions(logits.squeeze(), labels.squeeze())["total_loss"]
            predictions = safe_detach(F.softmax(logits, dim=-1).argmax(dim=-1).squeeze())
            labels = safe_detach(labels.squeeze())

            if metric_recorder is not None:
                metric_recorder.update_group_all(
                    "classification", predictions=predictions, targets=labels, m_types=miss_types
                )

            if return_test_info:
                all_predictions.append(predictions)
                all_labels.append(labels)
                all_miss_types.append(miss_type)
                all_sample_ids.append(sample_ids)

        self.train()

        if return_test_info:
            # Flatten the accumulated arrays for single batch processing
            flat_sample_ids = []
            flat_miss_types = []
            for ids_batch in all_sample_ids:
                if isinstance(ids_batch, (list, np.ndarray)):
                    flat_sample_ids.extend(ids_batch)
                else:
                    flat_sample_ids.append(ids_batch)
            for miss_batch in all_miss_types:
                if isinstance(miss_batch, (list, np.ndarray)):
                    flat_miss_types.extend(miss_batch)
                else:
                    flat_miss_types.append(miss_batch)

            return {
                "loss": loss.item(),
                "predictions": safe_detach(predictions),
                "labels": labels,
                "miss_type": flat_miss_types,
                "targets": labels,
                "logits": safe_detach(logits),
                "sample_ids": flat_sample_ids,
                "preds": safe_detach(predictions),  # For compatibility with older code
                "ground_truth_logits": safe_detach(logits),  # For compatibility with older code
                "ground_truth_embeddings": ground_truth_embeddings,

            }
        return {
            "loss": loss.item(),
            "predictions": safe_detach(predictions),
            "labels": labels,
            "miss_type": miss_types,
            "targets": labels,
            "logits": safe_detach(logits),
            "sample_ids": sample_ids,
            "preds": safe_detach(predictions),  # For compatibility with older code
            "ground_truth_logits": safe_detach(logits),  # For compatibility with older code
            "ground_truth_embeddings": ground_truth_embeddings
        }
