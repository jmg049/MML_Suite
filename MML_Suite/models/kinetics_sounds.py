from __future__ import annotations
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from experiment_utils.utils import safe_detach
from experiment_utils.printing import get_console
from models.mixins import MultimodalMonitoringMixin
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Any, Dict
from models.conv import ConvBlock
from data.kinetics_sounds import KineticsSounds as KineticsSoundDataset
from modalities import Modality
from torch.nn import Module, AvgPool2d, Linear, ReLU, Dropout, Module
from torch import Tensor

console = get_console()


class KineticsSoundsAudioEncoder(Module):
    def __init__(
        self,
        conv_block_one: ConvBlock,
        conv_block_two: ConvBlock,
        conv_block_three: ConvBlock,
        kernel_size_one: int | tuple[int] = (2, 2),
        kernel_size_two: int | tuple[int] = (4, 4),
        kernel_size_three: int | tuple[int] = (4, 8),
        dropout_one: float = 0.554,
        dropout_two: float = 0.336,
        fc_one_input_size: int = 512,
        fc_one_output_size: int = 64,
        fc_two_output_size: int = 64,
    ) -> None:
        super().__init__()
        self.conv_block_one = conv_block_one
        self.conv_block_two = conv_block_two
        self.conv_block_three = conv_block_three

        self.avg_pool_one = AvgPool2d(kernel_size=kernel_size_one)
        self.avg_pool_two = AvgPool2d(kernel_size=kernel_size_two)
        self.avg_pool_three = AvgPool2d(kernel_size=kernel_size_three)

        self.dropout_one = Dropout(dropout_one)
        self.dropout_two = Dropout(dropout_two)

        self.fc_one = Linear(fc_one_input_size, fc_one_output_size)
        self.fc_two = Linear(fc_one_output_size, fc_two_output_size)

        self.embedding_size = fc_two_output_size

        self.ReLU = ReLU()

    def get_embedding_size(self) -> int:
        return self.embedding_size

    def forward(self, audio: Tensor) -> Tensor:
        audio = audio.unsqueeze(1)
        x = self.conv_block_one(audio)
        x = self.avg_pool_one(x)
        x = self.conv_block_two(x)
        x = self.avg_pool_two(x)
        x = self.conv_block_three(x)
        x = self.avg_pool_three(x)

        x = x.view(x.size(0), -1)

        x = self.ReLU(x)
        x = self.dropout_one(x)
        x = self.fc_one(x)
        x = self.ReLU(x)
        x = self.dropout_two(x)
        x = self.fc_two(x)

        return x


class KineticsSoundsVideoEncoder(Module):
    def __init__(
        self, fc_one_input_size: int = 400, hidden_dim_one: int = 256, hidden_dim_two: int = 128, dropout: float = 0.56
    ):
        super().__init__()
        self.fc_one = Linear(fc_one_input_size, hidden_dim_one)
        self.fc_two = Linear(hidden_dim_one, hidden_dim_two)
        self.dropout = Dropout(dropout)
        self.embedding_size = hidden_dim_two
        self.ReLU = ReLU()

    def get_embedding_size(self) -> int:
        return self.embedding_size

    def forward(self, video: Tensor) -> Tensor:
        x = self.fc_one(video)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc_two(x)
        x = self.ReLU(x)

        return x


class KineticsSounds(Module, MultimodalMonitoringMixin):
    def __init__(
        self,
        audio_encoder: KineticsSoundsAudioEncoder,
        video_encoder: KineticsSoundsVideoEncoder,
        hidden_dim_one: int,
        hidden_dim_two: int,
        dropout: float = 0.38,
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.fc_one = Linear(audio_encoder.get_embedding_size() + video_encoder.get_embedding_size(), hidden_dim_one)
        self.ReLU = ReLU()
        self.embedding_size = hidden_dim_one
        self.fc_two = Linear(hidden_dim_one, hidden_dim_two)
        self.fc_out = Linear(hidden_dim_two, KineticsSoundDataset.NUM_CLASSES)
        self.dropout = Dropout(dropout)

    def get_embedding_size(self) -> int:
        return self.embedding_size

    def forward(
        self, A: Optional[Tensor] = None, V: Optional[Tensor] = None, is_embd_A: bool = False, is_embd_V: bool = False
    ) -> Tensor:
        # assert not all((A is None, V is None)), "At least one of A, V must be provided"
        # assert not all([is_embd_A, is_embd_V]), "Only one of is_embd_A, is_embd_V can be True"

        A = A if A is not None else torch.zeros(V.size(0), self.audio_encoder.get_embedding_size())
        V = V if V is not None else torch.zeros(A.size(0), self.video_encoder.get_embedding_size())

        audio = self.audio_encoder(A) if not is_embd_A else A
        video = self.video_encoder(V) if not is_embd_V else V

        # print(f"Audio Shape: {audio.shape}")
        # print(f"Video Shape: {video.shape}")

        fused = torch.cat((audio, video), dim=1)
        x = self.fc_one(fused)
        x = self.ReLU(x)
        x = self.dropout(x)

        x = self.fc_two(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc_out(x)

        return x

    def get_encoder(self, modality: Modality) -> Module:
        match modality:
            case Modality.AUDIO:
                return self.audio_encoder
            case Modality.VIDEO:
                return self.video_encoder
            case _:
                raise ValueError(f"Invalid modality: {modality}")

    def train_step(
        self,
        batch: Dict[Any, Any],
        optimizer: Optimizer,
        loss_functions: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        **kwargs,
    ) -> Dict[str, Any]:
        A, V, labels, miss_type = (
            batch[Modality.AUDIO].to(device).float(),
            batch[Modality.VIDEO].to(device).float(),
            batch["labels"].to(device),
            batch["pattern_name"],
        )

        self.train()
        optimizer.zero_grad()
        logits = self.forward(A, V)

        loss = loss_functions(logits, labels)["total_loss"]
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        miss_type = np.array(miss_type)

        metric_recorder.update_group_all("classification", predictions=predictions, targets=labels, m_types=miss_type)
        return {
            "loss": loss.item(),
        }

    def validation_step(
        self,
        batch: Dict[Any, Any],
        loss_functions: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        return_test_info: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        self.eval()

        with torch.no_grad():
            A, V, labels, miss_type, sample_ids = (
                batch[Modality.AUDIO].to(device).float(),
                batch[Modality.VIDEO].to(device).float(),
                batch["labels"].to(device),
                batch["pattern_name"],
                batch["sample_idx"],
            )

            logits = self.forward(A, V)
            loss = loss_functions(logits, labels)["total_loss"]
            predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            miss_type = np.array(miss_type)

            metric_recorder.update_group_all(
                "classification", predictions=predictions, targets=labels, m_types=miss_type
            )

        return {
            "loss": loss.item(),
            "predictions": predictions,
            "targets": labels,
            "logits": safe_detach(logits),
            "miss_type": miss_type,
            "sample_ids": sample_ids,
        }

    def get_embeddings(self, dataloader: DataLoader, device: torch.device) -> Dict[Modality, np.ndarray]:
        """
        Extract embeddings from the model for a given dataloader.

        Args:
            dataloader (DataLoader): DataLoader for extracting embeddings.
            device (torch.device): Device to perform computations.

        Returns:
            Dict[Modality, np.ndarray]: Extracted embeddings for audio and image.
        """
        embeddings = defaultdict(list)
        self.to(device)
        self.eval()

        for batch in dataloader:
            with torch.no_grad():
                A, V, miss_type = (
                    batch[Modality.AUDIO],
                    batch[Modality.VIDEO],
                    batch["pattern_name"],
                )

                # Filter to full modality availability
                miss_type = np.array(miss_type)
                A = A[miss_type == KineticsSoundDataset.get_full_modality()]
                V = V[miss_type == KineticsSoundDataset.get_full_modality()]
                A = A.unsqueeze(1)
                audio_embedding = self.audio_encoder(A.to(device).float())
                video_embedding = self.video_encoder(V.to(device).float())

                embeddings[Modality.AUDIO].append(safe_detach(audio_embedding))
                embeddings[Modality.VIDEO].append(safe_detach(video_embedding))
                embeddings["label"] += batch["labels"]

        return embeddings
