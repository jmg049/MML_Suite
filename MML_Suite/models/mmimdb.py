from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
from data.mmimdb import MMIMDb as MMIMDbDataset
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.utils import safe_detach
from modalities import Modality
from models.gates import GatedBiModalNetwork
from models.maxout import MaxOut
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, Module, Sequential
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class MLPGenreClassifier(Module):
    """
    Multi-Layer Perceptron classifier for genre classification in the MMIMDb dataset.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int) -> None:
        """
        Initialize the MLP classifier.

        Args:
            input_size (int): Input feature size.
            output_size (int): Number of output classes.
            hidden_size (int): Dimension of the hidden layers.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.net = Sequential(
            BatchNorm1d(input_size),
            MaxOut(input_size, hidden_size, use_bias=False),
            Dropout(p=0.5),
            BatchNorm1d(hidden_size),
            MaxOut(hidden_size, hidden_size, use_bias=False),
            Dropout(p=0.5),
            BatchNorm1d(hidden_size),
            Linear(hidden_size, output_size),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Perform a forward pass through the classifier.

        Args:
            tensor (Tensor): Input tensor.

        Returns:
            Tensor: Output logits.
        """
        return self.net(tensor)


class MMIMDbModalityEncoder(Module):
    """
    Modality encoder for MMIMDb dataset, used for both image and text modalities.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize the modality encoder.

        Args:
            input_dim (int): Input feature size.
            output_dim (int): Output embedding size.
        """
        super().__init__()
        self.net = Sequential(
            BatchNorm1d(input_dim),
            Linear(input_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the modality encoder.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Encoded tensor.
        """
        return self.net(x)


class GMUModel(Module):
    """
    Gated Multimodal Unit (GMU)-based model for multimodal learning on MMIMDb dataset.
    Combines image and text modalities for genre classification.
    """

    def __init__(
        self,
        image_encoder: MMIMDbModalityEncoder,
        text_encoder: MMIMDbModalityEncoder,
        gated_bimodal_network: GatedBiModalNetwork,
        classifier: MLPGenreClassifier,
        binary_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the GMUModel.

        Args:
            image_encoder (MMIMDbModalityEncoder): Encoder for the image modality.
            text_encoder (MMIMDbModalityEncoder): Encoder for the text modality.
            gated_bimodal_network (GatedBiModalNetwork): GMU for fusing modalities.
            classifier (MLPGenreClassifier): MLP classifier for final predictions.
            binary_threshold (float): Threshold for binary classification.
        """
        super().__init__()
        self.image_model = image_encoder
        self.text_model = text_encoder
        self.gmu = gated_bimodal_network
        self.mm_mlp = classifier
        self.binary_threshold = binary_threshold

    def logits_transform(self, logits: Tensor) -> Tensor:
        predictions = torch.nn.functional.sigmoid(logits).detach().cpu().numpy()
        predictions = (predictions > 0.5).astype(int)

        return predictions

    def get_encoder(self, modality: Modality) -> MMIMDbModalityEncoder:
        match modality:
            case Modality.IMAGE:
                return self.image_model
            case Modality.TEXT:
                return self.text_model
            case _:
                raise ValueError(f"Invalid modality: {modality}. Must be one of {[Modality.IMAGE, Modality.TEXT]}")

    def forward(
        self,
        I: Tensor,
        T: Tensor,
        *,
        is_embd_I: bool = False,
        is_embd_T: bool = False,
    ) -> Tensor:
        """
        Perform a forward pass through the model.

        Args:
            I (Tensor): Image input or embedding.
            T (Tensor): Text input or embedding.
            is_embd_I (bool): Whether the image input is pre-embedded.
            is_embd_T (bool): Whether the text input is pre-embedded.

        Returns:
            Tensor: Output logits.
        """

        image = self.image_model(I) if not is_embd_I else I
        text = self.text_model(T) if not is_embd_T else T

        z = self.gmu(image, text)
        logits = self.mm_mlp(z)

        return logits

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: Optimizer,
        loss_functions: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        epoch: int,
    ) -> Dict[str, Any]:
        """
        Perform a single training step.

        Args:
            batch (Dict[str, Any]): Batch of input data and labels.
            optimizer (Optimizer): Optimizer for training.
            criterion (LossFunctionGroup): Loss function for training.
            device (torch.device): Computation device.
            metric_recorder (MetricRecorder): Metric recorder for tracking performance.

        Returns:
            Dict[str, Any]: Dictionary containing the training loss.
        """
        I, T, labels, miss_type = (
            batch[Modality.IMAGE].to(device).float(),
            batch[Modality.TEXT].to(device).float(),
            batch["label"].to(device),
            batch["pattern_name"],
        )

        self.train()
        optimizer.zero_grad()

        logits = self.forward(I=I, T=T)
        loss = loss_functions(inputs=logits, targets=labels)["total_loss"]
        loss.backward()
        optimizer.step()

        predictions = safe_detach(torch.sigmoid(logits))
        predictions = (predictions > self.binary_threshold).astype(int)
        labels = safe_detach(labels)
        miss_type = np.array(miss_type)

        metric_recorder.update_group_all("classification", predictions=predictions, targets=labels, m_types=miss_type)
        return {"loss": loss.item()}

    def validation_step(
        self,
        batch: Dict[str, Any],
        loss_functions: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        return_test_info: bool = False,
        epoch: int = None,
    ) -> Dict[str, Any]:
        """
        Perform a single validation step.

        Args:
            batch (Dict[str, Any]): Batch of input data and labels.
            criterion (LossFunctionGroup): Loss function for validation.
            device (torch.device): Computation device.
            metric_recorder (MetricRecorder): Metric recorder for tracking performance.
            return_test_info (bool): Whether to return additional test information.

        Returns:
            Dict[str, Any]: Validation results, including loss and optionally predictions.
        """
        self.eval()
        with torch.no_grad():
            I, T, labels, miss_type = (
                batch[Modality.IMAGE].to(device).float(),
                batch[Modality.TEXT].to(device).float(),
                batch["label"].to(device),
                batch["pattern_name"],
            )

            logits = self.forward(I=I, T=T)
            loss = loss_functions(inputs=logits, targets=labels)["total_loss"]
            predictions = safe_detach(torch.sigmoid(logits))
            predictions = (predictions > self.binary_threshold).astype(int)

            labels = safe_detach(labels)
            miss_type = np.array(miss_type)

        metric_recorder.update_group_all("classification", predictions=predictions, targets=labels, m_types=miss_type)

        return {
            "loss": loss.item(),
            "targets": labels,
            "predictions": predictions,
            "logits": safe_detach(logits),
            "miss_type": miss_type,
        }

    def __str__(self) -> str:
        """
        Return a string representation of the model.

        Returns:
            str: Model components as a string.
        """
        return f"{self.image_model}\n{self.text_model}\n{self.gmu}\n{self.mm_mlp}"

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
                I, T, miss_type = (
                    batch[Modality.IMAGE],
                    batch[Modality.TEXT],
                    batch["pattern_name"],
                )

                # Filter to full modality availability
                miss_type = np.array(miss_type)
                I = I[miss_type == MMIMDbDataset.get_full_modality()]
                T = T[miss_type == MMIMDbDataset.get_full_modality()]

                I, T = I.to(device), T.to(device)
                I = I.float()
                T = T.float()

                image_embedding = self.image_model(I)
                text_embedding = self.text_model(T)

                embeddings[Modality.IMAGE].append(safe_detach(image_embedding))
                embeddings[Modality.TEXT].append(safe_detach(text_embedding))
                embeddings["label"] += batch["label"]

        return embeddings
