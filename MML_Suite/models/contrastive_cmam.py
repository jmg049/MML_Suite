from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, OrderedDict
from torch import Tensor
import numpy as np
import torch
from models import L2Normalization
from experiment_utils.utils import safe_detach
from experiment_utils.loss import LossFunctionGroup
from loss_functions.contrastive_loss import CombinedMSEContrastiveLoss
from experiment_utils.metric_recorder import MetricRecorder
from modalities import Modality
from models.protocols import MultimodalModelProtocol
from torch.nn import (
    BatchNorm1d,
    Dropout,
    Identity,
    Linear,
    Module,
    ReLU,
    Sequential,
)
from torch.utils.data import DataLoader
from experiment_utils.printing import get_console
from torch.optim import Optimizer

console = get_console()


class AssociationNetwork(Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, batch_norm: bool = False, dropout: float = 0.0
    ) -> None:
        super(AssociationNetwork, self).__init__()
        self.input_size = input_size
        self.assoc_net = Sequential(
            Linear(input_size, hidden_size),
            BatchNorm1d(hidden_size) if batch_norm else Identity(),
            ReLU(),
            Dropout(dropout) if dropout > 0.0 else Identity(),
            Linear(hidden_size, output_size),
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> AssociationNetwork:
        return AssociationNetwork(
            input_size=data["input_size"],
            hidden_size=data["hidden_size"],
            output_size=data["output_size"],
            batch_norm=data.get("batch_norm", False),
            dropout=data.get("dropout", 0.0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.assoc_net(x)


class ContrastiveSimpleCMAM(Module):
    def __init__(
        self,
        association_network: AssociationNetwork,
        input_modalities: list[Modality],
        target_modality: Modality,
        *,
        fusion_fn: str = "concat",
        grad_clip: float = 0.0,
        labels_key: str = "labels",
        mse_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        margin: float = 1.0,
        use_adaptive_margin: bool = True,
        distance_function: str = "euclidean",
        **kwargs,
    ) -> None:
        super(ContrastiveSimpleCMAM, self).__init__()
        self.association_network = association_network
        self.input_modalities = input_modalities

        match fusion_fn.lower():
            case "concat":
                self.fusion_fn = torch.cat
            case "sum":
                self.fusion_fn = torch.sum
            case "mean":
                self.fusion_fn = torch.mean
            case _:
                raise ValueError(f"Unknown fusion function: {fusion_fn}")

        self.target_modality = target_modality
        self.grad_clip = grad_clip
        self.labels_key = labels_key

        self.do_normalisation = False
        self.norm_layer = L2Normalization() if self.do_normalisation else None

        # Initialize combined loss function
        self.loss_function = CombinedMSEContrastiveLoss(
            mse_weight=mse_weight,
            contrastive_weight=contrastive_weight,
            margin=margin,
            use_adaptive_margin=use_adaptive_margin,
            distance_function=distance_function,
        )

    def load_encoder_state_for(self, encoders_state: Dict[Modality, dict]) -> None:
        for modality, state in encoders_state.items():
            self.encoders[str(modality)].load_state_dict(state)
            console.print(f"Loaded state for {modality}")

    def display(self) -> str:
        assoc_params = sum(p.numel() for p in self.association_network.parameters())
        assoc_params_size_mb = assoc_params * 4 / 1024 / 1024
        return f"Contrastive CMAM Model: \n\tAssociation Network Parameters: {assoc_params} ({assoc_params_size_mb:.2f} MB)"

    @property
    def parameters_size_bytes(self) -> int:
        """Returns the size of the C-MAM model's parameters in bytes."""
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def to(self, device):
        super().to(device)
        self.association_network.to(device)
        self.loss_function.to(device)
        return self

    def forward(self, fused: Tensor) -> Tensor:
        if self.norm_layer is not None:
            fused = self.norm_layer(fused)
        return self.association_network(fused)

    def train_step(
        self,
        batch: Dict[Modality, Tensor],
        loss_functions: LossFunctionGroup,
        optimizer: Optimizer,
        device: torch.device,
        trained_model: MultimodalModelProtocol,
        metric_recorder: MetricRecorder,
        *,
        epoch: int = 0,
    ):
        """
        Training step using combined MSE + Contrastive loss.
        
        Args:
            batch: Batch data containing modalities and labels
            loss_functions: Loss function group (ignored, using internal combined loss)
            optimizer: Optimizer for C-MAM parameters
            device: Device to run on
            trained_model: Pre-trained base model for encoding
            metric_recorder: Metric recorder for tracking performance
            epoch: Current epoch number
        """
        self.train()
        self.to(device)

        target_modality = batch[self.target_modality].float().to(device)
        input_modalities = {modality: batch[modality].float().to(device) for modality in self.input_modalities}

        other_modalities = {
            k: v
            for k, v in batch.items()
            if isinstance(k, Modality) and k != self.target_modality and k not in self.input_modalities
        }

        try:
            labels = batch[self.labels_key].to(device)
        except KeyError as _:
            console.error(f"Failed to find key 'labels' in batch, available keys: {batch.keys()}")
            exit(1)

        try:
            miss_type = batch["pattern_names"]
        except KeyError as _:
            try:
                miss_type = batch["pattern_name"]
            except KeyError as e:
                raise e

        # Get the target embedding without computing gradients
        with torch.no_grad():
            trained_model.to(device)
            trained_model.eval()
            trained_encoder = trained_model.get_encoder(self.target_modality)
            target_embd = trained_encoder(target_modality.to(device))

            input_embds = OrderedDict(
                {m: trained_model.get_encoder(m)(input_modalities[m]) for m in self.input_modalities}
            )

        # Ensure trained_model's parameters do not require gradients
        for param in trained_model.parameters():
            param.requires_grad = False
        
        optimizer.zero_grad()
        fused = self.fusion_fn([input_embds[m] for m in input_embds], dim=1)
        
        # Forward pass through CMAM
        rec_embd = self.forward(fused)

        # Compute combined MSE + Contrastive loss
        loss_dict = self.loss_function(
            predictions=rec_embd,
            targets=target_embd,
            labels=labels,
        )
        total_loss = loss_dict["total_loss"]
        
        # Backward pass
        total_loss.backward()

        # Optional gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        optimizer.step()

        # Prepare input for the pretrained model to get classification metrics
        other_modalities = {str(k)[0].upper(): torch.zeros_like(v.to(device=device)) for k, v in other_modalities.items()}
        encoder_data = {str(k)[0].upper(): input_embds[k].to(device=device) for k in self.input_modalities}
        is_embd = {f"is_embd_{str(i)[0].upper()}": True for i in self.input_modalities}
        is_embd[f"is_embd_{str(self.target_modality)[0].upper()}"] = True
        
        m_kwargs = {
            **encoder_data,
            **is_embd,
            f"{str(self.target_modality)[0]}": rec_embd.to(device=device),
            **other_modalities,
        }

        # Compute logits for metrics
        with torch.no_grad():
            logits = trained_model(**m_kwargs)
            if isinstance(logits, tuple):
                logits = logits[0]

        logits_transform = (
            trained_model.logits_transform if hasattr(trained_model, "logits_transform") else lambda x: x.argmax(dim=1)
        )

        predictions = logits_transform(logits)

        if predictions.ndim == 2:
            predictions = predictions.squeeze(-1)
        if labels.ndim == 2:
            labels = labels.squeeze(-1)

        if np.unique(miss_type).size != 1:
            console.warning(
                f"Multiple missing types detected in the batch: {np.unique(miss_type)}. "
                "This may lead to incorrect metric calculations."
            )
            raise ValueError(f"Multiple missing types detected in the batch. {np.unique(miss_type)}")

        metric_recorder.update_group_all("classification", predictions, labels, miss_type)
        metric_recorder.update_group_all("reconstruction", rec_embd, target_embd, miss_type)

        # Prepare output losses
        other_losses = {k: v.item() for k, v in loss_dict.items() if k != "total_loss"}

        return {
            "loss": total_loss.item(),
            "other_losses": other_losses,
        }

    def validation_step(
        self,
        batch: Dict[Modality, Tensor],
        loss_functions: LossFunctionGroup,
        device: torch.device,
        trained_model: MultimodalModelProtocol,
        metric_recorder: MetricRecorder,
        return_eval_data: bool = False,
    ):
        """
        Validation step using combined MSE + Contrastive loss.
        """
        self.eval()
        trained_model.eval()
        self.to(device)
        trained_model.to(device)
        
        with torch.no_grad():
            sample_ids = batch["sample_idx"]
            target_modality = batch[self.target_modality].float().to(device)
            input_modalities = {modality: batch[modality].float().to(device) for modality in self.input_modalities}

            other_modalities = {
                k: v
                for k, v in batch.items()
                if isinstance(k, Modality) and k != self.target_modality and k not in self.input_modalities
            }

            try:
                labels = batch[self.labels_key].to(device)
            except KeyError as _:
                console.error(f"Failed to find key 'labels' in batch, available keys: {batch.keys()}")
                exit(1)

            try:
                miss_type = batch["pattern_names"]
            except KeyError as _:
                try:
                    miss_type = batch["pattern_name"]
                except KeyError as e:
                    raise e

            # Get the target embedding
            trained_model.to(device)
            trained_model.eval()
            trained_encoder = trained_model.get_encoder(self.target_modality)
            target_embd = trained_encoder(target_modality.to(device))
            input_embds = OrderedDict(
                {m: trained_model.get_encoder(m)(input_modalities[m]) for m in self.input_modalities}
            )

            fused = self.fusion_fn([input_embds[m] for m in input_embds], dim=1)

            # Forward pass through CMAM
            rec_embd = self.forward(fused)

            # Compute combined loss
            loss_dict = self.loss_function(
                predictions=rec_embd,
                targets=target_embd,
                labels=labels,
            )
            total_loss = loss_dict["total_loss"]

            # Prepare input for classification metrics
            encoder_data = {str(k)[0].upper(): input_embds[k].to(device=device) for k in self.input_modalities}
            is_embd = {f"is_embd_{str(i)[0].upper()}": True for i in self.input_modalities}
            is_embd[f"is_embd_{str(self.target_modality)[0].upper()}"] = True
            other_modalities = {str(k)[0].upper(): torch.zeros_like(v.to(device=device)) for k, v in other_modalities.items()}

            m_kwargs = {
                **encoder_data,
                **is_embd,
                f"{str(self.target_modality)[0]}": rec_embd.to(device=device),
                **other_modalities,
            }

            # Get classification logits
            logits = trained_model(**m_kwargs)
            if isinstance(logits, tuple):
                logits = logits[0]

            logits_transform = (
                trained_model.logits_transform
                if hasattr(trained_model, "logits_transform")
                else lambda x: x.argmax(dim=1)
            )
            predictions = logits_transform(logits)

            if predictions.ndim == 2:
                predictions = predictions.squeeze(-1)
            if labels.ndim == 2:
                labels = labels.squeeze(-1)

            other_losses = {k: v.item() for k, v in loss_dict.items() if k != "total_loss"}

            miss_type = np.array(miss_type)
            metric_recorder.update_group_all(
                predictions=predictions, targets=labels, group_name="classification", m_types=miss_type
            )
            metric_recorder.update_group_all(
                predictions=rec_embd, targets=target_embd, group_name="reconstruction", m_types=miss_type
            )

            self.train()

            return {
                "loss": total_loss.item(),
                "losses": other_losses,
                "predictions": safe_detach(predictions),
                "labels": labels,
                "targets": np.array(safe_detach(labels)),
                "rec_embd": rec_embd,
                "target_embd": target_embd,
                "logits": safe_detach(logits),
                "miss_type": miss_type,
                "sample_ids": sample_ids,
                "preds": safe_detach(predictions),
                "ground_truth_embeddings": safe_detach(target_embd),
                "reconstructed_embeddings": safe_detach(rec_embd),
            }

    def get_embeddings(
        self,
        dataloader: DataLoader,
        trained_model,
        device: torch.device,
        out_fp: str
    ) -> Dict[Modality, np.ndarray]:
        """Get embeddings for analysis (same as SimpleCMAM)."""
        console.print("Getting embeddings...")
        self.to(device)
        self.eval()
        trained_model.to(device)

        batch_data = {
            self.labels_key: [],
            "obs_embd": [],             
            "target_embd": [],          
            "rec_embd": [],
        }
        
        for batch in dataloader:
            with torch.no_grad():
                target_modality = batch[self.target_modality].float().to(device)
                input_modalities = {
                    m: batch[m].float().to(device) for m in self.input_modalities
                }

                input_embds = OrderedDict(
                    (m, trained_model.get_encoder(m)(input_modalities[m]))
                    for m in self.input_modalities
                )

                z_obs = trained_model.get_encoder(self.target_modality)(torch.zeros_like(target_modality).to(device))
                z_gt = trained_model.get_encoder(self.target_modality)(target_modality)

                fused = self.fusion_fn(list(input_embds.values()), dim=1)
                z_hat = self.forward(fused)

                batch_data[self.labels_key].append(batch[self.labels_key].cpu().numpy())
                batch_data["obs_embd"].append(z_obs.cpu().numpy().astype(np.float32))
                batch_data["target_embd"].append(z_gt.cpu().numpy().astype(np.float32))
                batch_data["rec_embd"].append(z_hat.cpu().numpy().astype(np.float32))

        # Concatenate along batch dimension
        for k in batch_data:
            batch_data[k] = np.concatenate(batch_data[k], axis=0)

        out_path = Path(out_fp)
        out_dir = out_path.parent
        out_stem = out_path.stem + ".npy"

        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / out_stem.format(targ="z_zero"),    batch_data["obs_embd"])
        np.save(out_dir / out_stem.format(targ="z_gt"),   batch_data["target_embd"])
        np.save(out_dir / out_stem.format(targ="z_rec"),   batch_data["rec_embd"])
        np.save(out_dir / out_stem.format(targ="labels"), batch_data[self.labels_key]) 
        
        console.print(f"[bold green] Z_zero saved to {out_dir / out_stem.format(targ='z_zero')}")
        console.print(f"[bold green] Z_gt saved to {out_dir / out_stem.format(targ='z_gt')}")
        console.print(f"[bold green] Z_rec saved to {out_dir / out_stem.format(targ='z_rec')}")
        console.print(f"[bold green] Labels saved to {out_dir / out_stem.format(targ='labels')}")
        
        return {
            "rec_embd": batch_data["rec_embd"],
            "target_embd": batch_data["target_embd"],
            "obs_embd": batch_data["obs_embd"],
            "sample_ids": batch["sample_idx"].cpu().numpy(),
        }

    @property
    def parameters_size_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())