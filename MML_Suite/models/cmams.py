from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Optional, OrderedDict
from torch import Tensor
import numpy as np
import torch
from experiment_utils.loss import LossFunctionGroup
from cmam_loss import CMAMLoss
from config.resolvers import resolve_encoder
from experiment_utils.metric_recorder import MetricRecorder
from modalities import Modality
from models.msa.utt_fusion import UttFusionModel
from models.protocols import MultimodalModelProtocol
from torch.nn import (
    BatchNorm1d,
    Dropout,
    Identity,
    Linear,
    Module,
    ModuleDict,
    ModuleList,
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


class InputEncoders(Dict[Modality, Module]):
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> InputEncoders:
        return InputEncoders({k: v for k, v in data.items()})


class SimpleCMAM(Module):
    def __init__(
        self,
        association_network: AssociationNetwork,
        input_modalities: list[Modality],
        target_modality: Modality,
        *,
        fusion_fn: str = "concat",
        grad_clip: float = 0.0,
        labels_key: str = "labels",
        **kwargs,
    ) -> None:
        super(SimpleCMAM, self).__init__()
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

    def load_encoder_state_for(self, encoders_state: Dict[Modality, dict]) -> None:
        for modality, state in encoders_state.items():
            self.encoders[str(modality)].load_state_dict(state)
            console.print(f"Loaded state for {modality}")

    def display(self) -> str:
        assoc_params = sum(p.numel() for p in self.association_network.parameters())

        assoc_params_size_mb = assoc_params * 4 / 1024 / 1024

        return f"CMAM Model: \n\tAssociation Network Parameters: {assoc_params} ({assoc_params_size_mb:.2f} MB)"

    def to(self, device):
        super().to(device)
        self.association_network.to(device)
        return self

    def reset_metric_recorders(self):
        self.metric_recorder.reset()

    def forward(self, fused: Tensor) -> Tensor:
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
        self.train()
        self.to(device)

        target_modality = batch[self.target_modality].float().to(device)
        input_modalities = {modality: batch[modality].float().to(device) for modality in self.input_modalities}

        other_modalities = {
            k: v
            for k, v in batch.items()
            if isinstance(k, Modality) and k != self.target_modality and k not in self.input_modalities
        }

        mi_input_modalities = [v.clone() for k, v in input_modalities.items()]
        try:
            labels = batch[self.labels_key].to(device)
        except KeyError as _:
            console.error(f"Failed to find key 'labels' in batch, available keys: {batch.keys()}")
            exit(1)

        miss_type = batch["pattern_name"]

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

        # Zero the gradients
        optimizer.zero_grad()

        fused = self.fusion_fn([input_embds[m] for m in input_embds for m in input_embds])

        # Forward pass through CMAM
        rec_embd = self.forward(fused)

        # Compute reconstruction loss

        other_modalities = {str(k)[0].upper(): v.to(device=device) for k, v in other_modalities.items()}

        # Prepare input for the pretrained model
        encoder_data = {str(k)[0].upper(): input_embds[k].to(device=device) for k in self.input_modalities}

        is_embd = {f"is_embd_{str(i)[0].upper()}": True for i in self.input_modalities}
        is_embd[f"is_embd_{str(self.target_modality)[0].upper()}"] = True
        m_kwargs = {
            **encoder_data,
            **is_embd,
            f"{str(self.target_modality)[0]}": rec_embd.to(device=device),
            **other_modalities,
        }

        # Compute logits without torch.no_grad()
        with torch.no_grad():
            logits = trained_model(**m_kwargs)

        logits_transform = (
            trained_model.logits_transform if hasattr(trained_model, "logits_transform") else lambda x: x.argmax(dim=1)
        )

        predictions = logits_transform(logits)

        metric_recorder.update_group_all("classification", predictions, labels, miss_type)
        metric_recorder.update_group_all("reconstruction", rec_embd, target_embd, miss_type)

        # Total loss and backward pass
        loss_dict = loss_functions(
            inputs=rec_embd,
            targets=target_embd,
            originals=mi_input_modalities,
            reconstructed=rec_embd,
            forward_func=None,
            cls_logits=logits,
            cls_labels=labels,
        )
        total_loss = loss_dict["total_loss"]
        total_loss.backward()

        # Optional gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        optimizer.step()

        other_losses = {k: v.item() for k, v in loss_dict.items() if k != "total_loss"}

        return {
            "loss": total_loss.item(),
            "losses": other_losses,
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
        self.eval()
        trained_model.eval()
        self.to(device)
        trained_model.to(device)
        with torch.no_grad():
            target_modality = batch[self.target_modality].float().to(device)
            input_modalities = {modality: batch[modality].float().to(device) for modality in self.input_modalities}

            other_modalities = {
                k: v
                for k, v in batch.items()
                if isinstance(k, Modality) and k != self.target_modality and k not in self.input_modalities
            }

            mi_input_modalities = [v.clone() for k, v in input_modalities.items()]
            try:
                labels = batch[self.labels_key].to(device)
            except KeyError as _:
                console.error(f"Failed to find key 'labels' in batch, available keys: {batch.keys()}")
                exit(1)

            miss_type = batch["pattern_name"]

            # Get the target embedding without computing gradients
            trained_model.to(device)
            trained_model.eval()
            trained_encoder = trained_model.get_encoder(self.target_modality)
            target_embd = trained_encoder(target_modality.to(device))

            input_embds = OrderedDict(
                {m: trained_model.get_encoder(m)(input_modalities[m]) for m in self.input_modalities}
            )

            fused = self.fusion_fn([input_embds[m] for m in input_embds for m in input_embds])

            # Forward pass through CMAM
            rec_embd = self.forward(fused)

            # Compute reconstruction loss

            other_modalities = {str(k)[0].upper(): v.to(device=device) for k, v in other_modalities.items()}

            # Prepare input for the pretrained model
            encoder_data = {str(k)[0].upper(): input_embds[k].to(device=device) for k in self.input_modalities}

            is_embd = {f"is_embd_{str(i)[0].upper()}": True for i in self.input_modalities}
            is_embd[f"is_embd_{str(self.target_modality)[0].upper()}"] = True
            m_kwargs = {
                **encoder_data,
                **is_embd,
                f"{str(self.target_modality)[0]}": rec_embd.to(device=device),
                **other_modalities,
            }

            logits = trained_model(**m_kwargs)

            logits_transform = (
                trained_model.logits_transform
                if hasattr(trained_model, "logits_transform")
                else lambda x: x.argmax(dim=1)
            )
            predictions = logits_transform(logits)

            ## compute all the losses
            loss_dict = loss_functions(
                inputs=rec_embd,
                targets=target_embd,
                originals=mi_input_modalities,  ## None for now since they refer to the original and reconstructed data for Cyclic Consistency Loss
                reconstructed=rec_embd,
                forward_func=None,
                cls_logits=logits,
                cls_labels=labels,
            )

            total_loss = loss_dict["total_loss"]
            other_losses = {k: v.item() for k, v in loss_dict.items() if k != "total_loss"}

            miss_type = np.array(miss_type)
            metric_recorder.update_group_all(
                predictions=predictions, targets=labels, group_name="classification", m_types=miss_type
            )
            metric_recorder.update_group_all(
                predictions=rec_embd, targets=target_embd, group_name="reconstruction", m_types=miss_type
            )
            self.train()

            if return_eval_data:
                return {
                    "loss": total_loss.item(),
                    "other_losses": other_losses,
                    "predictions": predictions,
                    "labels": labels,
                    "rec_embd": rec_embd,
                    "target_embd": target_embd,
                }

            return {
                "loss": total_loss.item(),
                "losses": other_losses,
            }


class CMAM(Module):
    def __init__(
        self,
        input_encoders: InputEncoders,
        association_network: AssociationNetwork,
        target_modality: Modality,
        *,
        fusion_fn: str = "concat",
        grad_clip: float = 0.0,
        labels_key: str = "labels",
        **kwargs,
    ) -> None:
        super(CMAM, self).__init__()
        self.encoders = ModuleDict({str(k): v for k, v in input_encoders.items()})
        self.association_network = association_network

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

    def load_encoder_state_for(self, encoders_state: Dict[Modality, dict]) -> None:
        for modality, state in encoders_state.items():
            self.encoders[str(modality)].load_state_dict(state)
            console.print(f"Loaded state for {modality}")

    def display(self) -> str:
        ## get parameter counts for each component of the model
        encoder_params = {
            modality: sum(p.numel() for p in encoder.parameters()) for modality, encoder in self.encoders.items()
        }
        assoc_params = sum(p.numel() for p in self.association_network.parameters())

        encoder_params_size_mb = sum(encoder_params.values()) * 4 / 1024 / 1024
        total_params = sum(encoder_params.values()) + assoc_params

        assoc_params_size_mb = assoc_params * 4 / 1024 / 1024

        return f"CMAM Model: \n\tTotal Parameters: {total_params} \n\tEncoder Parameters: {encoder_params} ({encoder_params_size_mb:.2f} MB) \n\tAssociation Network Parameters: {assoc_params} ({assoc_params_size_mb:.2f} MB)"

    def to(self, device):
        super().to(device)
        self.encoders.to(device)
        self.association_network.to(device)
        return self

    def reset_metric_recorders(self):
        self.metric_recorder.reset()

    def forward(self, modalities: Dict[Modality, Tensor]) -> Tensor:
        embeddings = [encoder(data) for encoder, data in zip(self.encoders.values(), modalities.values())]
        z = self.fusion_fn(embeddings, dim=1)
        return self.association_network(z)

    def get_embeddings(
        self,
        dataloader: DataLoader,
        trained_model,
        device: torch.device,
    ) -> Dict[Modality, np.ndarray]:
        console.print("Getting embeddings...")
        self.to(device)
        self.eval()
        trained_model.to(device)

        batch_data = defaultdict(list)

        for batch in dataloader:
            with torch.no_grad():
                target_modality = batch[self.target_modality].float().to(device)
                input_modalities = {
                    modality: batch[Modality.from_str(modality)].float().to(device) for modality in self.encoders
                }
                labels = batch[self.labels_key].to(device)

                ## get the target
                trained_model.to(device)
                target_modality = target_modality.to(device)
                trained_encoder = trained_model.get_encoder(self.target_modality)
                target_embd = trained_encoder(target_modality)
                rec_embd = self.forward(input_modalities)

                batch_data[self.labels_key].append(labels.cpu().numpy())
                batch_data["rec_embd"].append(rec_embd.cpu().numpy())
                batch_data["target_embd"].append(target_embd.cpu().numpy())

        labels = np.concatenate(batch_data[self.labels_key], axis=0)
        rec_embd = np.concatenate(batch_data["rec_embd"], axis=0)
        target_embd = np.concatenate(batch_data["target_embd"], axis=0)

        console.print(f"[bold green]Embeddings retrieved![/] Rec: {rec_embd.shape}, Target: {target_embd.shape}")

        return {
            self.labels_key: labels,
            "rec_embd": rec_embd,
            "target_embd": target_embd,
        }

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
        self.train()
        self.to(device)

        target_modality = batch[self.target_modality].float().to(device)
        input_modalities = {
            modality: batch[Modality.from_str(modality)].float().to(device) for modality in self.encoders
        }

        other_modalities = {
            k: v
            for k, v in batch.items()
            if isinstance(k, Modality) and k != self.target_modality and str(k) not in self.encoders
        }

        mi_input_modalities = [v.clone() for k, v in input_modalities.items()]
        try:
            labels = batch[self.labels_key].to(device)
        except KeyError as _:
            console.error(f"Failed to find key 'labels' in batch, available keys: {batch.keys()}")
            exit(1)

        miss_type = batch["pattern_name"]

        # Get the target embedding without computing gradients
        with torch.no_grad():
            trained_model.to(device)
            trained_model.eval()
            trained_encoder = trained_model.get_encoder(self.target_modality)
            target_embd = trained_encoder(target_modality.to(device))

        # Ensure trained_model's parameters do not require gradients
        for param in trained_model.parameters():
            param.requires_grad = False

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through CMAM
        rec_embd = self.forward(input_modalities)

        # Compute reconstruction loss

        other_modalities = {str(k)[0]: v.to(device=device) for k, v in other_modalities.items()}

        # Prepare input for the pretrained model
        encoder_data = {str(k)[0].upper(): batch[Modality.from_str(k)].to(device=device) for k in self.encoders.keys()}
        m_kwargs = {
            **encoder_data,
            f"{str(self.target_modality)[0]}": rec_embd.to(device=device),
            f"is_embd_{str(self.target_modality)[0]}": True,
            **other_modalities,
        }
        # Compute logits without torch.no_grad()
        logits = trained_model(**m_kwargs)

        logits_transform = (
            trained_model.logits_transform if hasattr(trained_model, "logits_transform") else lambda x: x.argmax(dim=1)
        )

        predictions = logits_transform(logits)

        metric_recorder.update_group_all("classification", predictions, labels, miss_type)
        metric_recorder.update_group_all("reconstruction", rec_embd, target_embd, miss_type)

        # Total loss and backward pass
        loss_dict = loss_functions(
            inputs=rec_embd,
            targets=target_embd,
            originals=mi_input_modalities,
            reconstructed=rec_embd,
            forward_func=None,
            cls_logits=logits,
            cls_labels=labels,
        )
        total_loss = loss_dict["total_loss"]
        total_loss.backward()

        # Optional gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        optimizer.step()

        other_losses = {k: v.item() for k, v in loss_dict.items() if k != "total_loss"}

        return {
            "loss": total_loss.item(),
            "losses": other_losses,
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
        self.eval()
        trained_model.eval()
        self.to(device)
        trained_model.to(device)
        with torch.no_grad():
            target_modality = batch[self.target_modality].float().to(device)
            input_modalities = {
                modality: batch[Modality.from_str(modality)].float().to(device) for modality in self.encoders
            }
            other_modalities = {
                k: v
                for k, v in batch.items()
                if isinstance(k, Modality) and k != self.target_modality and str(k) not in self.encoders
            }

            mi_input_modalities = [v.clone() for k, v in input_modalities.items()]
            miss_type = batch["pattern_name"]  ## should be a list of the same string/miss_type
            labels = batch[self.labels_key].to(device)

            ## get the target
            trained_model.to(device)
            target_modality = target_modality.to(device)
            trained_encoder = trained_model.get_encoder(self.target_modality)
            target_embd = trained_encoder(target_modality)
            rec_embd = self.forward(input_modalities)

            encoder_data = {
                str(k)[0].upper(): batch[Modality.from_str(k)].to(device=device) for k in self.encoders.keys()
            }

            other_modalities = {str(k)[0]: v.to(device=device) for k, v in other_modalities.items()}

            m_kwargs = {
                **encoder_data,
                f"{str(self.target_modality)[0]}": rec_embd.to(device=device),
                f"is_embd_{str(self.target_modality)[0]}": True,
                **other_modalities,
            }
            logits = trained_model(**m_kwargs)

            logits_transform = (
                trained_model.logits_transform
                if hasattr(trained_model, "logits_transform")
                else lambda x: x.argmax(dim=1)
            )
            predictions = logits_transform(logits)

            ## compute all the losses
            loss_dict = loss_functions(
                inputs=rec_embd,
                targets=target_embd,
                originals=mi_input_modalities,  ## None for now since they refer to the original and reconstructed data for Cyclic Consistency Loss
                reconstructed=rec_embd,
                forward_func=None,
                cls_logits=logits,
                cls_labels=labels,
            )

            total_loss = loss_dict["total_loss"]
            other_losses = {k: v.item() for k, v in loss_dict.items() if k != "total_loss"}

            miss_type = np.array(miss_type)
            metric_recorder.update_group_all(
                predictions=predictions, targets=labels, group_name="classification", m_types=miss_type
            )
            metric_recorder.update_group_all(
                predictions=rec_embd, targets=target_embd, group_name="reconstruction", m_types=miss_type
            )
            self.train()

            if return_eval_data:
                return {
                    "loss": total_loss.item(),
                    "other_losses": other_losses,
                    "predictions": predictions,
                    "labels": labels,
                    "rec_embd": rec_embd,
                    "target_embd": target_embd,
                }

            return {
                "loss": total_loss.item(),
                "losses": other_losses,
            }

    def incongruent_train_step(
        self,
        batch: dict[Tensor],
        labels: Tensor,
        criterion: Module,
        optimizer: Optimizer,
        device: torch.device,
        mm_model: MultimodalModelProtocol,
    ) -> dict[str, Any]:
        self.train()
        mm_model.train()
        self.to(device=device)
        mm_model.to(device=device)

        input_modalities = {
            modality: batch[Modality.from_str(modality)].float().to(device) for modality in self.encoders
        }
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through CMAM
        rec_embd = self.forward(input_modalities)

        # Prepare input for the pretrained model
        encoder_data = {str(k)[0].upper(): batch[Modality.from_str(k)].to(device=device) for k in self.encoders.keys()}
        m_kwargs = {
            **encoder_data,
            f"{str(self.target_modality)[0]}": rec_embd.to(device=device),
            f"is_embd_{str(self.target_modality)[0]}": True,
        }
        logits = mm_model(**m_kwargs, device=device)
        predictions = logits.argmax(dim=1)

        loss = criterion(logits, labels)

        if self.binarize:
            (
                binary_preds,
                binary_truth,
                non_zeros_mask,
            ) = UttFusionModel.msa_binarize(predictions.cpu().numpy(), labels)

            ## should be unnecessary now
            # binary_preds = de_device(binary_preds)
            # binary_truth = de_device(binary_truth)
            # non_zeros_mask = de_device(non_zeros_mask)

            # Calculate metrics for all elements (including zeros)
            binary_metrics = self.metric_recorder.calculate_metrics(predictions=binary_preds, targets=binary_truth)

            # Calculate metrics for non-zero elements only using the non_zeros_mask
            non_zeros_binary_preds = binary_preds[non_zeros_mask]
            non_zeros_binary_truth = binary_truth[non_zeros_mask]

            non_zero_metrics = self.metric_recorder.calculate_metrics(
                predictions=non_zeros_binary_preds,
                targets=non_zeros_binary_truth,
            )

            # Store the metrics in a dictionary
            metrics = {}

            for k, v in binary_metrics.items():
                metrics[f"HasZero_{k}"] = v

            for k, v in non_zero_metrics.items():
                metrics[f"NonZero_{k}"] = v
        else:
            metrics = self.metric_recorder.calculate_metrics(predictions, labels)

        ## TODO : Add in "sigmoid" functionality

        # Optional gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        optimizer.step()

        return {
            "loss": loss.item(),
            **metrics,
        }

    def incongruent_evaluate(
        self,
        batch: Dict[Modality, Tensor],
        labels: Tensor,
        criterion: Module,
        device: torch.device,
        mm_model: MultimodalModelProtocol,
    ):
        self.eval()
        mm_model.eval()
        self.to(device)
        mm_model.to(device)
        with torch.no_grad():
            input_modalities = {
                modality: batch[Modality.from_str(modality)].float().to(device) for modality in self.encoders
            }
            miss_type = batch["miss_type"]

            labels = labels.to(device)

            rec_embd = self.forward(input_modalities)

            encoder_data = {
                str(k)[0].upper(): batch[Modality.from_str(k)].to(device=device) for k in self.encoders.keys()
            }
            m_kwargs = {
                **encoder_data,
                f"{str(self.target_modality)[0]}": rec_embd.to(device=device),
                f"is_embd_{str(self.target_modality)[0]}": True,
            }
            logits = mm_model(**m_kwargs, device=device)
            predictions = logits.argmax(dim=1)

            ## compute all the losses
            loss = criterion(logits, labels)
            if self.binarize:
                (
                    binary_preds,
                    binary_truth,
                    non_zeros_mask,
                ) = UttFusionModel.msa_binarize(predictions.cpu().numpy(), labels)

                # should be unnecessary now
                # binary_preds = de_device(binary_preds)
                # binary_truth = de_device(binary_truth)
                # non_zeros_mask = de_device(non_zeros_mask)

                miss_types = np.array(miss_type)
                metrics = {}
                for miss_type in set(miss_types):
                    mask = miss_types == miss_type

                    # Apply the mask to get values for the current miss type
                    binary_preds_masked = binary_preds[mask]
                    binary_truth_masked = binary_truth[mask]
                    non_zeros_mask_masked = non_zeros_mask[mask]

                    # Calculate metrics for all elements (including zeros)
                    binary_metrics = self.metric_recorder.calculate_metrics(
                        predictions=binary_preds_masked, targets=binary_truth_masked
                    )

                    # Calculate metrics for non-zero elements only using the non_zeros_mask
                    non_zeros_binary_preds_masked = binary_preds_masked[non_zeros_mask_masked]
                    non_zeros_binary_truth_masked = binary_truth_masked[non_zeros_mask_masked]

                    non_zero_metrics = self.metric_recorder.calculate_metrics(
                        predictions=non_zeros_binary_preds_masked,
                        targets=non_zeros_binary_truth_masked,
                    )

                    # Store metrics in the metrics dictionary
                    for k, v in binary_metrics.items():
                        metrics[f"HasZero_{k}_{miss_type.replace('z', '').upper()}"] = v

                    for k, v in non_zero_metrics.items():
                        metrics[f"NonZero_{k}_{miss_type.replace('z', '').upper()}"] = v
            else:
                metrics = self.metric_recorder.calculate_metrics(predictions, labels)
                miss_type = np.array(miss_type)

                for m_type in set(miss_type):
                    mask = miss_type == m_type
                    mask_preds = predictions[mask]
                    mask_labels = labels[mask]
                    mask_metrics = self.metric_recorder.calculate_metrics(
                        predictions=mask_preds,
                        targets=mask_labels,
                        skip_metrics=["ConfusionMatrix"],
                    )
                    for k, v in mask_metrics.items():
                        metrics[f"{k}_{m_type.replace('z', '').upper()}"] = v
            self.train()

            return {
                "loss": loss.item(),
                **metrics,
            }


class DualCMAM(Module):
    """
    Given a single modality this C-MAM will reconstruct the embeddings of two other modalities.
    """

    def __init__(
        self,
        input_encoder_info: dict[Modality, dict[str, Any]],
        shared_encoder_output_size: int,
        decoder_hidden_size: int,
        target_modality_one_embd_size: int,
        target_modality_two_embd_size: int,
        input_modality: Modality,
        target_modality_one: Modality,
        target_modality_two: Modality,
        metric_recorder,
        *,
        dropout: float = 0.1,
        grad_clip: float = 0.0,
        binarize: bool = False,
    ):
        super(DualCMAM, self).__init__()
        self.target_modality_one = target_modality_one
        self.target_modality_two = target_modality_two
        self.input_modality = input_modality
        encoders = []
        for modality, encoder_params in input_encoder_info.items():
            if isinstance(encoder_params, Module):
                encoders.append(encoder_params)
                continue
            encoder_cls = resolve_encoder(encoder_params["name"])
            encoder_params.pop("name")
            encoders.append(encoder_cls(**encoder_params))

        self.input_encoder = encoders[0]

        self.decoders = ModuleList(
            [
                Sequential(
                    Linear(shared_encoder_output_size, decoder_hidden_size),
                    ReLU(),
                    Dropout(dropout),
                    Linear(decoder_hidden_size, target_modality_one_embd_size),
                ),
                Sequential(
                    Linear(shared_encoder_output_size, decoder_hidden_size),
                    ReLU(),
                    Dropout(dropout),
                    Linear(decoder_hidden_size, target_modality_two_embd_size),
                ),
            ]
        )

        self.grad_clip = grad_clip
        self.metric_recorder = metric_recorder
        self.binarize = binarize

    def reset_metric_recorders(self):
        self.metric_recorder.reset()

    def to(self, device):
        super().to(device)
        self.input_encoder.to(device)
        self.decoders.to(device)
        return self

    def forward(self, input_modality: Tensor) -> tuple[Tensor, Tensor]:
        input_embd = self.input_encoder(input_modality)
        reconstructed_embd_one = self.decoders[0](input_embd)
        reconstructed_embd_two = self.decoders[1](input_embd)

        return reconstructed_embd_one, reconstructed_embd_two

    def train_step(
        self,
        batch: Dict[Modality, Tensor],
        labels: Tensor,
        cmam_criterion: CMAMLoss,
        optimizer: Optimizer,
        device: torch.device,
        trained_model: Module,
    ):
        self.train()
        target_one = batch[self.target_modality_one].float().to(device)
        target_two = batch[self.target_modality_two].float().to(device)
        input_modalities = batch[self.input_modality].float().to(device)
        mi_input_modalities = input_modalities.clone()

        labels = labels.to(device)

        # Get the target embedding without computing gradients
        with torch.no_grad():
            trained_model.eval()
            trained_encoder = trained_model.get_encoder(self.target_modality_one)
            target_embd_one = trained_encoder(target_one)
            trained_encoder = trained_model.get_encoder(self.target_modality_two)
            target_embd_two = trained_encoder(target_two)

        # Ensure trained_model's parameters do not require gradients
        for param in trained_model.parameters():
            param.requires_grad = False

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through CMAM
        rec_embd_one, rec_embd_two = self.forward(input_modalities)

        # prepare input for the pretrained model
        encoder_data = {str(self.input_modality)[0]: batch[self.input_modality].to(device=device)}

        m_kwargs = {
            **encoder_data,
            f"{str(self.target_modality_one)[0]}": rec_embd_one.to(device=device),
            f"{str(self.target_modality_two)[0]}": rec_embd_two.to(device=device),
            f"is_embd_{str(self.target_modality_one)[0]}": True,
            f"is_embd_{str(self.target_modality_two)[0]}": True,
        }

        # Compute logits without torch.no_grad()
        logits = trained_model(**m_kwargs, device=device)
        predictions = logits.argmax(dim=1)

        if self.binarize:
            (
                binary_preds,
                binary_truth,
                non_zeros_mask,
            ) = UttFusionModel.msa_binarize(predictions.cpu().numpy(), labels)

            # Calculate metrics for all elements (including zeros)
            binary_metrics = self.metric_recorder.calculate_metrics(predictions=binary_preds, targets=binary_truth)

            # Calculate metrics for non-zero elements only using the non_zeros_mask
            non_zeros_binary_preds = binary_preds[non_zeros_mask.detach().cpu().numpy()]
            non_zeros_binary_truth = binary_truth[non_zeros_mask.detach().cpu().numpy()]

            non_zero_metrics = self.metric_recorder.calculate_metrics(
                predictions=non_zeros_binary_preds,
                targets=non_zeros_binary_truth,
            )

            # Store the metrics in a dictionary
            metrics = {}

            for k, v in binary_metrics.items():
                metrics[f"HasZero_{k}"] = v

            for k, v in non_zero_metrics.items():
                metrics[f"NonZero_{k}"] = v
        else:
            metrics = self.metric_recorder.calculate_metrics(predictions, labels)

        rec_one_loss_dict = cmam_criterion(
            predictions=rec_embd_one,
            targets=target_embd_one,
            originals=mi_input_modalities,
            reconstructed=rec_embd_one,
            forward_func=None,
            cls_logits=logits,
            cls_labels=labels,
        )

        rec_two_loss_dict = cmam_criterion(
            predictions=rec_embd_two,
            targets=target_embd_two,
            originals=mi_input_modalities,
            reconstructed=rec_embd_two,
            forward_func=None,
            cls_logits=logits,
            cls_labels=labels,
        )

        total_loss = rec_one_loss_dict["total_loss"] + rec_two_loss_dict["total_loss"]

        total_loss.backward()

        # Optional gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        optimizer.step()

        rec_one_other_losses = {k: v.item() for k, v in rec_one_loss_dict.items() if k != "total_loss"}

        rec_two_other_losses = {k: v.item() for k, v in rec_two_loss_dict.items() if k != "total_loss"}

        other_losses = {f"rec_{k}_one": v for k, v in rec_one_other_losses.items()}

        other_losses.update({f"rec_{k}_two": v for k, v in rec_two_other_losses.items()})

        return {
            "loss": total_loss.item(),
            **other_losses,
            **metrics,
        }

    def evaluate(
        self,
        batch,
        labels,
        cmam_criterion: CMAMLoss,
        device,
        trained_model,
        return_eval_data=False,
    ):
        self.eval()
        trained_model.eval()
        with torch.no_grad():
            target_one = batch[self.target_modality_one].float().to(device)
            target_two = batch[self.target_modality_two].float().to(device)
            input_modalities = batch[self.input_modality].float().to(device)
            mi_input_modalities = input_modalities.clone()
            miss_type = batch["miss_type"]

            labels = labels.to(device)

            trained_encoder = trained_model.get_encoder(self.target_modality_one)
            target_embd_one = trained_encoder(target_one)
            trained_encoder = trained_model.get_encoder(self.target_modality_two)
            target_embd_two = trained_encoder(target_two)

            rec_embd_one, rec_embd_two = self.forward(input_modalities)

            # prepare input for the pretrained model
            encoder_data = {str(self.input_modality)[0]: batch[self.input_modality].to(device=device)}

            m_kwargs = {
                **encoder_data,
                f"{str(self.target_modality_one)[0]}": rec_embd_one.to(device=device),
                f"{str(self.target_modality_two)[0]}": rec_embd_two.to(device=device),
                f"is_embd_{str(self.target_modality_one)[0]}": True,
                f"is_embd_{str(self.target_modality_two)[0]}": True,
            }

            logits = trained_model(**m_kwargs, device=device)
            predictions = logits.argmax(dim=1)

            if self.binarize:
                (
                    binary_preds,
                    binary_truth,
                    non_zeros_mask,
                ) = UttFusionModel.msa_binarize(predictions.cpu().numpy(), labels)

                miss_types = np.array(miss_type)
                metrics = {}
                for miss_type in set(miss_types):
                    mask = miss_types == miss_type

                    # Apply the mask to get values for the current miss type
                    ## should be unnecessary now
                    # binary_truth = de_device(binary_truth)
                    # binary_preds = de_device(binary_preds)
                    # non_zeros_mask = de_device(non_zeros_mask)

                    binary_preds_masked = binary_preds[mask]
                    binary_truth_masked = binary_truth[mask]
                    non_zeros_mask_masked = non_zeros_mask[mask]

                    # Calculate metrics for all elements (including zeros)
                    binary_metrics = self.metric_recorder.calculate_metrics(
                        predictions=binary_preds_masked, targets=binary_truth_masked
                    )

                    # Calculate metrics for non-zero elements only using the non_zeros_mask
                    non_zeros_binary_preds_masked = binary_preds_masked[non_zeros_mask_masked]
                    non_zeros_binary_truth_masked = binary_truth_masked[non_zeros_mask_masked]

                    non_zero_metrics = self.metric_recorder.calculate_metrics(
                        predictions=non_zeros_binary_preds_masked,
                        targets=non_zeros_binary_truth_masked,
                    )

                    # Store metrics in the metrics dictionary
                    for k, v in binary_metrics.items():
                        metrics[f"HasZero_{k}_{miss_type.replace('z', '').upper()}"] = v

                    for k, v in non_zero_metrics.items():
                        metrics[f"NonZero_{k}_{miss_type.replace('z', '').upper()}"] = v
            else:
                metrics = self.metric_recorder.calculate_metrics(predictions, labels)
                miss_type = np.array(miss_type)

                for m_type in set(miss_type):
                    mask = miss_type == m_type
                    mask_preds = predictions[mask]
                    mask_labels = labels[mask]
                    mask_metrics = self.metric_recorder.calculate_metrics(
                        predictions=mask_preds,
                        targets=mask_labels,
                        skip_metrics=["ConfusionMatrix"],
                    )
                    for k, v in mask_metrics.items():
                        metrics[f"{k}_{m_type.replace('z', '').upper()}"] = v

            rec_one_loss_dict = cmam_criterion(
                predictions=rec_embd_one,
                targets=target_embd_one,
                originals=mi_input_modalities,
                reconstructed=rec_embd_one,
                forward_func=None,
                cls_logits=logits,
                cls_labels=labels,
            )

            rec_two_loss_dict = cmam_criterion(
                predictions=rec_embd_two,
                targets=target_embd_two,
                originals=mi_input_modalities,
                reconstructed=rec_embd_two,
                forward_func=None,
                cls_logits=logits,
                cls_labels=labels,
            )

            total_loss = rec_one_loss_dict["total_loss"] + rec_two_loss_dict["total_loss"]

            rec_one_other_losses = {k: v.item() for k, v in rec_one_loss_dict.items() if k != "total_loss"}
            rec_two_other_losses = {k: v.item() for k, v in rec_two_loss_dict.items() if k != "total_loss"}

            ## merge the two loss dicts and give them a prefix
            other_losses = {f"rec_{k}_one": v for k, v in rec_one_other_losses.items()}
            other_losses.update({f"rec_{k}_two": v for k, v in rec_two_other_losses.items()})

            if return_eval_data:
                return {
                    "loss": total_loss.item(),
                    **other_losses,
                    "predictions": predictions,
                    "labels": labels,
                    "rec_embd_one": rec_embd_one,
                    "rec_embd_two": rec_embd_two,
                    "target_embd_one": target_embd_one,
                    "target_embd_two": target_embd_two,
                    **metrics,
                }

            return {
                "loss": total_loss.item(),
                **other_losses,
                **metrics,
            }
