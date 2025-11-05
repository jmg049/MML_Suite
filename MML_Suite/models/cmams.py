from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, OrderedDict
from torch import Tensor
import numpy as np
import torch
from models import L2Normalization
from experiment_utils.utils import safe_detach
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

        self.do_normalisation = False
        self.norm_layer = L2Normalization() if self.do_normalisation else None

    def load_encoder_state_for(self, encoders_state: Dict[Modality, dict]) -> None:
        for modality, state in encoders_state.items():
            self.encoders[str(modality)].load_state_dict(state)
            console.print(f"Loaded state for {modality}")

    def display(self) -> str:
        assoc_params = sum(p.numel() for p in self.association_network.parameters())
        assoc_params_size_mb = assoc_params * 4 / 1024 / 1024
        return f"CMAM Model: \n\tAssociation Network Parameters: {assoc_params} ({assoc_params_size_mb:.2f} MB)"

    @property
    def parameters_size_bytes(self) -> int:
        """Returns the size of the C-MAM model's parameters in bytes."""
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def to(self, device):
        super().to(device)
        self.association_network.to(device)
        return self

    def forward(self, fused: Tensor) -> Tensor:
        if self.norm_layer is not None:
            fused = self.norm_layer(fused)
        return self.association_network(fused)

    def train_incongruent_step(
            self, batch: dict[Modality, Tensor],
            loss_functions: LossFunctionGroup,
            optimizer: Optimizer,
            device: torch.device,
            model: MultimodalModelProtocol,
            metric_recorder: MetricRecorder,
            model_optimizer: Optimizer,
            model_loss_functions: LossFunctionGroup,
            *,
            epoch: int = 0,
    ):
        """
        Joint training of C-MAM and base model for incongruent federated learning.
        
        The C-MAM reconstructs missing modalities and the base model uses both 
        available and reconstructed modalities for classification training.
        Both models are optimized simultaneously with separate optimizers.
        """
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

        # Set models to training mode
        self.train()
        self.to(device)
        model.train()
        model.to(device)

        # Zero gradients for both optimizers
        optimizer.zero_grad()
        model_optimizer.zero_grad()

        # Get available modalities and target modality data
        input_modalities = {modality: batch[modality].float().to(device) for modality in self.input_modalities}
        target_modality_data = batch[self.target_modality].float().to(device)
        
        other_modalities = {
            k: v for k, v in batch.items()
            if isinstance(k, Modality) and k != self.target_modality and k not in self.input_modalities
        }

        # === C-MAM Forward Pass ===
        # Get embeddings from available modalities using base model encoders
        with torch.no_grad():
            # Get ground truth embedding for C-MAM reconstruction loss
            target_embd_gt = model.get_encoder(self.target_modality)(target_modality_data)
        
        # Get input embeddings for C-MAM (these require gradients for base model training)
        embd_input_modalities = {m: model.get_encoder(m)(input_modalities[m]) for m in self.input_modalities}
        
        # Reconstruct missing modality embedding with C-MAM
        fused = self.fusion_fn([embd_input_modalities[m] for m in embd_input_modalities], dim=1)
        rec_embd = self.forward(fused)

        # === Base Model Forward Pass ===
        # Prepare input for base model using available + reconstructed modalities
        m_kwargs = {
            str(k)[0].upper(): embd_input_modalities[k].to(device=device)
            for k in self.input_modalities
        }
        m_kwargs[f"{str(self.target_modality)[0]}"] = rec_embd.to(device=device)
        
        # Zero out other modalities (not available to this client)
        m_kwargs.update({
            str(k)[0].upper(): torch.zeros_like(v.to(device=device)) 
            for k, v in other_modalities.items()
        })
        
        # Mark available and reconstructed modalities as embeddings
        m_kwargs.update({
            f"is_embd_{str(k)[0].upper()}": True for k in self.input_modalities
        })
        m_kwargs[f"is_embd_{str(self.target_modality)[0].upper()}"] = True
        
        # Add raw data for other modalities (zeros)
        m_kwargs.update({
            f"{str(k)[0]}": v.to(device=device) for k, v in other_modalities.items()
        })

        # Get classification logits from base model
        logits = model(**m_kwargs)

        # === Loss Computation ===
        # Base model classification loss
        classification_loss_dict = model_loss_functions(logits, labels)
        classification_loss = classification_loss_dict["total_loss"]
        
        # # C-MAM reconstruction loss
        # reconstruction_loss_dict = loss_functions(
        #     inputs=rec_embd,
        #     targets=target_embd_gt,
        #     originals=[input_modalities[m] for m in self.input_modalities],
        #     reconstructed=rec_embd,
        #     forward_func=None,
        #     cls_logits=logits,
        #     cls_labels=labels,
        # )
        # reconstruction_loss = reconstruction_loss_dict["total_loss"]
        
        # Combined loss for joint training
        total_loss = classification_loss

        # === Backward Pass ===
        total_loss.backward()

        # Optional gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

        # Step both optimizers
        optimizer.step()
        model_optimizer.step()

        # === Metrics ===
        logits_transform = (
            model.logits_transform if hasattr(model, "logits_transform") else lambda x: x.argmax(dim=1)
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
        # metric_recorder.update_group_all("reconstruction", rec_embd, target_embd_gt, miss_type)

        # Combine losses for output
        other_losses = {}
        for k, v in classification_loss_dict.items():
            if k != "total_loss":
                other_losses[f"classification_{k}"] = v.item()
        # for k, v in reconstruction_loss_dict.items():
        #     if k != "total_loss":
        #         other_losses[f"reconstruction_{k}"] = v.item()
        
        other_losses["classification_loss"] = classification_loss.item()
        # other_losses["reconstruction_loss"] = reconstruction_loss.item()

        return {
            "loss": total_loss.item(),
            "other_losses": other_losses,
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

        # Compute reconstruction loss

        # In theory, these are already zero. But this guarantees that any modality that is not part of the input is set to zero.

        other_modalities = {str(k)[0].upper(): torch.zeros_like(v.to(device=device)) for k, v in other_modalities.items()}

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
            if isinstance(logits, tuple):
                logits = logits[0]

        logits_transform = (
            trained_model.logits_transform if hasattr(trained_model, "logits_transform") else lambda x: x.argmax(dim=1)
        )

        predictions = logits_transform(logits)

        if predictions.ndim == 2:
            predictions = predictions.sqeueeze(-1)
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
            sample_ids = batch["sample_idx"]
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

            try:
                miss_type = batch["pattern_names"]
            except KeyError as _:
                try:
                    miss_type = batch["pattern_name"]
                except KeyError as e:
                    raise e

            # Get the target embedding without computing gradients
            trained_model.to(device)
            trained_model.eval()
            trained_encoder = trained_model.get_encoder(self.target_modality)
            target_embd = trained_encoder(target_modality.to(device))
            input_embds = OrderedDict(
                {m: trained_model.get_encoder(m)(input_modalities[m]) for m in self.input_modalities}
            )

            # Prepare input for the pretrained model
            encoder_data = {str(k)[0].upper(): input_embds[k].to(device=device) for k in self.input_modalities}

            is_embd = {f"is_embd_{str(i)[0].upper()}": True for i in self.input_modalities}
            is_embd[f"is_embd_{str(self.target_modality)[0].upper()}"] = True

            other_modalities = {str(k)[0].upper(): torch.zeros_like(v.to(device=device)) for k, v in other_modalities.items()}

            m_kwargs = {
                **encoder_data,
                **is_embd,
                f"{str(self.target_modality)[0]}": torch.zeros_like(target_embd).to(device=device),
                **other_modalities,
            }
            m_kwargs = {str(k): v for k, v in m_kwargs.items()}
            gt_logits = trained_model.forward(**m_kwargs)

            if isinstance(gt_logits, tuple):
                other = gt_logits[1:]
                gt_logits = gt_logits[0]


            fused = self.fusion_fn([input_embds[m] for m in input_embds], dim=1)

            # Forward pass through CMAM
            rec_embd = self.forward(fused)

            is_embd = {f"is_embd_{str(i)[0].upper()}": True for i in self.input_modalities}
            is_embd[f"is_embd_{str(self.target_modality)[0].upper()}"] = True
            m_kwargs = {
                **encoder_data,
                **is_embd,
                f"{str(self.target_modality)[0]}": rec_embd.to(device=device),
                **other_modalities,
            }

            logits = trained_model(**m_kwargs)

            if isinstance(logits, tuple):
                logits = logits[0]


            # get input modalities only
            other_logits = {}

            for modality in self.input_modalities:
                # Get the trained model input embedding
                input_embd = input_embds[modality]
                # Zero out the other modalities and the target modality
                _other_modalities = {k: torch.zeros_like(v) for k, v in other_modalities.items()}
                is_embd = {f"is_embd_{str(i)[0].upper()}": True for i in self.input_modalities}
                is_embd[f"is_embd_{str(self.target_modality)[0].upper()}"] = True

                target = torch.zeros_like(target_embd)

                # get all the the other input modalities
                encoder_data = {
                    str(k)[0].upper(): torch.zeros_like(input_embds[k]).to(device=device)
                    for k in self.input_modalities
                    if k != modality
                }

                encoder_data[str(modality)[0].upper()] = input_embd.to(device=device)

                m_kwargs = {
                    **encoder_data,
                    **is_embd,
                    f"{str(self.target_modality)[0]}": target.to(device=device),
                    **_other_modalities,
                }

                m_logits = trained_model(**m_kwargs)

                if isinstance(m_logits, tuple):
                    m_logits = m_logits[0]


                other_logits[f"{modality}"] = safe_detach(m_logits)

            for modality in other_modalities:
                # Get the trained model input embedding
                input_embd = torch.zeros_like(other_modalities[modality])
                # Zero out the other modalities and the target modality
                _other_modalities = {k: torch.zeros_like(v) for k, v in other_modalities.items() if k != modality}
                is_embd = {f"is_embd_{str(i)[0].upper()}": True for i in self.input_modalities}
                is_embd[f"is_embd_{str(self.target_modality)[0].upper()}"] = True

                target = torch.zeros_like(target_embd)

                # get all the the other input modalities
                encoder_data = {
                    str(k)[0].upper(): torch.zeros_like(input_embds[k]).to(device=device)
                    for k in self.input_modalities
                    if k != modality
                }

                encoder_data[str(modality)[0].upper()] = input_embd.to(device=device)

                m_kwargs = {
                    **encoder_data,
                    **is_embd,
                    f"{str(self.target_modality)[0]}": target.to(device=device),
                    **_other_modalities,
                }

                m_logits = trained_model(**m_kwargs)
                if isinstance(m_logits, tuple):
                    m_logits = m_logits[0]
                other_logits[f"{modality}"] = safe_detach(m_logits)

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

            rec_logits = safe_detach(logits)
            # Compute reconstruction loss
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

            return {
                "loss": total_loss.item(),
                "losses": other_losses,
                "predictions": safe_detach(predictions),
                "labels": labels,
                "targets": np.array(safe_detach(labels)),
                "rec_embd": rec_embd,
                "target_embd": target_embd,
                "logits": safe_detach(gt_logits),
                "rec_logits": safe_detach(rec_logits),
                "modality_logits": other_logits,
                "miss_type": miss_type,
                "sample_ids": sample_ids,
                "preds": safe_detach(predictions),
                "ground_truth_logits": safe_detach(gt_logits),
                "ground_truth_embeddings": safe_detach(target_embd),
                "reconstructed_embeddings": safe_detach(rec_embd),
                "reconstructed_logits": safe_detach(rec_logits),
            }

    def get_embeddings(
        self,
        dataloader: DataLoader,
        trained_model,
        device: torch.device,
        out_fp: str
    ) -> Dict[Modality, np.ndarray]:
        console.print("Getting embeddings...")
        self.to(device)
        self.eval()
        trained_model.to(device)

        batch_data = {
            self.labels_key: [],        # ground-truth class labels
            "obs_embd": [],             
            "target_embd": [],          
            "rec_embd": [],
        }
        for batch in dataloader:

            with torch.no_grad():
                # ------------------------------------------------------------------
                # 1. Collect the raw tensors
                # ------------------------------------------------------------------
                target_modality = batch[self.target_modality].float().to(device)
                input_modalities = {
                    m: batch[m].float().to(device) for m in self.input_modalities
                }

                # ------------------------------------------------------------------
                # 2. Encode the inputs - only need to encode the input modalities and not the "other" modalities
                # Why?  We only pass them through the encoders to get the embeddings - no actual prediction is done here so others never get used. 

                input_embds = OrderedDict(
                    (m, trained_model.get_encoder(m)(input_modalities[m]))
                    for m in self.input_modalities
                )

                ## Output embeddings of a zero tensor for the target modality
                z_obs = trained_model.get_encoder(self.target_modality)(torch.zeros_like(target_modality).to(device))

                # ------------------------------------------------------------------
                # 3. Encode the *ground-truth* embedding of the missing modality
                # ------------------------------------------------------------------
                # Get the target embedding when the target modality is not missing
                z_gt = trained_model.get_encoder(self.target_modality)(target_modality)

                # ------------------------------------------------------------------
                # 4. Reconstruct the missing modality with C-MAM
                # ------------------------------------------------------------------
                fused = self.fusion_fn(list(input_embds.values()), dim=1)
                z_hat = self.forward(fused)
                # ------------------------------------------------------------------
                # 5. Stash everything on CPU as float32
                # ------------------------------------------------------------------
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
        console.print(f"Encoders: {self.encoders}")
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
                "predictions": predictions,
                "labels": labels,
                "rec_embd": rec_embd,
                "target_embd": target_embd,
                
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
