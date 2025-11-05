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
from models.msa.networks.autoencoder import ResidualAE
from models.msa.networks.classifier import FcClassifier
from models.msa.networks.lstm import LSTMEncoder
from models.msa.networks.textcnn import TextCNN
from models.msa.utt_fusion import UttFusionModel
from models.protocols import MultimodalModelProtocol
from torch.nn import Module
from torch.optim import Optimizer

console = get_console()


class MMINModel(Module, MultimodalMonitoringMixin, MultimodalModelProtocol):
    """
    Multimodal Imagination Network (MMIN) for multimodal sentiment analysis.

    This model uses autoencoders to reconstruct missing modalities and learns
    from a pretrained UttFusion teacher model through knowledge distillation.
    """

    def __init__(
        self,
        netA: LSTMEncoder,
        netV: LSTMEncoder,
        netT: TextCNN,
        netC: FcClassifier,
        netAE: ResidualAE,
        netAE_cycle: Optional[ResidualAE] = None,
        *,
        clip: Optional[float] = 1.0,
        teacher_fusion_model: UttFusionModel,
        ce_weight: float = 1.0,
        mse_weight: float = 1.0,
        cycle_weight: float = 1.0,
        share_ae_weights: bool = False,
    ) -> None:
        """
        Initialize the MMIN Model.

        Args:
            netA (LSTMEncoder): LSTM encoder for audio modality.
            netV (LSTMEncoder): LSTM encoder for video modality.
            netT (TextCNN): TextCNN encoder for text modality.
            netC (FcClassifier): Fully connected classifier for fused features.
            netAE (ResidualAE): Autoencoder for reconstruction.
            netAE_cycle (Optional[ResidualAE]): Cycle autoencoder (if not sharing weights).
            clip (Optional[float]): Gradient clipping value.
            pretrained_path (Optional[str]): Path to pretrained MMIN weights.
            teacher_fusion_model (Optional[Module]): Pretrained UttFusion teacher model.
            ce_weight (float): Weight for cross-entropy loss.
            mse_weight (float): Weight for MSE reconstruction loss.
            cycle_weight (float): Weight for cycle consistency loss.
            share_ae_weights (bool): Whether to share autoencoder weights.
        """
        super().__init__()
        self.netA = netA
        self.netV = netV
        self.netT = netT
        self.netC = netC
        self.netAE = netAE
        self.netAE_cycle = netAE_cycle if netAE_cycle is not None else netAE

        self.clip = clip
        self.ce_weight = ce_weight
        self.mse_weight = mse_weight
        self.cycle_weight = cycle_weight
        self.share_ae_weights = share_ae_weights

        # Store teacher model (pretrained UttFusion)
        self.teacher_fusion_model = teacher_fusion_model
        if self.teacher_fusion_model is not None:
            assert hasattr(
                self.teacher_fusion_model, "pretrained_path"
            ), "Teacher model must have a pretrained_path attribute"
            self.teacher_fusion_model.eval()
            # Freeze teacher model parameters
            for param in self.teacher_fusion_model.parameters():
                param.requires_grad = False

    def initialize_from_teacher(self) -> None:
        """
        Initialize student encoders with weights from the pretrained UttFusion teacher model.
        """
        if self.teacher_fusion_model is None:
            console.print("[bold red]WARNING: No teacher model provided for initialization.[/]")
            return

        console.print("Initializing student encoders from teacher UttFusion model...")

        # Copy encoder weights from teacher to student
        self.netA.load_state_dict(self.teacher_fusion_model.netA.state_dict())
        self.netV.load_state_dict(self.teacher_fusion_model.netV.state_dict())
        self.netT.load_state_dict(self.teacher_fusion_model.netT.state_dict())

        console.print("Student encoders initialized from teacher model.")

    def load_pretrained(self) -> None:
        """Load pretrained MMIN weights."""
        if self.pretrained_path is not None:
            self.pretrained_path = format_path_with_env(self.pretrained_path)
            self.pretrained_path = self.pretrained_path.format_map(
                SafeDict(run_id=get_current_run_id(), exp_name=get_current_exp_name())
            )

            console.print(f"Loading pretrained MMIN weights from {self.pretrained_path}")
            state_dict = torch.load(self.pretrained_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict=state_dict["model_state_dict"])

    def get_encoder(self, modality: Modality | str) -> Module:
        """Get the encoder module for a specific modality."""
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
        Perform forward pass of the MMIN model.

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
        # Get utterance level representations
        a_embd = self.netA(A) if not is_embd_A and A is not None else A
        v_embd = self.netV(V) if not is_embd_V and V is not None else V
        t_embd = self.netT(T) if not is_embd_T and T is not None else T

        # Fusion of available modalities
        feat_fusion = torch.cat([embd for embd in [a_embd, v_embd, t_embd] if embd is not None], dim=-1)

        # Autoencoder reconstruction
        recon_fusion, latent = self.netAE(feat_fusion)
        # Cycle consistency
        recon_cycle, latent_cycle = self.netAE_cycle(recon_fusion)

        # Classification from latent representations
        logits = self.netC(latent)

        return logits, {"gt": feat_fusion, "recon": recon_fusion }

    def flatten_parameters(self) -> None:
        """Flatten parameters for RNN layers to optimize training performance."""
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
        # Get missing modalities (already processed by dataset)
        A_miss, V_miss, T_miss, labels, _miss_type = (
            batch[Modality.AUDIO].to(device).float(),
            batch[Modality.VIDEO].to(device).float(),
            batch[Modality.TEXT].to(device).float(),
            batch["label"].to(device),
            batch["pattern_names"],
        )

        # Get complete modalities for teacher (reverse of missing)
        A_reverse = batch[f"{Modality.AUDIO}_reverse"].to(device).float()
        V_reverse = batch[f"{Modality.VIDEO}_reverse"].to(device).float()
        T_reverse = batch[f"{Modality.TEXT}_reverse"].to(device).float()

        self.train()

        # Student forward pass with missing modalities
        feat_A_miss = self.netA(A_miss)
        feat_V_miss = self.netV(V_miss)
        feat_T_miss = self.netT(T_miss)

        # Fusion of missing modalities
        feat_fusion_miss = torch.cat([feat_A_miss, feat_V_miss, feat_T_miss], dim=-1)

        # Autoencoder reconstruction
        recon_fusion, latent = self.netAE(feat_fusion_miss)
        recon_cycle, _latent_cycle = self.netAE_cycle(recon_fusion)

        # Get predictions
        logits = self.netC(latent)

        # Teacher forward pass for knowledge distillation
        with torch.no_grad():
            T_embd_A = self.teacher_fusion_model.netA(A_reverse)
            T_embd_V = self.teacher_fusion_model.netV(V_reverse)
            T_embd_T = self.teacher_fusion_model.netT(T_reverse)
            T_embds = torch.cat([T_embd_A, T_embd_V, T_embd_T], dim=-1)

        optimizer.zero_grad()

        # Compute losses
        loss_CE = self.ce_weight * F.cross_entropy(logits.squeeze(), labels.squeeze())
        loss_mse = self.mse_weight * F.mse_loss(T_embds, recon_fusion)
        loss_cycle = self.cycle_weight * F.mse_loss(feat_fusion_miss.detach(), recon_cycle)

        total_loss = loss_CE + loss_mse + loss_cycle
        total_loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)

        optimizer.step()

        predictions = safe_detach(F.softmax(logits, dim=-1).argmax(dim=-1).squeeze())
        labels = safe_detach(labels.squeeze())

        metric_recorder.update_group_all(
            "classification", predictions=predictions, targets=labels, m_types=np.array(_miss_type)
        )

        return {
            "loss": total_loss.item(),
            "loss_CE": loss_CE.item(),
            "loss_mse": loss_mse.item(),
            "loss_cycle": loss_cycle.item(),
        }

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

        with torch.no_grad():
            try:
                A, V, T, labels, miss_type, sample_ids = (
                    batch[Modality.AUDIO].to(device).float(),
                    batch[Modality.VIDEO].to(device).float(),
                    batch[Modality.TEXT].to(device).float(),
                    batch["label"].to(device),
                    batch["pattern_names"],
                    batch["sample_idx"],
                )
            except KeyError as e:
                console.print(f"[bold red]KeyError in validation step: {e}[/]")
                console.print(f"Batch keys: {list(batch.keys())}")
                raise
            # For validation, use the available modalities as they are
            # (missing modalities handling depends on your validation setup)
            logits, embds = self.forward(A, V, T)

            miss_types = np.array(miss_type)
            loss = loss_functions(logits.squeeze(), labels.squeeze())["total_loss"]
            predictions = safe_detach(F.softmax(logits, dim=-1).argmax(dim=-1).squeeze())
            labels = safe_detach(labels.squeeze())

            metric_recorder.update_group_all(
                "classification", predictions=predictions, targets=labels, m_types=miss_types
            )

        self.train()

        result = {
            "loss": loss.item(),
            "predictions": safe_detach(predictions),
            "labels": labels,
            "miss_type": miss_types,
            "targets": labels,
            "logits": safe_detach(logits),
            "sample_ids": sample_ids,
            "gt_embds": embds["gt"],
            "recon_embds": embds["recon"],
        }

        if return_test_info:
            result.update(
                {
                    "predictions": predictions.cpu().numpy(),
                    "labels": labels.cpu().numpy(),
                    "miss_type": miss_type,
                    "sample_ids": [sample_ids],
                }
            )

        return result


    def get_embeddings(
        self,
        dataloader,
        device: torch.device,
        out_fp: str
    ) -> Dict[Modality, np.ndarray]:

        self.eval()

        tracking = {
            "a": [],
            "v": [],
            "t": [],
            "av": [],
            "at": [],
            "tv": [],
        }

        with torch.no_grad():
            for batch in dataloader:
                A, V, T, miss_types, labels = (
                    batch[Modality.AUDIO].to(device).float(),
                    batch[Modality.VIDEO].to(device).float(),
                    batch[Modality.TEXT].to(device).float(),
                    batch["pattern_names"],
                    batch["label"]
                )

                A_orig, V_orig, T_orig = (
                    batch[f"{Modality.AUDIO}_original"].to(device).float(),
                    batch[f"{Modality.VIDEO}_original"].to(device).float(),
                    batch[f"{Modality.TEXT}_original"].to(device).float(),
                )

                gt_embd_A = self.netA(A_orig)
                gt_embd_V = self.netV(V_orig) 
                gt_embd_T = self.netT(T_orig)

                embd_A = self.netA(A)
                embd_V = self.netV(V)
                embd_T = self.netT(T)
                fused_with_missing = torch.cat([embd for embd in [embd_A, embd_V, embd_T]], dim=-1)

                rec_logits, embds = self.forward(A, V, T)
                try:
                    rec_embds = embds["recon"]
                except Exception:
                    console.print("[bold red]Error: 'recon' key not found in embeddings[/]")
                    console.print(f"Available keys in embeddings: {embds}")
                    raise

                rec_A = rec_embds[:, :embd_A.shape[-1]]
                rec_V = rec_embds[:, embd_A.shape[-1]:embd_A.shape[-1] + embd_V.shape[-1]]
                rec_T = rec_embds[:, -embd_T.shape[-1]:]

                for m_type, embd_A, embd_V, embd_T, fused_with_missing, rec_logits, A, V, T, label in zip(
                    miss_types, gt_embd_A, gt_embd_V, gt_embd_T, fused_with_missing, rec_logits, rec_A, rec_V, rec_T, labels
                ):
                    if m_type == "a":
                        tracking["a"].append(
                            {
                                "gt_embd": np.stack([safe_detach(embd_V), safe_detach(embd_T)], axis=0),
                                "obs_embd": safe_detach(fused_with_missing),
                                "rec_logits": safe_detach(rec_logits),
                                "rec_embds": np.stack([safe_detach(V), safe_detach(T)], axis=0),
                                "labels": safe_detach(label),
                            }
                        )
                    elif m_type == "v":
                        tracking["v"].append(
                            {
                                "gt_embd": np.stack([safe_detach(embd_A), safe_detach(embd_T)], axis=0),
                                "obs_embd": safe_detach(fused_with_missing),
                                "rec_logits": safe_detach(rec_logits),
                                "rec_embds": np.stack([safe_detach(A), safe_detach(T)], axis=0),
                                "labels": safe_detach(label),
                            }
                        )
                    elif m_type == "t":
                        tracking["t"].append(
                            {
                                "gt_embd": np.stack([safe_detach(embd_A), safe_detach(embd_V)], axis=0),
                                "obs_embd": safe_detach(fused_with_missing),
                                "rec_logits": safe_detach(rec_logits),
                                "rec_embds": np.stack([safe_detach(A), safe_detach(V)], axis=0),
                                "labels": safe_detach(label),
                                
                            }
                        )
                    elif m_type == "av":
                        tracking["av"].append(
                            {
                                "gt_embd": safe_detach(embd_T),
                                "obs_embd": safe_detach(fused_with_missing),
                                "rec_logits": safe_detach(rec_logits),
                                "rec_embds": safe_detach(T),
                                "labels": safe_detach(label),
                            }
                        )
                    elif m_type == "at":
                        tracking["at"].append(
                            {
                                "gt_embd": embd_V,
                                "obs_embd": safe_detach(fused_with_missing),
                                "rec_logits": safe_detach(rec_logits),
                                "rec_embds": safe_detach(V),
                                "labels": safe_detach(label),
                            }
                        )
                    elif m_type == "vt" or m_type == "tv":
                        tracking["tv"].append(
                            {
                                "gt_embd": embd_A,
                                "obs_embd": safe_detach(fused_with_missing),
                                "rec_logits": safe_detach(rec_logits),
                                "rec_embds": safe_detach(A),
                                "labels": safe_detach(label),
                            }
                        )
        # Convert lists to numpy arrays
        for key in tracking:
            tracking[key] = np.array([item for item in tracking[key] if item is not None])
        # Save embeddings to file
        np.savez_compressed(
            out_fp,
            a=tracking["a"],
            v=tracking["v"],
            t=tracking["t"],
            av=tracking["av"],
            at=tracking["at"],
            vt=tracking["tv"],
        )
        console.print(f"Embeddings saved to {out_fp}")




