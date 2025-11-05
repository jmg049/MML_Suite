from collections import OrderedDict, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from modalities import Modality
from torch.nn import CrossEntropyLoss, Linear, MSELoss, ReLU, Sequential, ModuleDict, Module
from torch import softmax

from models.avmnist import MNISTAudio, MNISTImage


class AVMNISTCoreNet(Module):
    """
    CoReNet - Compatible Representations Network
    """

    audio = str(Modality.AUDIO)[0].upper()
    image = str(Modality.IMAGE)[0].upper()

    def __init__(self, audio_encoder: MNISTAudio, image_encoder: MNISTImage, fusion_type="concat", hidden_dim=128, **kwargs):
        super().__init__()
        self.encoders = ModuleDict(OrderedDict([
            (AVMNISTCoreNet.audio, audio_encoder),
            (AVMNISTCoreNet.image, image_encoder),
        ]))
       

        # Shared projection heads
        self.inv_heads = ModuleDict(
            {
                AVMNISTCoreNet.audio: Linear(audio_encoder.get_embedding_size(), hidden_dim),
                AVMNISTCoreNet.image: Linear(image_encoder.get_embedding_size(), hidden_dim),
            }
        )
        self.spec_heads = ModuleDict(
            {
                AVMNISTCoreNet.audio: Linear(audio_encoder.get_embedding_size(), hidden_dim),
                AVMNISTCoreNet.image: Linear(image_encoder.get_embedding_size(), hidden_dim),
            }
        )

        # Fusion strategy (e.g., concat, attention, etc.)
        self.fusion_type = fusion_type
        self.classifier = Linear(hidden_dim * 2, 10)  # For digit classification

        # Auxiliary C-MAM modules: reconstruct INV from SPEC
        self.reconstructors = ModuleDict(
            {
                "audio_from_image": Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)),
                "image_from_audio": Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)),
            }
        )

    def forward(self, x_audio, x_image, return_latents=False):
        # Encode both modalities
        h_audio = self.encoders[AVMNISTCoreNet.audio](x_audio)
        h_image = self.encoders[AVMNISTCoreNet.image](x_image)

        # Project to INV and SPEC
        z_audio_inv = self.inv_heads[AVMNISTCoreNet.audio](h_audio)
        z_audio_spec = self.spec_heads[AVMNISTCoreNet.audio](h_audio)
        z_image_inv = self.inv_heads[AVMNISTCoreNet.image](h_image)
        z_image_spec = self.spec_heads[AVMNISTCoreNet.image](h_image)

        # Main task prediction
        if self.fusion_type == "concat":
            fused = torch.cat([z_audio_inv, z_image_inv], dim=-1)
        logits = self.classifier(fused)

        # Reconstructions
        rec_audio_inv = self.reconstructors["audio_from_image"](z_image_spec)
        rec_image_inv = self.reconstructors["image_from_audio"](z_audio_spec)

        if return_latents:
            return logits, {
                "z_audio_inv": z_audio_inv,
                "z_audio_spec": z_audio_spec,
                "z_image_inv": z_image_inv,
                "z_image_spec": z_image_spec,
                "rec_audio_inv": rec_audio_inv,
                "rec_image_inv": rec_image_inv,
            }
        return logits

    def train_step(
        self,
        batch,
        optimizer,
        loss_functions,
        device,
        metric_recorder,
        **kwargs,
    ):
        """
        CoReNet-specific training step.
        """
        A, I, labels, miss_type = (
            batch[Modality.AUDIO].to(device).float(),
            batch[Modality.IMAGE].to(device).float(),
            batch["labels"].to(device),
            batch["pattern_name"],
        )

        self.train()
        optimizer.zero_grad()

        # Forward pass with latent extraction
        logits, latents = self.forward(x_audio=A, x_image=I, return_latents=True)

        # Classification loss
        ce_loss = CrossEntropyLoss()(logits, labels)

        # Reconstruction losses
        rec_loss_fn = MSELoss()
        rec_audio_loss = rec_loss_fn(latents["rec_audio_inv"], latents["z_audio_inv"].detach())
        rec_image_loss = rec_loss_fn(latents["rec_image_inv"], latents["z_image_inv"].detach())
        rec_loss = rec_audio_loss + rec_image_loss

        # Compatibility loss (KL divergence between main and reconstructed predictions)
        with torch.no_grad():
            logits_rec_audio = self.classifier(torch.cat([latents["rec_audio_inv"], latents["z_image_inv"]], dim=-1))
            logits_rec_image = self.classifier(torch.cat([latents["z_audio_inv"], latents["rec_image_inv"]], dim=-1))
        kl_loss = F.kl_div(
            F.log_softmax(logits_rec_audio, dim=-1), F.softmax(logits, dim=-1), reduction="batchmean"
        ) + F.kl_div(F.log_softmax(logits_rec_image, dim=-1), F.softmax(logits, dim=-1), reduction="batchmean")

        total_loss = ce_loss + 0.5 * rec_loss + 0.1 * kl_loss
        total_loss.backward()
        optimizer.step()

        predictions = softmax(logits, dim=1).argmax(dim=1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        miss_type = np.array(miss_type)

        metric_recorder.update_group_all("classification", predictions=predictions, targets=labels, m_types=miss_type)
        metric_recorder.update_group_all("reconstruction_audio", latents["rec_audio_inv"], latents["z_audio_inv"], m_types=miss_type)
        metric_recorder.update_group_all("reconstruction_image", latents["rec_image_inv"], latents["z_image_inv"], m_types=miss_type)

        return {"loss": total_loss.item()}

    def validation_step(
        self,
        batch,
        loss_functions,
        device,
        metric_recorder,
        return_test_info=False,
        **kwargs,
    ):
        """
        CoReNet-specific validation step.
        """
        self.eval()
        with torch.no_grad():
            A, I, labels, miss_type, sample_ids = (
                batch[Modality.AUDIO].to(device).float(),
                batch[Modality.IMAGE].to(device).float(),
                batch["labels"].to(device),
                batch["pattern_name"],
                batch["sample_idx"],
            )

            logits, latents = self.forward(x_audio=A, x_image=I, return_latents=True)

            ce_loss = CrossEntropyLoss()(logits, labels)
            rec_audio_loss = F.mse_loss(latents["rec_audio_inv"], latents["z_audio_inv"])
            rec_image_loss = F.mse_loss(latents["rec_image_inv"], latents["z_image_inv"])
            rec_loss = rec_audio_loss + rec_image_loss

            logits_rec_audio = self.classifier(torch.cat([latents["rec_audio_inv"], latents["z_image_inv"]], dim=-1))
            logits_rec_image = self.classifier(torch.cat([latents["z_audio_inv"], latents["rec_image_inv"]], dim=-1))
            kl_loss = F.kl_div(
                F.log_softmax(logits_rec_audio, dim=-1), F.softmax(logits, dim=-1), reduction="batchmean"
            ) + F.kl_div(F.log_softmax(logits_rec_image, dim=-1), F.softmax(logits, dim=-1), reduction="batchmean")

            total_loss = ce_loss + 0.5 * rec_loss + 0.1 * kl_loss

            predictions = softmax(logits, dim=1).argmax(dim=1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            miss_type = np.array(miss_type)

            metric_recorder.update_group_all(
                group_name="classification", predictions=predictions, targets=labels_np, m_types=miss_type
            )

            metric_recorder.update_group_all("reconstruction_audio", latents["rec_audio_inv"], latents["z_audio_inv"], m_types=miss_type)
            metric_recorder.update_group_all("reconstruction_image", latents["rec_image_inv"], latents["z_image_inv"], m_types=miss_type)


            result = {
                "loss": total_loss.item(),
                "logits": logits.detach().cpu(),
                "predictions": predictions,
                "preds": predictions,
                "targets": labels_np,
                "miss_type": miss_type,
                "sample_ids": sample_ids,
                "ground_truth_logits": logits.detach().cpu(),
                "ground_truth_embeddings": {
                    AVMNISTCoreNet.audio: latents["z_audio_inv"].detach().cpu(),
                    AVMNISTCoreNet.image: latents["z_image_inv"].detach().cpu(),
                },
                "observed_embeddings": torch.cat([latents["z_audio_inv"], latents["z_image_inv"]], dim=-1)
                .detach()
                .cpu(),
            }

            return result

    def get_embeddings(self,dataloader,
        trained_model,
        device: torch.device,
        out_fp: str) -> dict[str, np.ndarray]:
        """
        Extracts invariant, specific, and reconstructed embeddings from CoReNet.

        Args:
            model (AVMNISTCoreNet): The CoReNet model.
            dataloader (DataLoader): Dataloader for evaluation.
            device (torch.device): Device for computation.

        Returns:
            Dict[str, np.ndarray]: Dictionary of extracted embeddings.
        """
        trained_model.to(device)
        trained_model.eval()
        embeddings = defaultdict(list)

        for batch in dataloader:
            with torch.no_grad():
                A = batch[Modality.AUDIO].to(device).float()
                I = batch[Modality.IMAGE].to(device).float()
                labels = batch["labels"]

                # Forward pass with latents
                _, latents = trained_model(A, I, return_latents=True)

                # Save latents
                for key, value in latents.items():
                    embeddings[key].append(value.detach().cpu().numpy())
                embeddings["label"].append(labels.numpy())

        # Concatenate batches
        embeddings = {k: np.concatenate(v, axis=0) for k, v in embeddings.items()}

        # Save embeddings to a npz file
        np.savez(out_fp, **embeddings)
        return embeddings