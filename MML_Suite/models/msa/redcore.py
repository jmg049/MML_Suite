from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.utils import safe_detach
from modalities import Modality
from models.mixins import MonitoringMixin
from models.msa.networks.autoencoder import ResidualAE, ResidualXE
from models.msa.networks.classifier import FcClassifier
from models.msa.networks.transformer import Transformer
from models.protocols import MultimodalModelProtocol
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from experiment_utils.printing import get_console
import torch.nn.functional as F
console = get_console()


class RedCore(Module, MonitoringMixin, MultimodalModelProtocol):
    feature_dim: int = 32  # Magic number
    lambda_one: float = 0.0008  # Another magic number

    def __init__(
        self,
        netA: Transformer,
        netV: Transformer,
        netT: Transformer,
        netAE: ResidualAE,
        netC: FcClassifier,
        netAT_V: ResidualXE,
        netAV_T: ResidualXE,
        netVT_A: ResidualAE,
        netC_A: FcClassifier,
        netC_V: FcClassifier,
        netC_T: FcClassifier,
        share_weight: bool = False,
        loss_beta: float = 0.95,
        interval_i: int = 2,
        eta: float = 0.001,
        eta_ext: float = 1.5,
        clip: float = 1.0,
        ce_weight: float = 1.0,
        cycle_weight: float = 1.0,
        mse_weight: float = 1.0,
    ) -> None:
        super(RedCore, self).__init__()
        self.netA = netA
        self.netA.initialize_parameters()
        self.netV = netV
        self.netV.initialize_parameters()
        self.netT = netT
        self.netT.initialize_parameters()

        self.netAE = netAE
        self.netC = netC
        self.netAT_V = netAT_V
        self.netAV_T = netAV_T
        self.netVT_A = netVT_A
        self.netC_A = netC_A
        self.netC_V = netC_V
        self.netCls_T = netC_T
        ae_input_dim = self.netA.embd_width + self.netV.embd_width + self.netT.embd_width

        if share_weight:
            self.netAE_cycle = self.netAE
        else:
            self.netAE_cycle = ResidualAE(
                self.netAE._layers, self.netAE.n_blocks, ae_input_dim, dropout=0.0, use_bn=False
            )

        self._loss_A = 0.0
        self._loss_V = 0.0
        self._loss_T = 0.0
        self._loss_beta = loss_beta
        self._beta = np.array([1.0, 1.0, 1.0])
        self._iter_count = 0
        self._interval_i = interval_i

        self._eta = eta
        self._eta_ext = eta_ext
        self.clip = clip

        self.ce_weight = ce_weight
        self.cycle_weight = cycle_weight
        self.mse_weight = mse_weight

    def forward(
        self,
        A: Tensor,
        V: Tensor,
        T: Tensor,
        A_missing_index: Tensor,
        V_missing_index: Tensor,
        T_missing_index: Tensor,
    ) -> Dict[str, Tensor]:
        feature_A_miss, fmu_A, flog_var_A = self.netA.forward(A)
        feature_V_miss, fmu_V, flog_var_V = self.netV.forward(V)
        feature_T_miss, fmu_T, flog_var_T = self.netT.forward(T)

        feature_fusion_miss = torch.cat([feature_A_miss, feature_V_miss, feature_T_miss], dim=-1)
        recon_fusion, latent = self.netAE.forward(feature_fusion_miss)
        recon_cycle, latent_cycle = self.netAE_cycle.forward(recon_fusion)

        gen_A, _latent_A = self.netVT_A.forward(torch.cat([feature_V_miss, feature_T_miss], dim=-1))
        gen_V, _latent_V = self.netAT_V.forward(torch.cat([feature_A_miss, feature_T_miss], dim=-1))
        gen_T, _latent_T = self.netAV_T.forward(torch.cat([feature_A_miss, feature_V_miss], dim=-1))

        batch_size = feature_A_miss.shape[0]
        feature_A_r = (
            A_missing_index.reshape(batch_size, 1) * feature_A_miss
            - (A_missing_index.reshape(batch_size, 1) - 1) * gen_A
        )

        feature_V_r = (
            V_missing_index.reshape(batch_size, 1) * feature_V_miss
            - (V_missing_index.reshape(batch_size, 1) - 1) * gen_V
        )

        feature_T_r = (
            T_missing_index.reshape(batch_size, 1) * feature_T_miss
            - (T_missing_index.reshape(batch_size, 1) - 1) * gen_T
        )

        feature_fusion_r = torch.cat([feature_A_r, feature_V_r, feature_T_r], dim=-1)
        logits = self.netC.forward(feature_fusion_r)

        logits_a = self.netC_A.forward(feature_A_r)
        logits_v = self.netC_V.forward(feature_V_r)
        logits_t = self.netCls_T.forward(feature_T_r)

        return {
            "logits": logits,
            "fusion": feature_fusion_miss,
            "recon_fusion": recon_fusion,
            "recon_cycle": recon_cycle,
            "latent": latent,
            "latent_cycle": latent_cycle,
            "feature_A_miss": feature_A_miss,
            "feature_V_miss": feature_V_miss,
            "feature_T_miss": feature_T_miss,
            "gen_A": gen_A,
            "logits_A": logits_a,
            "fmu_A": fmu_A,
            "flog_var_A": flog_var_A,
            "gen_V": gen_V,
            "logits_V": logits_v,
            "fmu_V": fmu_V,
            "flog_var_V": flog_var_V,
            "gen_T": gen_T,
            "logits_T": logits_t,
            "fmu_T": fmu_T,
            "flog_var_T": flog_var_T,
        }

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: Optimizer,
        loss_functions: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        **kwargs,
    ) -> Dict[str, Any]:
        try:
            A, V, T, missing_index_A, missing_index_A, missing_index_T, labels, miss_type = (
            batch[Modality.AUDIO],
            batch[Modality.VIDEO],
            batch[Modality.TEXT],
            batch[str(Modality.AUDIO) + "_missing_index"],
            batch[str(Modality.VIDEO) + "_missing_index"],
            batch[str(Modality.TEXT) + "_missing_index"],
            batch["label"],
            batch["pattern_names"],
        )
        except KeyError as e:
            console.print(f"Failed to find expected key in batch: {e}")
            console.print(f"Available keys: {list(batch.keys())}")
            raise e

        A, V, T, missing_index_A, missing_index_A, missing_index_T, labels = (
            A.float().to(device),
            V.float().to(device),
            T.float().to(device),
            missing_index_A.float().to(device),
            missing_index_A.float().to(device),
            missing_index_T.float().to(device),
            labels.to(device),
        )
        miss_type = np.array(miss_type)

        self.train()
        optimizer.zero_grad()

        forward_results = self.forward(A, V, T, missing_index_A, missing_index_A, missing_index_T)
        batch_size = missing_index_A.shape[0]

        logits = forward_results["logits"]

        index_A = missing_index_A.reshape(batch_size, 1)
        index_V = missing_index_A.reshape(batch_size, 1)
        index_T = missing_index_T.reshape(batch_size, 1)

        labels = labels.squeeze() if labels.ndim > 1 else labels
        ## Below handles the * cross_entropy_weight variable too, just inside the loss_function call
        ## Each one is equivalent to ``cross_entropy_weight * cross_entropy_loss(logits, labels)``
        loss_ce = F.cross_entropy(logits, labels) * self.ce_weight
        assert isinstance(loss_ce, Tensor), "Expected loss_ce to be a Tensor, got: {}".format(type(loss_ce))
        loss_ce_A = F.cross_entropy(forward_results["logits_A"], labels) * self.ce_weight
        assert isinstance(loss_ce_A, Tensor), "Expected loss_ce_A to be a Tensor, got: {}".format(type(loss_ce_A))
        loss_ce_V = F.cross_entropy(forward_results["logits_V"], labels) * self.ce_weight
        assert isinstance(loss_ce_V, Tensor), "Expected loss_ce_V to be a Tensor, got: {}".format(type(loss_ce_V))
        loss_ce_T = F.cross_entropy(forward_results["logits_T"], labels) * self.ce_weight
        assert isinstance(loss_ce_T, Tensor), "Expected loss_ce_T to be a Tensor, got: {}".format(type(loss_ce_T))

        fmu_A = forward_results["fmu_A"]
        flog_var_A = forward_results["flog_var_A"]

        fmu_V = forward_results["fmu_V"]
        flog_var_V = forward_results["flog_var_V"]

        fmu_T = forward_results["fmu_T"]
        flog_var_T = forward_results["flog_var_T"]

        KLD_feature_A: Tensor = (
            -1.0
            * self.lambda_one
            * torch.sum((1.0 + flog_var_A - fmu_A.pow(2) - flog_var_A.exp()) * index_A)
            / batch_size
        )

        KLD_feature_V: Tensor = (
            -1.0
            * self.lambda_one
            * torch.sum((1.0 + flog_var_V - fmu_V.pow(2) - flog_var_V.exp()) * index_V)
            / batch_size
        )

        KLD_feature_T: Tensor = (
            -1.0
            * self.lambda_one
            * torch.sum((1.0 + flog_var_T - fmu_T.pow(2) - flog_var_T.exp()) * index_T)
            / batch_size
        )

        batch_size_A = sum(missing_index_A)
        batch_size_V = sum(missing_index_A)
        batch_size_T = sum(missing_index_T)

        feature_A_miss, gen_A = forward_results["feature_A_miss"], forward_results["gen_A"]
        feature_V_miss, gen_V = forward_results["feature_V_miss"], forward_results["gen_V"]
        feature_T_miss, gen_T = forward_results["feature_T_miss"], forward_results["gen_T"]

        loss_mse_A =  F.mse_loss(gen_A * index_A, feature_A_miss * index_A) / batch_size_A 
        loss_mse_A = loss_mse_A * self.mse_weight
        
        assert isinstance(loss_mse_A, Tensor), "Expected loss_mse_A to be a Tensor, got: {}".format(type(loss_mse_A))
        
        loss_mse_V = F.mse_loss(gen_V * index_V, feature_V_miss * index_V) / batch_size_V
        loss_mse_V = loss_mse_V * self.mse_weight
        assert isinstance(loss_mse_V, Tensor), "Expected loss_mse_V to be a Tensor, got: {}".format(type(loss_mse_V))

        loss_mse_T = F.mse_loss(gen_T * index_T, feature_T_miss * index_T)/ batch_size_T
        loss_mse_T = loss_mse_T * self.mse_weight
        assert isinstance(loss_mse_T, Tensor), "Expected loss_mse_T to be a Tensor, got: {}".format(type(loss_mse_T))

        loss_update_A = loss_mse_A if loss_mse_A.item() != 0.0 else self._loss_A
        loss_update_V = loss_mse_V if loss_mse_V.item() != 0.0 else self._loss_V
        loss_update_T = loss_mse_T if loss_mse_T.item() != 0.0 else self._loss_T

        self._loss_A = (1.0 - self._loss_beta) * self._loss_A + self._loss_beta * loss_update_A
        self._loss_V = (1.0 - self._loss_beta) * self._loss_V + self._loss_beta * loss_update_V
        self._loss_T = (1.0 - self._loss_beta) * self._loss_T + self._loss_beta * loss_update_T

        # losses are not tensors at this point, but floats
        self._loss_A = torch.clone(self._loss_A) if isinstance(self._loss_A, Tensor) else torch.tensor(self._loss_A)
        self._loss_V = torch.clone(self._loss_V) if isinstance(self._loss_V, Tensor) else torch.tensor(self._loss_V)
        self._loss_T = torch.clone(self._loss_T) if isinstance(self._loss_T, Tensor) else torch.tensor(self._loss_T)

        loss_ATV = self._loss_A + self._loss_V + self._loss_T

        loss_ATV_avg = loss_ATV / 3.0
        ra = (loss_ATV_avg - loss_ATV) / loss_ATV_avg
        ra = float(ra)
        if self._iter_count % 500 == 0:
            self._eta = self._eta * self._eta_ext

        if self._iter_count % self._interval_i == 0:
            self._beta = self._beta * self._eta * ra
            self._beta[0] = max(0.1, self._beta[0])
            self._beta[1] = max(0.1, self._beta[1])
            self._beta[2] = max(0.1, self._beta[2])
            self._beta = self._beta / (sum(self._beta**2) ** (0.5))

        self._iter_count += 1

        loss_mse: Tensor = self.mse_weight * (
            self._beta[0] * loss_mse_A + self._beta[1] * loss_mse_V + self._beta[2] * loss_mse_T
        )
        loss = loss_ce + KLD_feature_A + KLD_feature_V + KLD_feature_T + loss_ce_A + loss_ce_V + loss_ce_T + loss_mse
        
        loss.backward()
        ## Clip grad norm
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        optimizer.step()

        # # print all losses
        # console.rule("Losses")
        # console.print(f"loss_ce: {loss_ce:.4f}, "
        #               f"loss_mse: {loss_mse:.4f}, "
        #               f"KLD_feature_A: {KLD_feature_A:.4f}, "
        #               f"KLD_feature_V: {KLD_feature_V:.4f}, "
        #               f"KLD_feature_T: {KLD_feature_T:.4f}, "
        #               f"loss_ce_A: {loss_ce_A:.4f}, "
        #               f"loss_ce_V: {loss_ce_V:.4f}, "
        #               f"loss_ce_T: {loss_ce_T:.4f}, "
        #               f"loss_mse_A: {loss_mse_A:.4f}, "
        #               f"loss_mse_V: {loss_mse_V:.4f}, "
        #               f"loss_mse_T: {loss_mse_T:.4f}, "
        #               f"loss: {loss:.4f}, "
        #               f"loss_ATV: {loss_ATV:.4f}, "  )
        # console.rule("-")

        predictions = logits.argmax(dim=1)
        labels = safe_detach(labels)
        predictions = safe_detach(predictions)
        metric_recorder.update_group_all(
            "classification", predictions=predictions, targets=labels, m_types=np.array(miss_type)
        )

        def safe_item(item: Tensor | float) -> float:
            """Safely convert a tensor to a float, handling both cases."""
            if isinstance(item, Tensor):
                return item.item()
            return item

        return {
            "loss": safe_item(loss),
            "losses": {
                "ce": safe_item(loss_ce),
                "mse": safe_item(loss_mse),
                "KLD_A": safe_item(KLD_feature_A),
                "KLD_V": safe_item(KLD_feature_V),
                "KLD_T": safe_item(KLD_feature_T),
                "ce_A": safe_item(loss_ce_A),
                "ce_V": safe_item(loss_ce_V),
                "ce_T": safe_item(loss_ce_T),
                "mse_A": safe_item(loss_mse_A),
                "mse_V": safe_item(loss_mse_V),
                "mse_T": safe_item(loss_mse_T),
            },
        }



    def validation_step(
        self,
        batch: Dict[str, Any],
        loss_functions: LossFunctionGroup,
        device: torch.device,
        metric_recorder: MetricRecorder,
        **kwargs,
    ) -> Dict[str, Any]:
        
        all_predictions, all_labels, all_miss_types, all_sample_ids = [], [], [], []

        with torch.no_grad():
            A, V, T, missing_index_A, missing_index_A, missing_index_T, labels, miss_type, sample_ids = (
                batch[Modality.AUDIO],
                batch[Modality.VIDEO],
                batch[Modality.TEXT],
                batch[str(Modality.AUDIO) + "_missing_index"],
                batch[str(Modality.VIDEO) + "_missing_index"],
                batch[str(Modality.TEXT) + "_missing_index"],
                batch["label"],
                batch["pattern_names"],
                batch["sample_idx"],
            )

            A, V, T, missing_index_A, missing_index_A, missing_index_T, labels = (
                A.float().to(device),
                V.float().to(device),
                T.float().to(device),
                missing_index_A.float().to(device),
                missing_index_A.float().to(device),
                missing_index_T.float().to(device),
                labels.to(device),
            )
            miss_type = np.array(miss_type)

            self.eval()

            forward_results = self.forward(A, V, T, missing_index_A, missing_index_A, missing_index_T)
            batch_size = missing_index_A.shape[0]

            logits = forward_results["logits"]

            index_A = missing_index_A.reshape(batch_size, 1)
            index_V = missing_index_A.reshape(batch_size, 1)
            index_T = missing_index_T.reshape(batch_size, 1)
            labels = labels.squeeze() if labels.ndim > 1 else labels

            ## Below handles the * cross_entropy_weight variable too, just inside the loss_function call
            ## Each one is equivalent to ``cross_entropy_weight * cross_entropy_loss(logits, labels)``
            loss_ce = F.cross_entropy(logits, labels) * self.ce_weight
            loss_ce_A = F.cross_entropy(forward_results["logits_A"], labels) * self.ce_weight
            loss_ce_V = F.cross_entropy(forward_results["logits_V"], labels) * self.ce_weight
            loss_ce_T = F.cross_entropy(forward_results["logits_T"], labels) * self.ce_weight

            fmu_A = forward_results["fmu_A"]
            flog_var_A = forward_results["flog_var_A"]

            fmu_V = forward_results["fmu_V"]
            flog_var_V = forward_results["flog_var_V"]

            fmu_T = forward_results["fmu_T"]
            flog_var_T = forward_results["flog_var_T"]

            KLD_feature_A: Tensor = (
                -1.0
                * self.lambda_one
                * torch.sum((1.0 + flog_var_A - fmu_A.pow(2) - flog_var_A.exp()) * index_A)
                / batch_size
            )

            KLD_feature_V: Tensor = (
                -1.0
                * self.lambda_one
                * torch.sum((1.0 + flog_var_V - fmu_V.pow(2) - flog_var_V.exp()) * index_V)
                / batch_size
            )

            KLD_feature_T: Tensor = (
                -1.0
                * self.lambda_one
                * torch.sum((1.0 + flog_var_T - fmu_T.pow(2) - flog_var_T.exp()) * index_T)
                / batch_size
            )

            batch_size_A = sum(missing_index_A)
            batch_size_V = sum(missing_index_A)
            batch_size_T = sum(missing_index_T)

            feature_A_miss, gen_A = forward_results["feature_A_miss"], forward_results["gen_A"]
            feature_V_miss, gen_V = forward_results["feature_V_miss"], forward_results["gen_V"]
            feature_T_miss, gen_T = forward_results["feature_T_miss"], forward_results["gen_T"]

            loss_mse_A = F.mse_loss(gen_A * index_A, feature_A_miss * index_A) / batch_size_A
            loss_mse_A = loss_mse_A * self.mse_weight
            loss_mse_V = F.mse_loss(gen_V * index_V, feature_V_miss * index_V) / batch_size_V
            loss_mse_V = loss_mse_V * self.mse_weight
            loss_mse_T = F.mse_loss(gen_T * index_T, feature_T_miss * index_T) / batch_size_T
            loss_mse_T = loss_mse_T * self.mse_weight

            loss_update_A = loss_mse_A if loss_mse_A.item() != 0.0 else self._loss_A
            loss_update_V = loss_mse_V if loss_mse_V.item() != 0.0 else self._loss_V
            loss_update_T = loss_mse_T if loss_mse_T.item() != 0.0 else self._loss_T

            self._loss_A = (1.0 - self._loss_beta) * self._loss_A + self._loss_beta * loss_update_A
            self._loss_V = (1.0 - self._loss_beta) * self._loss_V + self._loss_beta * loss_update_V
            self._loss_T = (1.0 - self._loss_beta) * self._loss_T + self._loss_beta * loss_update_T

            self._loss_A = torch.clone(self._loss_A) if isinstance(self._loss_A, Tensor) else torch.tensor(self._loss_A)
            self._loss_V = torch.clone(self._loss_V) if isinstance(self._loss_V, Tensor) else torch.tensor(self._loss_V)
            self._loss_T = torch.clone(self._loss_T) if isinstance(self._loss_T, Tensor) else torch.tensor(self._loss_T)

            loss_ATV = self._loss_A + self._loss_V + self._loss_T
            loss_ATV_avg = loss_ATV / 3.0
            ra = (loss_ATV_avg - loss_ATV) / loss_ATV_avg
            ra = float(ra)

            if self._iter_count % 500 == 0:
                self._eta = self._eta * self._eta_ext

            if self._iter_count % self._interval_i == 0:
                self._beta = self._beta * self._eta * ra
                self._beta[0] = max(0.1, self._beta[0])
                self._beta[1] = max(0.1, self._beta[1])
                self._beta[2] = max(0.1, self._beta[2])
                self._beta = self._beta / (sum(self._beta**2) ** (0.5))

            self._iter_count += 1

            loss_mse: Tensor = self.mse_weight * (
                self._beta[0] * loss_mse_A + self._beta[1] * loss_mse_V + self._beta[2] * loss_mse_T
            )
            loss = loss_ce + KLD_feature_A + KLD_feature_V + KLD_feature_T + loss_ce_A + loss_ce_V + loss_ce_T + loss_mse

            predictions = logits.argmax(dim=1)
            labels = safe_detach(labels)
            predictions = safe_detach(predictions)
            metric_recorder.update_group_all(
                "classification", predictions=predictions, targets=labels, m_types=np.array(miss_type)
            )

            def safe_item(item: Tensor | float) -> float:
                """Safely convert a tensor to a float, handling both cases."""
                if isinstance(item, Tensor):
                    return item.item()
                return item
            all_predictions.append(safe_detach(predictions))
            all_labels.append(safe_detach(labels))
            all_miss_types.append(miss_type)
            all_sample_ids.append(sample_ids)
            assert len(miss_type) == len(sample_ids), f"Miss type and sample IDs must have the same length, not {len(miss_type)} and {len(sample_ids)}"
            return {
                "loss": safe_item(loss),
                "losses": {
                    "ce": safe_item(loss_ce),
                    "mse": safe_item(loss_mse),
                    "KLD_A": safe_item(KLD_feature_A),
                    "KLD_V": safe_item(KLD_feature_V),
                    "KLD_T": safe_item(KLD_feature_T),
                    "ce_A": safe_item(loss_ce_A),
                    "ce_V": safe_item(loss_ce_V),
                    "ce_T": safe_item(loss_ce_T),
                    "mse_A": safe_item(loss_mse_A),
                    "mse_V": safe_item(loss_mse_V),
                    "mse_T": safe_item(loss_mse_T),
                },
                "predictions": safe_detach(predictions),
                "labels": safe_detach(labels),
                "miss_type": miss_type,
                "sample_ids": sample_ids,
                "logits": safe_detach(logits),
                "targets": safe_detach(labels),
            }
        
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
            for batch in tqdm(dataloader):
                A, V, T, missing_index_A, missing_index_A, missing_index_T, labels, miss_types = (
            batch[Modality.AUDIO],
            batch[Modality.VIDEO],
            batch[Modality.TEXT],
            batch[str(Modality.AUDIO) + "_missing_index"],
            batch[str(Modality.VIDEO) + "_missing_index"],
            batch[str(Modality.TEXT) + "_missing_index"],
            batch["label"],
            batch["pattern_names"],
                )

                A_orig, V_orig, T_orig = (
                    batch[f"{Modality.AUDIO}_original"].to(device).float(),
                    batch[f"{Modality.VIDEO}_original"].to(device).float(),
                    batch[f"{Modality.TEXT}_original"].to(device).float(),
                )
                A, V, T, missing_index_A, missing_index_A, missing_index_T = (
                    A.float().to(device),
                    V.float().to(device),
                    T.float().to(device),
                    missing_index_A.float().to(device),
                    missing_index_A.float().to(device),
                    missing_index_T.float().to(device),
                )

                gt_embd_A, _, _ = self.netA(A_orig)
                gt_embd_V, _, _ = self.netV(V_orig) 
                gt_embd_T, _, _ = self.netT(T_orig)

                embd_A, _, _ = self.netA.forward(A)
                embd_V, _, _ = self.netV.forward(V)
                embd_T, _, _ = self.netT.forward(T)
                fused_with_missing = torch.cat([embd_A, embd_V, embd_T], dim=-1)

                embds = self.forward(A, V, T, missing_index_A, missing_index_A, missing_index_T)
                gen_A = embds["gen_A"]
                gen_V = embds["gen_V"]
                gen_T = embds["gen_T"]
                rec_logits = embds["logits"]
                rec_embds = {
                    "A": gen_A,
                    "V": gen_V,
                    "T": gen_T,
                }

                rec_A = gen_A
                rec_V = gen_V
                rec_T = gen_T

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
                                "labels": label,
                            }
                        )
                    elif m_type == "v":
                        tracking["v"].append(
                            {
                                "gt_embd": np.stack([safe_detach(embd_A), safe_detach(embd_T)], axis=0),
                                "obs_embd": safe_detach(fused_with_missing),
                                "rec_logits": safe_detach(rec_logits),
                                "rec_embds": np.stack([safe_detach(A), safe_detach(T)], axis=0),
                                "labels": label,
                                
                            }
                        )
                    elif m_type == "t":
                        tracking["t"].append(
                            {
                                "gt_embd": np.stack([safe_detach(embd_A), safe_detach(embd_V)], axis=0),
                                "obs_embd": safe_detach(fused_with_missing),
                                "rec_logits": safe_detach(rec_logits),
                                "rec_embds": np.stack([safe_detach(A), safe_detach(V)], axis=0),
                                "labels": label,
                                
                            }
                        )
                    elif m_type == "av":
                        tracking["av"].append(
                            {
                                "gt_embd": safe_detach(embd_T),
                                "obs_embd": safe_detach(fused_with_missing),
                                "rec_logits": safe_detach(rec_logits),
                                "rec_embds": safe_detach(T),
                                "labels": label,
                                "labels": label,
                                
                                
                            }
                        )
                    elif m_type == "at":
                        tracking["at"].append(
                            {
                                "gt_embd": embd_V,
                                "obs_embd": safe_detach(fused_with_missing),
                                "rec_logits": safe_detach(rec_logits),
                                "rec_embds": safe_detach(V),
                                "labels": label,
                            }
                        )
                    elif m_type == "vt" or m_type == "tv":
                        tracking["tv"].append(
                            {
                                "gt_embd": embd_A,
                                "obs_embd": safe_detach(fused_with_missing),
                                "rec_logits": safe_detach(rec_logits),
                                "rec_embds": safe_detach(A),
                                "labels": label,
                            }
                        )

        # Convert lists to numpy arrays
        for key in tracking:
            tracking[key] = np.array([item for item in tracking[key] if item is not None])
        # Save embeddings to file
        console.print(f"Saving embeddings to {out_fp}...")
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