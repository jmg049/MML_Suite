from itertools import combinations
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from config.multimodal_training_config import TrainingConfig
from config.federated_configs import FederatedExperimentConfig
from experiment_utils.checkpoints import CheckpointManager
from experiment_utils.logging import get_logger
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.printing import (
    get_console,
    log_epoch_time,
    log_total_time,
    print_debug,
    print_error,
    print_info,
    print_metric_summary,
    print_success,
    print_warning,
)
from experiment_utils.utils import (
    CONDITION,
    clean_checkpoints,
    ensure_dir,
    flatten_dict,
    get_correct_cmam_dataset_selected_patterns,
    prepare_metrics_for_json,
    safe_extend,
)
from models.cmams import SimpleCMAM
from models.protocols import MultimodalModelProtocol
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from train_multimodal import check_early_stopping

from fed import DataSplitType

console = get_console()
logger = get_logger()


def move_optimizer_to_device(optimizer: Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


@dataclass
class ClientModelData:
    """
    Represents the training data for a client.
    Contains the model, dataloaders, optimizer, loss function, and other training parameters.
    """

    model: MultimodalModelProtocol
    optimizer: Optimizer
    loss_function: LossFunctionGroup
    checkpoint_manager: CheckpointManager
    scheduler: Optional[LRScheduler] = None
    

@dataclass
class ClientCMAMData:
    """
    Represents the CMAM (Class Memory Augmentation Module) data for a client.
    Contains the CMAM model, optimizer, and loss function.
    """

    cmam: SimpleCMAM
    cmam_optimizer: Optimizer
    cmam_loss_function: LossFunctionGroup
    checkpoint_manager: CheckpointManager


@dataclass
class ClientExpData:
    config: FederatedExperimentConfig
    experiment_data: dict[str, Any]
    metric_recorder: MetricRecorder
    device: str = "cpu"


@dataclass
class ClientMetricsResult:
    """
    Represents the result of a client's training process.
    Contains the results for each data split.
    """

    client_id: int
    results: dict[DataSplitType, dict[str, Any]]

    epoch: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], client_id: int) -> "ClientMetricsResult":
        """
        Initializes the ClientMetricsResult from a dictionary.
        """
        results = {
            DataSplitType.TRAIN: data.get("train", {}),
            DataSplitType.VALIDATION: data.get("validation", {}),
            DataSplitType.TEST: data.get("test", {}),
        }

        return cls(results=results, client_id=client_id)

    @property
    def train_results(self) -> dict[str, Any]:
        """
        Returns the training results for the 'train' data split.
        """
        return self.results.get(DataSplitType.TRAIN, {})

    @property
    def val_results(self) -> dict[str, Any]:
        """
        Returns the validation results for the 'val' data split.
        """
        return self.results.get(DataSplitType.VALIDATION, {})

    @property
    def test_results(self) -> dict[str, Any]:
        """
        Returns the testing results for the 'test' data split.
        """
        return self.results.get(DataSplitType.TEST, {})


@dataclass(kw_only=True)
class Client:
    id: int
    model: ClientModelData
    exp_data: ClientExpData
    available_modalities: str
    dataloaders: dict[DataSplitType, DataLoader]
    metric_recorder: MetricRecorder
    full_modality_pattern: str
    baseline: bool = False
    tracked_modalities: list[str] = None
    cmam_data: Optional[list[ClientCMAMData]] = None
    is_incongruent: bool = False
    bytes_received: int = 0
    bytes_sent: int = 0
    cmam_bytes_received: int = 0
    cmam_bytes_sent: int = 0
    current_round: int = 0

    @property
    def num_cmams(self) -> int:
        """
        Returns the number of CMAMs associated with the client.
        If no CMAM data is present, returns 0.
        """
        return len(self.cmam_data) if self.cmam_data else 0

    @property
    def local_model(self) -> MultimodalModelProtocol:
        """
        Returns the local model for the client.
        """
        return self.model.model

    @property
    def lm_optimizer(self) -> Optimizer:
        """
        Returns the optimizer for the local model.
        """
        return self.model.optimizer

    @property
    def lm_loss(self) -> LossFunctionGroup:
        """
        Returns the loss function for the local model.
        """
        return self.model.loss_function

    @property
    def lm_scheduler(self) -> Optional[LRScheduler]:
        """
        Returns the learning rate scheduler for the local model, if any.
        """
        return self.model.scheduler

    @property
    def cmam(self) -> Optional[SimpleCMAM]:
        """
        Returns the CMAM model if present. For clients with multiple CMAMs,
        returns the first one. Most clients will have only one CMAM.
        """
        if self.cmam_data and len(self.cmam_data) > 0:
            return self.cmam_data[0].cmam
        return None

    @property
    def cmam_optimizer(self) -> Optional[Optimizer]:
        """
        Returns the optimizer for the CMAM model if present.
        """
        if self.cmam_data and len(self.cmam_data) > 0:
            return self.cmam_data[0].cmam_optimizer
        return None

    @property
    def cmam_loss_function(self) -> Optional[LossFunctionGroup]:
        """
        Returns the loss function for the CMAM model if present.
        """
        if self.cmam_data and len(self.cmam_data) > 0:
            return self.cmam_data[0].cmam_loss_function
        return None

    @property
    def device(self) -> str:
        """
        Returns the device used by the client.
        """
        return self.exp_data.device

    # @property
    # def checkpoint_manager(self) -> CheckpointManager:
    #     """
    #     Returns the checkpoint manager for the client.
    #     """
    #     return self.exp_data.checkpoint_manager

    @property
    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.
        """
        try:
            return self.dataloaders[DataSplitType.TRAIN]
        except KeyError as key_err:
            raise KeyError(
                f"Train dataloader not found in dataloaders. Available keys: {list(self.dataloaders.keys())}"
            ) from key_err

    @property
    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.
        """
        try:
            return self.dataloaders[DataSplitType.VALIDATION]
        except KeyError as key_err:
            raise KeyError(
                f"Validation dataloader not found in dataloaders. Available keys: {list(self.dataloaders.keys())}"
            ) from key_err

    def get_cmam_dataloaders(self, cmam: SimpleCMAM) -> dict[DataSplitType, DataLoader]:
        """
        Create pattern-filtered dataloaders for a specific C-MAM using dynamic patterns.

        This uses the new context manager approach to avoid dataset duplication while
        ensuring C-MAMs only receive samples with their required modality pattern.

        Args:
            cmam: The C-MAM model to create filtered dataloaders for

        Returns:
            Dictionary of context managers that yield filtered dataloaders for train/validation/test splits
        """
        from fed.data_utils import create_cmam_context_dataloader

        filtered_dataloaders = {}

        for split_type, base_dataloader in self.dataloaders.items():
            # Create context manager for this C-MAM using the new dynamic pattern approach
            filtered_dataloader = create_cmam_context_dataloader(
                base_dataloader=base_dataloader,
                input_modalities=cmam.input_modalities,
                target_modality=cmam.target_modality,
                batch_size=base_dataloader.batch_size,
                num_workers=base_dataloader.num_workers,
                pin_memory=base_dataloader.pin_memory,
                drop_last=base_dataloader.drop_last,
                collate_fn=base_dataloader.collate_fn,
            )

            filtered_dataloaders[split_type] = filtered_dataloader

        return filtered_dataloaders
    
    def get_dataloaders(self, cmam: SimpleCMAM, required_pattern: Optional[str] = None) -> dict[DataSplitType, DataLoader]:
        """
        Create pattern-filtered dataloaders for a specific C-MAM using dynamic patterns.

        This uses the new context manager approach to avoid dataset duplication while
        ensuring C-MAMs only receive samples with their required modality pattern.

        Args:
            cmam: The C-MAM model to create filtered dataloaders for

        Returns:
            Dictionary of context managers that yield filtered dataloaders for train/validation/test splits
        """
        from fed.data_utils import create_cmam_context_dataloader

        filtered_dataloaders = {}

        for split_type, base_dataloader in self.dataloaders.items():
            # Create context manager for this C-MAM using the new dynamic pattern approach
            filtered_dataloader = create_cmam_context_dataloader(
                base_dataloader=base_dataloader,
                input_modalities=cmam.input_modalities,
                target_modality=None,
                batch_size=base_dataloader.batch_size,
                num_workers=base_dataloader.num_workers,
                pin_memory=base_dataloader.pin_memory,
                drop_last=base_dataloader.drop_last,
                collate_fn=base_dataloader.collate_fn,
                required_pattern=required_pattern
            )

            filtered_dataloaders[split_type] = filtered_dataloader

        return filtered_dataloaders

    @property
    def test_dataloader(self) -> DataLoader:
        """
        Returns the testing dataloader.
        """
        try:
            return self.dataloaders[DataSplitType.TEST]
        except KeyError as key_err:
            raise KeyError(
                f"Test dataloader not found in dataloaders. Available keys: {list(self.dataloaders.keys())}"
            ) from key_err

    @property
    def experiment_data(self) -> dict[str, Any]:
        """
        Returns the experiment data for the client.
        """
        return self.exp_data.experiment_data

    @property
    def do_early_stopping(self) -> bool:
        """
        Returns whether early stopping is enabled for the client.
        """
        return self.exp_data.config.fed_config.client_training.early_stopping

    @property
    def metrics_fp(self) -> Path:
        """Returns the file path for the client with round information."""
        return ensure_dir(
            Path(self.exp_data.config.logging.metrics_path) / f"client_{self.id}" / f"round_{self.current_round}"
        )

    @property
    def model_fp(self) -> Path:
        """Returns the file path for the client's model with round information."""
        return ensure_dir(
            Path(self.exp_data.config.logging.model_output_path) / f"client_{self.id}" / f"round_{self.current_round}"
        )

    @property
    def logging_fp(self) -> Path:
        """Returns the file path for the client's logging with round information."""
        return ensure_dir(
            Path(self.exp_data.config.logging.logging_path) / f"client_{self.id}" / f"round_{self.current_round}"
        )

    @property
    def model_paramters(self) -> dict[str, Any]:
        """
        Returns the parameters of the local model.
        This is useful for saving and loading model parameters.
        """
        return self.local_model.state_dict()

    @property
    def model_parameters_size_bytes(self) -> int:
        """
        Returns the size of the local model's parameters in bytes.
        This is useful for tracking model size.
        """
        return sum(p.numel() * p.element_size() for p in self.local_model.parameters())

    @property
    def cmam_parameters(self) -> Optional[dict[str, Any]]:
        """
        Returns the parameters of the CMAM model if present.
        This is useful for saving and loading CMAM parameters.
        """
        return self.cmam.state_dict() if self.cmam else None

    def set_current_round(self, round_number: int) -> None:
        """
        Set the current federated learning round number for this client.
        This affects the directory structure for saving metrics, models, and logs.

        Args:
            round_number: The current round number (0-indexed or 1-indexed depending on convention)
        """
        self.current_round = round_number
        print_info(console, f"Client {self.id} set to round {round_number}")

    def run_base_model_round(self) -> ClientMetricsResult:
        """
        Train only the base model for one round (congruent federated learning phase 1).
        """
        self.local_model.to(self.device)
        move_optimizer_to_device(self.lm_optimizer, self.device)

        console.start_task(f"Client {self.id} -- Base Model Training", total=self.epochs, style="blue")

        results = self._train_mm_model(do_test=False, requires_regular_training=True)

        console.complete_task(f"Client {self.id} -- Base Model Training")
        self.local_model.to("cpu")
        move_optimizer_to_device(self.lm_optimizer, "cpu")

        print_success(console, f"✓ Client {self.id} base model training completed.")
        return results

    def run_cmam_round(
        self,
    ) -> Optional[ClientMetricsResult]:
        """
        Train only the C-MAMs for one round (congruent federated learning phase 2).
        """
        if not self.cmam_data:
            print_warning(console, f"Client {self.id} has no C-MAM data, skipping C-MAM training")
            return None

        # Move models to device
        self.local_model.to(self.device)
        self.local_model.eval()  # Base model should be frozen during C-MAM training

        for cmam_data in self.cmam_data:
            cmam_data.cmam.to(self.device)

        console.start_task(f"Client {self.id} -- C-MAM Training", total=1, style="cyan")

        cmam_results = self._train_cmams()

        console.complete_task(f"Client {self.id} -- C-MAM Training")

        # Move models back to CPU
        self.local_model.to("cpu")
        for cmam_data in self.cmam_data:
            cmam_data.cmam.to("cpu")
            clean_checkpoints(
                cmam_data.checkpoint_manager.model_dir, round=self.current_round, keep_best=True, keep_last=True
            )

        print_success(console, f"✓ Client {self.id} C-MAM training completed.")
        return cmam_results

    def run_incongruent_round(self) -> ClientMetricsResult:
        """
        Run incongruent federated training round.

        Based on algorithm specification:
        - Complete clients (have all modalities): Train like congruent clients (base model first, then C-MAMs)
        - Incomplete clients (missing some modalities): Train MM and C-MAM simultaneously using classification loss

        Returns:
            ClientMetricsResult containing both base model and C-MAM training results
        """
        logger.info(f"Client {self.id} starting incongruent training round")
        print_info(console, f"Client {self.id} performing incongruent training")

        if self.baseline:
            cmam_epochs = 0
        elif isinstance(self.exp_data.config.fed_config.client_cmam_training, TrainingConfig):
            cmam_epochs = self.exp_data.config.fed_config.client_cmam_training.epochs
        elif isinstance(self.exp_data.config.fed_config.client_cmam_training, dict):
            # try and find epochs in there
            try:
                cmam_epochs = self.exp_data.config.fed_config.client_cmam_training["epochs"]
            except KeyError as e:
                print_error(
                    console,
                    f"Key 'epochs' not found in client_cmam_training config for Client {self.id}. "
                    "Ensure your config is correct. Available keys: {list(self.exp_data.config.fed_config.client_cmam_training.keys())}",
                )
                raise e            

        # Determine if client has complete modalities
        is_complete_client = self._is_complete_modality_client()

        if is_complete_client and cmam_epochs > 0:
            # Complete clients: Train like congruent setting (base model first, then C-MAMs)
            print_info(console, f"Client {self.id} has complete modalities - training congruently")
            base_result = self.run_base_model_round()
            print_info(console, f"Client {self.id} base model training completed")
            cmam_result = self.run_cmam_round() if self.cmam_data else None
            print_info(console, f"Client {self.id} C-MAM training completed")

            # Combine results into single ClientMetricsResult
            return self._combine_client_results(base_result, cmam_result)
        else:
            # Incomplete clients: Different behavior based on C-MAM availability
            if self.cmam_data and cmam_epochs > 0:
                # Train MM and C-MAM simultaneously
                print_info(console, f"Client {self.id} has incomplete modalities {self.available_modalities} - training incongruently with C-MAMs")
                return self._train_incongruent_simultaneous()
            else:
                # Baseline: Train only base model with available modalities (no C-MAMs)
                print_info(console, f"Client {self.id} has incomplete modalities - training baseline (base model only)")
                return self._train_incongruent_baseline()

    def _is_complete_modality_client(self) -> bool:
        """
        Determine if client has access to all modalities.

        Returns:
            True if client has complete modalities, False if missing some modalities
        """
        # Check if available_modalities indicates completeness
        if self.available_modalities == "complete":
            return True

        # Check if client is missing specific modalities
        if self.available_modalities.startswith("missing_"):
            return False

        # For pattern-based assignments (e.g., "av", "ai", "tv"), check against all expected modalities
        # Assume we expect at least 2-3 modalities for most multimodal tasks
        from modalities import Modality

        expected_modalities = {Modality.AUDIO, Modality.VIDEO, Modality.TEXT, Modality.IMAGE}
        available_modalities = self._parse_available_modalities()

        # Consider complete if has 3+ modalities or all expected modalities
        return len(available_modalities) >= 3 or available_modalities == expected_modalities

    def _parse_available_modalities(self) -> set:
        """Parse client's available_modalities string into set of Modality objects."""
        from modalities import Modality
        # Handle short patterns like "ai", "av", "tv", etc.
        pattern = self.available_modalities.lower()
        modality_map = {"a": Modality.AUDIO, "v": Modality.VIDEO, "t": Modality.TEXT}
        m = {modality_map[char] for char in pattern if char in modality_map}
        print_debug(console, f"Client {self.id} available modalities parsed: {m}")
        return m

    def _train_incongruent_simultaneous(self) -> ClientMetricsResult:
        """
        Train MM and C-MAM simultaneously for incomplete clients.

        The C-MAM reconstructs missing modalities and the MM uses both available and
        reconstructed embeddings for classification training.

        Returns:
            ClientMetricsResult containing training metrics
        """
        if not self.cmam_data:
            print_error(console, f"Client {self.id} has no C-MAM data for incongruent training")
            raise ValueError(
                f"Client {self.id} has no C-MAM data for incongruent training. Ensure C-MAMs are initialized."
            )

        # Move models to device
        self.local_model.to(self.device)
        move_optimizer_to_device(self.lm_optimizer, self.device)

        for cmam_data in self.cmam_data:
            cmam_data.cmam.to(self.device)
            move_optimizer_to_device(cmam_data.cmam_optimizer, self.device)

        console.start_task(f"Client {self.id} -- Incongruent Training", total=self.epochs, style="magenta")

        best_metrics = None
        wait = 0

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # === Simultaneous Training ===
            train_metrics = self._run_incongruent_train_epoch(epoch)
            self.write_metrics(train_metrics, epoch, DataSplitType.TRAIN, self.available_modalities)
            print_metric_summary(console, epoch, train_metrics, f"Client {self.id} incongruent train")
            log_epoch_time(console, logger, self.id, epoch, train_metrics["train_time_secs"])

            # === Validation ===
            val_metrics = self._run_incongruent_validation_epoch(epoch)
            self.write_metrics(val_metrics, epoch, DataSplitType.VALIDATION, self.available_modalities)
            print_metric_summary(console, epoch, val_metrics, f"Client {self.id} incongruent val")
            log_epoch_time(console, logger, self.id, epoch, val_metrics["validation_time_secs"])


                
            # Save best MM model checkpoint
            self.model.checkpoint_manager.save_checkpoint(
                model=self.local_model,
                optimizer=self.lm_optimizer,
                scheduler=self.lm_scheduler,
                epoch=epoch,
                round=self.current_round,
                metrics=val_metrics,
                is_best=True,
            )
            print_success(console, f">> New best MM model saved at epoch {epoch} for Client {self.id}")
            
            # Save best C-MAM model checkpoints
            for cmam_data in self.cmam_data:
                cmam_data.checkpoint_manager.save_checkpoint(
                    model=cmam_data.cmam,
                    optimizer=cmam_data.cmam_optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=True,
                    scheduler=None,
                    round=self.current_round,
                )
                target_modality = cmam_data.cmam.target_modality
                print_success(console, f">> New best C-MAM {target_modality} saved at epoch {epoch} for Client {self.id}")
            
            log_epoch_time(console, logger, self.id, epoch, time.time() - epoch_start)

        console.complete_task(f"Client {self.id} -- Incongruent Training")

        # Final testing
        try:
            test_metrics = self._test_incongruent()
        except Exception as e:
            print_error(console, f"Error during incongruent testing for Client {self.id}: {e}")
            raise e
        
        # Move models back to CPU
        self.local_model.to("cpu")
        move_optimizer_to_device(self.lm_optimizer, "cpu")
        for cmam_data in self.cmam_data:
            cmam_data.cmam.to("cpu")
            move_optimizer_to_device(cmam_data.cmam_optimizer, "cpu")

        result = ClientMetricsResult.from_dict(
            {
                DataSplitType.TRAIN: train_metrics,
                DataSplitType.VALIDATION: val_metrics,
                DataSplitType.TEST: test_metrics,
            },
            client_id=self.id,
        )

        print_success(console, f"✓ Client {self.id} incongruent training completed")
        return result

    def _combine_client_results(
        self, base_result: ClientMetricsResult, cmam_result: Optional[ClientMetricsResult]
    ) -> ClientMetricsResult:
        """
        Combine base model and C-MAM results for complete clients.

        Args:
            base_result: Results from base model training
            cmam_result: Results from C-MAM training (can be None)

        Returns:
            Combined ClientMetricsResult
        """
        if cmam_result is None:
            return base_result

        # Combine metrics by prefixing with model type
        combined_data = {}

        for split_type in [DataSplitType.TRAIN, DataSplitType.VALIDATION, DataSplitType.TEST]:
            base_metrics = base_result.results.get(split_type, {})
            if isinstance(cmam_result, ClientMetricsResult):
                cmam_metrics = cmam_result.results.get(split_type, {})
            elif isinstance(cmam_result, dict):
                cmam_metrics = cmam_result.get(split_type, {})
            else:
                print_error(
                    console,
                    f"Unexpected type for cmam_result: {type(cmam_result)}. Expected ClientMetricsResult or dict.",
                )
                exit(1)

            # Prefix metrics to avoid conflicts
            combined_metrics = {}
            for key, value in base_metrics.items():
                combined_metrics[f"base_{key}"] = value
            for key, value in cmam_metrics.items():
                combined_metrics[f"cmam_{key}"] = value

            combined_data[split_type] = combined_metrics

        return ClientMetricsResult(client_id=self.id, results=combined_data)

    def _run_incongruent_train_epoch(self, epoch: int) -> dict[str, Any]:
        """
        Run one training epoch with simultaneous MM and C-MAM training.

        Args:
            epoch: Current epoch number

        Returns:
            Training metrics for this epoch
        """
        self.metric_recorder.reset()
        losses = defaultdict(list)
        timings = []

        console.start_task(f"Client {self.id} Incongruent Training", total=len(self.train_dataloader), style="green")

        dataloaders = self.get_cmam_dataloaders(self.cmam_data[0].cmam)

        with dataloaders.get(DataSplitType.TRAIN, self.train_dataloader) as train_dataloader:
            for batch in train_dataloader:
                start = time.time()

                # Train each C-MAM with simultaneous MM + C-MAM updates
                for cmam_data in self.cmam_data:
                    output = cmam_data.cmam.train_incongruent_step(
                        batch,
                        optimizer=cmam_data.cmam_optimizer,
                        loss_functions=cmam_data.cmam_loss_function,
                        device=self.device,
                        metric_recorder=self.metric_recorder,
                        model=self.local_model,
                        model_optimizer=self.lm_optimizer,  # Add base model optimizer
                        model_loss_functions=self.lm_loss,  # Add base model loss functions
                    )

                    losses["loss"].append(output["loss"])
                    for k, v in output.get("other_losses", {}).items():
                        losses[k].append(v.item() if hasattr(v, "item") else v)

                timings.append(time.time() - start)
                console.update_task(f"Client {self.id} Incongruent Training", advance=1)

            console.complete_task(f"Client {self.id} Incongruent Training")

            avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
            duration = np.sum(timings)
            per_sample = np.mean(timings) / train_dataloader.batch_size

            metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=epoch, loss=avg_loss))
            metrics.update(
                {
                    "loss": avg_loss,
                    "train_time_secs": duration,
                    "per_sample_time": per_sample,
                }
            )

            # Add other losses to metrics
            for key, values in losses.items():
                if key != "loss":
                    metrics[key] = np.mean(values)

            return metrics

    def _run_incongruent_validation_epoch(self, epoch: int) -> dict[str, Any]:
        """
        Run one validation epoch for incongruent training.

        Args:
            epoch: Current epoch number

        Returns:
            Validation metrics for this epoch
        """
        self.metric_recorder.reset()
        losses = defaultdict(list)
        timings = []

        console.start_task(f"Client {self.id} Incongruent Validation", total=len(self.val_dataloader), style="yellow")

        targets, preds, logits = [], [], []
        cmam_dataloaders = self.get_cmam_dataloaders(self.cmam_data[0].cmam)

        with torch.no_grad():
            with cmam_dataloaders.get(DataSplitType.VALIDATION, self.val_dataloader) as val_dataloader:
                for batch in val_dataloader:
                    start = time.time()

                    # Use the first C-MAM for validation (assuming single C-MAM per incomplete client)
                    if self.cmam_data:
                        cmam = self.cmam_data[0].cmam
                        loss_functions = self.cmam_data[0].cmam_loss_function

                        # Run validation step similar to C-MAM validation but for incongruent setting
                        output = cmam.validation_step(
                            batch=batch,
                            loss_functions=loss_functions,
                            device=torch.device(self.device),
                            metric_recorder=self.metric_recorder,
                            trained_model=self.local_model,
                        )

                        losses["loss"].append(output["loss"])
                        for k, v in output.get("other_losses", {}).items():
                            losses[k].append(v.item() if hasattr(v, "item") else v)

                        safe_extend(output, targets, "targets")
                        safe_extend(output, preds, "preds")
                        safe_extend(output, logits, "logits")

                    timings.append(time.time() - start)
                    console.update_task(f"Client {self.id} Incongruent Validation", advance=1)

            console.complete_task(f"Client {self.id} Incongruent Validation")

            avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
            duration = np.sum(timings)
            per_sample = np.mean(timings) / val_dataloader.batch_size

            metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=epoch, loss=avg_loss))
            metrics.update(
                {
                    "loss": avg_loss,
                    "validation_time_secs": duration,
                    "per_sample_time": per_sample,
                }
            )

            # Add other losses to metrics
            for key, values in losses.items():
                if key != "loss":
                    metrics[key] = np.mean(values)

            return metrics

    def _test_incongruent(self) -> dict[str, Any]:
        """
        Test the models after incongruent training.

        Returns:
            Test metrics
        """
        print_info(console, f"Client {self.id}: DEBUG - Starting _test_incongruent with baseline={getattr(self, 'baseline', 'NOT_SET')}")
        console.start_task(f"Client {self.id} -- Testing Incongruent", total=1, style="purple")
        self.metric_recorder.reset()
        losses = defaultdict(list)

        targets, preds, logits = [], [], []
        ground_truth_logits, reconstructed_logits = [], []
        ground_truth_embds, reconstructed_embds = defaultdict(list), defaultdict(list)
        ids, miss_types = [], []

        # Ensure model is on the correct device for testing
        self.local_model.to(self.device)
        self.local_model.eval()
        if self.cmam_data:
            for cmam_data in self.cmam_data:
                cmam_data.cmam.eval()
        cmam_dataloaders = self.get_cmam_dataloaders(self.cmam_data[0].cmam) if self.cmam_data else {}

        print_info(console, f"Client {self.id}: DEBUG - baseline={self.baseline}, cmam_data_exists={self.cmam_data is not None}")

        with torch.no_grad():
            with cmam_dataloaders.get(DataSplitType.TEST, self.test_dataloader) as test_dataloader:
                for batch in test_dataloader:
                    if self.cmam_data and not self.baseline:
                        print_info(console, f"Client {self.id}: DEBUG - Taking C-MAM testing path")
                        # Testing with C-MAMs (normal incongruent case)
                        cmam = self.cmam_data[0].cmam
                        loss_functions = self.cmam_data[0].cmam_loss_function

                        output = cmam.validation_step(
                            batch=batch,
                            loss_functions=loss_functions,
                            device=torch.device(self.device),
                            metric_recorder=self.metric_recorder,
                            trained_model=self.local_model,
                        )

                        for key in ["miss_type", "sample_ids"]:
                            if key not in output:
                                print_warning(
                                    console, f"Key '{key}' not found in incongruent test output for Client {self.id}"
                                )

                        safe_extend(output, targets, "targets")
                        safe_extend(output, preds, "preds")
                        safe_extend(output, logits, "logits")
                        safe_extend(output, ground_truth_logits, "ground_truth_logits")
                        safe_extend(output, ids, "sample_ids")
                        safe_extend(output, miss_types, "miss_type")

                        if len(output.get("reconstructed_logits", [])) == 0 and len(
                            output.get("reconstructed_embeddings", [])
                        ) == 0:
                            print_warning(
                                console,
                                f"Client {self.id} incongruent test output has no reconstructed logits or embeddings",
                            )
                            raise ValueError(
                                f"Client {self.id} incongruent test output has no reconstructed logits or embeddings"
                            )

                        # Handle embeddings if present
                        if "ground_truth_embeddings" in output:
                            # CMAM returns ground truth embeddings for the target modality
                            target_modality = str(cmam.target_modality).lower()
                            print_info(console, f"Client {self.id}: Extracting ground truth embeddings for target modality '{target_modality}'")
                            safe_extend(ground_truth_embds, output["ground_truth_embeddings"], target_modality)

                        if "reconstructed_logits" in output:
                            safe_extend(output, reconstructed_logits, "reconstructed_logits")

                        if "reconstructed_embeddings" in output:
                            # CMAM returns reconstructed embeddings for the target modality
                            target_modality = str(cmam.target_modality).lower()
                            print_info(console, f"Client {self.id}: Extracting reconstructed embeddings for target modality '{target_modality}'")
                            safe_extend(reconstructed_embds, output["reconstructed_embeddings"], target_modality)

                        losses["loss"].append(output["loss"])
                        for k, v in output.get("other_losses", {}).items():
                            losses[k].append(v.item() if hasattr(v, "item") else v)
                    else:
                        print_info(console, f"Client {self.id}: DEBUG - Taking baseline testing path")
                        # Baseline testing without C-MAMs (baseline + selective aggregation case)
                        print_info(console, f"Client {self.id}: Testing baseline model (no C-MAMs)")

                        # For baseline testing, we need to handle metrics manually since we want to use available modalities
                        # instead of missing patterns
                        available_pattern = self.available_modalities.upper()
                        print_info(console, f"Client {self.id}: Using pattern '{available_pattern}' for baseline metrics (available modalities: {self.available_modalities})")

                        # Create a temporary metric recorder for this batch to avoid contamination
                        temp_metric_recorder = MetricRecorder(
                            config=self.metric_recorder.config
                        )

                        # Call validation step with the temporary metric recorder
                        output = self.local_model.validation_step(
                            batch,
                            loss_functions=self.lm_loss,
                            device=torch.device(self.device),
                            metric_recorder=temp_metric_recorder,
                            return_test_info=True,
                        )

                        # Clear the temp recorder and manually update with correct pattern
                        temp_metric_recorder.reset()
                        from experiment_utils.utils import safe_detach
                        predictions = safe_detach(output["preds"])
                        labels = safe_detach(output["targets"])
                        # Use available pattern instead of missing pattern
                        corrected_miss_types = [available_pattern] * len(predictions)

                        print_info(console, f"Client {self.id}: Manually updating metrics with pattern '{available_pattern}' for {len(predictions)} predictions")

                        temp_metric_recorder.update_group_all(
                            "classification",
                            predictions=predictions,
                            targets=labels,
                            m_types=corrected_miss_types
                        )

                        # Transfer the corrected metrics to the main metric recorder
                        self.metric_recorder.update_group_all(
                            "classification",
                            predictions=predictions,
                            targets=labels,
                            m_types=corrected_miss_types
                        )

                        # Override the miss_type in output to reflect available modalities
                        output["miss_type"] = corrected_miss_types

                        for key in ["miss_type", "sample_ids"]:
                            if key not in output:
                                print_warning(
                                    console, f"Key '{key}' not found in baseline test output for Client {self.id}"
                                )

                        safe_extend(output, targets, "targets")
                        safe_extend(output, preds, "preds")
                        safe_extend(output, logits, "logits")
                        safe_extend(output, ground_truth_logits, "ground_truth_logits")
                        safe_extend(output, ids, "sample_ids")
                        safe_extend(output, miss_types, "miss_type")

                        # Handle ground truth embeddings (no reconstructed embeddings in baseline)
                        if "ground_truth_embeddings" in output:
                            # In baseline mode, extract embeddings for all available modalities
                            print_info(console, f"Client {self.id}: Extracting ground truth embeddings for available modalities: {list(output['ground_truth_embeddings'].keys())}")

                            # For baseline mode, we want to save embeddings keyed by the available modalities
                            # The tracked_modalities should match available_modalities for baseline clients
                            for modality in output["ground_truth_embeddings"]:
                                if output["ground_truth_embeddings"][modality] is not None:
                                    modality_str = str(modality).lower()
                                    safe_extend(
                                        output["ground_truth_embeddings"],
                                        ground_truth_embds[modality_str],
                                        modality,
                                        error=True,
                                    )
                                    print_info(console, f"Client {self.id}: Extracted {len(output['ground_truth_embeddings'][modality])} embeddings for modality {modality_str}")

                        losses["loss"].append(output["loss"])
                        for k, v in output.get("other_losses", {}).items():
                            losses[k].append(v.item() if hasattr(v, "item") else v)

            avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0

            # Reset and recalculate metrics to ensure clean state
            if self.baseline:
                print_info(console, f"Client {self.id}: DEBUG - Baseline mode, resetting metric recorder before final calculation")
                self.metric_recorder.reset()

                # Re-add all the correctly labeled data
                if targets and preds and miss_types:
                    # Handle concatenation of potentially zero-dimensional arrays
                    try:
                        all_predictions = np.concatenate([np.atleast_1d(p) for p in preds]) if preds else np.array([])
                        all_targets = np.concatenate([np.atleast_1d(t) for t in targets]) if targets else np.array([])
                        # miss_types are strings, handle differently
                        all_miss_types = []
                        for mt in miss_types:
                            if isinstance(mt, (list, np.ndarray)):
                                all_miss_types.extend(mt)
                            else:
                                all_miss_types.append(mt)
                        all_miss_types = np.array(all_miss_types)
                    except Exception as e:
                        print_error(console, f"Client {self.id}: Error concatenating arrays: {e}")
                        print_error(console, f"preds shapes: {[np.array(p).shape for p in preds]}")
                        print_error(console, f"targets shapes: {[np.array(t).shape for t in targets]}")
                        import sys
                        sys.exit(1)

                    if len(all_predictions) > 0:
                        print_info(console, f"Client {self.id}: DEBUG - Re-adding {len(all_predictions)} predictions with corrected patterns")
                        self.metric_recorder.update_group_all(
                            "classification",
                            predictions=all_predictions,
                            targets=all_targets,
                            m_types=all_miss_types
                        )

            test_metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=None, loss=avg_loss))
            test_metrics["loss"] = avg_loss

            if self.baseline:
                print_info(console, f"Client {self.id}: DEBUG - Final test metrics keys: {list(test_metrics.keys())}")

            # Add other losses to metrics
            for key, values in losses.items():
                if key != "loss":
                    test_metrics[key] = np.mean(values)

            if len(reconstructed_logits) == 0:
                reconstructed_logits = None
                reconstructed_embds = None

            self.write_metrics(test_metrics, epoch=None, split=DataSplitType.TEST, available_modalities=self.available_modalities)
            try:
                self.write_logits_and_embeddings(
                targets=targets,
                preds=preds,
                logits=logits,
                miss_types=miss_types,
                ids=ids,
                ground_truth_logits=ground_truth_logits,
                reconstructed_logits=reconstructed_logits,
                ground_truth_embds=ground_truth_embds,
                reconstructed_embds=reconstructed_embds,
            )   
            except Exception as e:
                raise e

            print_success(console, f"✓ Test metrics written for Client {self.id}")
            console.complete_task(f"Client {self.id} -- Testing Incongruent")

            # Move model back to CPU to save memory
            self.local_model.to("cpu")
            if self.cmam_data:
                for cmam_data in self.cmam_data:
                    cmam_data.cmam.to("cpu")

            return test_metrics

    def _train_incongruent_baseline(self) -> ClientMetricsResult:
        """
        Train baseline model for incomplete clients without C-MAMs.

        This provides a baseline comparison where incomplete clients train only on their
        available modalities without any missing modality reconstruction.

        Returns:
            ClientMetricsResult containing baseline training metrics
        """
        print_info(console, f"Client {self.id} training baseline (available modalities only)")

        # Train only the base model using available modalities
        # This is essentially the same as regular base model training
        base_result = self.run_base_model_round()

        # For baseline, there are no C-MAM results
        print_success(console, f"✓ Client {self.id} baseline training completed")
        return base_result

    def load_lm_state(self, state: dict[str, Any]) -> None:
        """
        Load the state of the local model from a given state dictionary.
        This is useful for resuming training or testing from a saved state.
        """
        try:
            self.local_model.load_state_dict(state["model_state_dict"])
        except KeyError as e:
            raise KeyError(f"Key 'model_state_dict' not found in state. Available keys: {list(state.keys())}") from e
        # self.lm_optimizer.load_state_dict(state.get("optimizer_state_dict", {}))
        # if self.lm_scheduler:
            # self.lm_scheduler.load_state_dict(state.get("scheduler_state_dict", {}))

    def load_cmam_state(self, state: dict[str, Any], idx: Optional[int]=None) -> None:
        """
        Load the state for the client's C-MAM models.

        Args:
            state: Dictionary containing C-MAM state data, typically with "model_state_dict" key
        """
        if not self.cmam_data:
            print_warning(console, f"Client {self.id} has no C-MAM data to load state into")
            return

        try:
            # For congruent federated training, usually there's only one C-MAM per client
            if "model_state_dict" in state:
                # Load state into the first (and typically only) C-MAM
                if len(self.cmam_data) > 0:
                    if idx is not None:
                        if idx < 0 or idx >= len(self.cmam_data):
                            raise IndexError(f"Index {idx} out of range for C-MAM data list of Client {self.id}")
                    else:
                        idx = 0
                    self.cmam_data[idx].cmam.load_state_dict(state["model_state_dict"])
                    print_info(console, f"Loaded C-MAM state for Client {self.id}")
                else:
                    print_warning(console, f"Client {self.id} has empty C-MAM data list")
            else:
                print_warning(console, f"No 'model_state_dict' found in state for Client {self.id}")

        except Exception as e:
            print_error(console, f"Error loading C-MAM state for Client {self.id}: {e}")
            raise e

    def _run_train_epoch(self, epoch: int, requires_regular_training: bool = False) -> dict[str, Any]:
        self.metric_recorder.reset()
        losses = defaultdict(list)
        timings = []

        # Regardless of whether or not the C-MAMs are trained I think I still create them - so this should work 


        if requires_regular_training:
            print_info(console, f"Client {self.id} running regular training epoch {epoch}")
            pattern = self.full_modality_pattern
        else:
            pattern = None
        
        dataloaders = self.get_dataloaders(self.cmam_data[0].cmam, required_pattern=pattern)
        
        with dataloaders[DataSplitType.TRAIN] as train_dataloader:

            for batch in train_dataloader:
                start = time.time()
                output = self.local_model.train_step(
                    batch,
                    optimizer=self.lm_optimizer,
                    loss_functions=self.lm_loss,
                    device=self.device,
                    metric_recorder=self.metric_recorder,
                )
                timings.append(time.time() - start)

                losses["loss"].append(output["loss"])
                for k, v in output.get("other_losses", {}).items():
                    losses[k].append(v.item())

                console.update_task("Training", advance=1)

            console.complete_task("Training")

            avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
            duration = np.sum(timings)
            per_sample = np.mean(timings) / train_dataloader.batch_size

            metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=epoch, loss=avg_loss))
            metrics.update(
                {
                    "loss": avg_loss,
                    "train_time_secs": duration,
                    "per_sample_time": per_sample,
                }
            )

            return metrics

    def _run_validation_epoch(self, epoch: int, requires_regular_validation: bool = False) -> dict[str, Any]:
        self.metric_recorder.reset()
        losses = defaultdict(list)
        
        if requires_regular_validation:
            print_info(console, f"Client {self.id} running regular validation epoch {epoch}")
            
            # Generate all possible patterns for comprehensive validation
            pattern = self.full_modality_pattern
            pattern_chars = list(pattern)  # Convert str to char list
            patterns = []
            for r in range(1, len(pattern_chars) + 1):
                for combo in combinations(pattern_chars, r):
                    pattern_name = "".join(sorted(combo))
                    patterns.append(pattern_name)
            patterns = sorted(patterns)
            print_debug(console, f"Client {self.id} validation patterns: {patterns}")
            
            # Run validation on each pattern separately and aggregate results
            all_targets, all_preds, all_logits = [], [], []
            total_duration = 0
            pattern_sample_counts = []
            
            for pattern_str in patterns:
                print_debug(console, f"Client {self.id} validating with pattern: {pattern_str}")
                
                # Get dataloader for this specific pattern
                dataloaders = self.get_dataloaders(self.cmam_data[0].cmam, required_pattern=pattern_str)
                
                with dataloaders[DataSplitType.VALIDATION] as val_dataloader:
                    if len(val_dataloader) == 0:
                        print_debug(console, f"No samples for pattern {pattern_str}, skipping")
                        continue
                        
                    pattern_targets, pattern_preds, pattern_logits = [], [], []
                    pattern_losses = defaultdict(list)
                    pattern_timings = []
                    
                    console.start_task(f"Validation ({pattern_str})", total=len(val_dataloader), style="yellow")
                    for batch in val_dataloader:
                        start = time.time()
                        output = self.local_model.validation_step(
                            batch,
                            loss_functions=self.lm_loss,
                            device=self.device,
                            metric_recorder=self.metric_recorder,
                            epoch=epoch,
                        )

                        pattern_timings.append(time.time() - start)

                        pattern_losses["loss"].append(output["loss"])
                        for k, v in output.get("other_losses", {}).items():
                            pattern_losses[k].append(v.item())

                        safe_extend(output, pattern_targets, "targets")
                        safe_extend(output, pattern_preds, "preds")
                        safe_extend(output, pattern_logits, "logits")

                        console.update_task(f"Validation ({pattern_str})", advance=1)

                    console.complete_task(f"Validation ({pattern_str})")
                    
                    # Aggregate pattern results
                    pattern_duration = np.sum(pattern_timings)
                    total_duration += pattern_duration
                    pattern_sample_counts.append(len(val_dataloader.dataset))
                    
                    # Add pattern losses to overall losses
                    for key, values in pattern_losses.items():
                        losses[key].extend(values)
                    
                    # Add pattern results to overall results
                    all_targets.extend(pattern_targets)
                    all_preds.extend(pattern_preds)
                    all_logits.extend(pattern_logits)
                    
                    print_debug(console, f"Pattern {pattern_str}: {len(pattern_targets)} samples, avg_loss: {np.mean(pattern_losses['loss']):.4f}")
            
            # Calculate aggregated metrics
            avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
            total_samples = sum(pattern_sample_counts)
            per_sample = total_duration / total_samples if total_samples > 0 else 0.0
            
            targets, preds, logits = all_targets, all_preds, all_logits
        else:
            # Standard validation using client's assigned pattern
            dataloaders = self.get_dataloaders(self.cmam_data[0].cmam, required_pattern=None)

            with dataloaders[DataSplitType.VALIDATION] as val_dataloader:
                targets, preds, logits = [], [], []
                pattern_timings = []
                console.start_task("Validation", total=len(val_dataloader), style="yellow")
                
                for batch in val_dataloader:
                    start = time.time()
                    output = self.local_model.validation_step(
                        batch,
                        loss_functions=self.lm_loss,
                        device=self.device,
                        metric_recorder=self.metric_recorder,
                        epoch=epoch,
                    )

                    pattern_timings.append(time.time() - start)

                    losses["loss"].append(output["loss"])
                    for k, v in output.get("other_losses", {}).items():
                        losses[k].append(v.item())

                    safe_extend(output, targets, "targets")
                    safe_extend(output, preds, "preds")
                    safe_extend(output, logits, "logits")

                    console.update_task("Validation", advance=1)

                console.complete_task("Validation")

                avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
                total_duration = np.sum(pattern_timings)
                per_sample = np.mean(pattern_timings) / val_dataloader.batch_size if len(val_dataloader) > 0 else 0.0

        # Calculate final metrics
        metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=epoch, loss=avg_loss))
        metrics.update(
            {
                "loss": avg_loss,
                "validation_time_secs": total_duration,
                "per_sample_time": per_sample,
            }
        )

        return metrics

    @property
    def epochs(self) -> int:
        """
        Returns the number of epochs for training.
        This is useful for controlling the training loop.
        """
        return self.exp_data.config.fed_config.client_training.epochs

    def _train_mm_model(self, do_test: bool = False, requires_regular_training: bool = False) -> dict[str, Any]:
        console.start_task(f"Client {self.id} -- Epoch", total=self.epochs, style="blue")
        best_metrics = None
        wait = 0
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # === Training ===
            self.local_model.train()
            self.local_model.to(self.device)
            for param in self.local_model.parameters():
                param.requires_grad = True

            train_metrics = self._run_train_epoch(epoch, requires_regular_training=requires_regular_training)
            self.write_metrics(train_metrics, epoch, DataSplitType.TRAIN, self.available_modalities)
            self.experiment_data["metrics_history"]["train"].append(train_metrics.copy())
            self.experiment_data["timing_history"]["train"].append(train_metrics["train_time_secs"])
            print_metric_summary(console=console, epoch=epoch, metrics=train_metrics, split="train")
            log_epoch_time(console, logger, self.id, epoch, train_metrics["train_time_secs"])

            # === Validation ===
            # Check if comprehensive validation is enabled for congruent training
            val_metrics = self._run_validation_epoch(epoch, requires_regular_validation=requires_regular_training)

            self.write_metrics(val_metrics, epoch, DataSplitType.VALIDATION, self.available_modalities)
            self.experiment_data["metrics_history"]["validation"].append(val_metrics.copy())
            self.experiment_data["timing_history"]["validation"].append(val_metrics["validation_time_secs"])
            print_metric_summary(console=console, epoch=epoch, metrics=val_metrics, split="validation")
            log_epoch_time(console, logger, self.id, epoch, val_metrics["validation_time_secs"])

            # === Checkpointing & Early Stopping ===
            is_best, should_continue, wait = check_early_stopping(
                val_metrics=val_metrics,
                best_metrics=best_metrics,
                patience=self.exp_data.config.fed_config.client_training.early_stopping_patience,
                min_delta=self.exp_data.config.fed_config.client_training.early_stopping_min_delta,
                wait=wait,
                mode="minimize"
                if self.exp_data.config.fed_config.client_training.early_stopping_metric == "loss"
                else "maximize",
                target_metric=self.exp_data.config.fed_config.client_training.early_stopping_metric,
            )

            if is_best:
                best_metrics = val_metrics.copy()
                self.model.checkpoint_manager.save_checkpoint(
                    model=self.local_model,
                    optimizer=self.lm_optimizer,
                    scheduler=self.lm_scheduler,
                    epoch=epoch,
                    round=self.current_round,
                    metrics=val_metrics,
                    is_best=True,
                )
                print_success(console, f">> New best model saved at epoch {epoch}")

            if self.do_early_stopping and not should_continue:
                print_warning(console, f"Early stopping triggered at epoch {epoch}")
                break

            if self.lm_scheduler:
                target = val_metrics[self.exp_data.config.fed_config.client_training.early_stopping_metric]
                if isinstance(self.lm_scheduler, ReduceLROnPlateau):
                    self.lm_scheduler.step(target)
                else:
                    self.lm_scheduler.step()
                print_info(console, f"Learning rate: {self.lm_optimizer.param_groups[0]['lr']:.2e}")

            log_epoch_time(console, logger, self.id, epoch, time.time() - epoch_start)

        console.complete_task(f"Client {self.id} -- Epoch")

        # Final testing phase
        self.model.checkpoint_manager.load_checkpoint(model=self.local_model, load_best=True, round=self.current_round, ignore="cmams")
        test_metrics = self._test_lm() if do_test else {}
        result = ClientMetricsResult.from_dict(
            {
                DataSplitType.TRAIN: train_metrics,
                DataSplitType.VALIDATION: val_metrics,
                DataSplitType.TEST: test_metrics,
            },
            client_id=self.id,
        )

        log_total_time(console, logger, self.id, train_metrics["train_time_secs"])
        return result

    def _test_lm(self) -> dict[str, Any]:
        """Test the local model on the client's assigned pattern only."""
        self.metric_recorder.reset()
        losses = defaultdict(list)

        targets, preds, logits = [], [], []
        ground_truth_logits, reconstructed_logits = [], []
        ground_truth_embds, reconstructed_embds = defaultdict(list), defaultdict(list)
        ids, miss_types = [], []
        # Regardless of whether or not the C-MAMs are trained I think I still create them - so this should work 
        if self.is_incongruent:
            exit(1)
            dataloaders = self.get_dataloaders(self.cmam_data[0].cmam)

            with dataloaders[DataSplitType.TEST] as test_dataloader:
                console.start_task(f"Client {self.id} -- Testing", total=len(test_dataloader), style="purple")
                for batch in test_dataloader:
                    output = self.local_model.validation_step(
                        batch,
                        loss_functions=self.lm_loss,
                        device=self.device,
                        metric_recorder=self.metric_recorder,
                    )

                    for key in ["miss_type", "sample_ids"]:
                        if key not in output:
                            raise KeyError(f"Key '{key}' not found in test output. Available keys: {list(output.keys())}")

                    safe_extend(output, targets, "targets")
                    safe_extend(output, preds, "preds")
                    safe_extend(output, logits, "logits")
                    safe_extend(output, ground_truth_logits, "ground_truth_logits")
                    safe_extend(output, ids, "sample_ids")
                    safe_extend(output, miss_types, "miss_type")
                    for modality in output["ground_truth_embeddings"]:
                        safe_extend(
                            output["ground_truth_embeddings"],
                            ground_truth_embds[modality],
                            modality,
                            error=True,
                        )

                    if "reconstructed_logits" in output:
                        safe_extend(output, reconstructed_logits, "reconstructed_logits")
                        for modality in output["reconstructed_embeddings"]:
                            safe_extend(
                                output["reconstructed_embeddings"],
                                reconstructed_embds[modality],
                                modality,
                                error=True,
                            )
                    losses["loss"].append(output["loss"])
                    for k, v in output.get("other_losses", {}).items():
                        losses[k].append(v.item())
                    console.update_task(f"Client {self.id} -- Testing", advance=1)
                avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
                test_metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=None, loss=avg_loss))
                test_metrics["loss"] = avg_loss

                if len(reconstructed_logits) == 0:
                    reconstructed_logits = None
                    reconstructed_embds = None

                self.write_metrics(test_metrics, epoch=None, split=DataSplitType.TEST, available_modalities=self.available_modalities)
                try:
                    self.write_logits_and_embeddings(
                    targets=targets,
                    preds=preds,
                    logits=logits,
                    miss_types=miss_types,
                    ids=ids,
                    ground_truth_logits=logits,
                    reconstructed_logits=reconstructed_logits,
                    ground_truth_embds=ground_truth_embds,
                    reconstructed_embds=reconstructed_embds,
                )
                except Exception as e:
                    raise e
                print_success(console, f"✓ Test metrics written for Client {self.id}")
                console.complete_task(f"Client {self.id} -- Testing")
                return test_metrics
        else:
                console.start_task(f"Client {self.id} -- Testing", total=len(self.test_dataloader), style="purple")
                for batch in self.test_dataloader:
                    print_info(console, f"Validating batch with {batch['pattern_names']}" )

                    if self.baseline:
                        # Baseline mode: use corrected patterns for metric recording
                        available_pattern = self.available_modalities.upper()
                        print_info(console, f"Client {self.id}: DEBUG - Baseline _test_lm using pattern '{available_pattern}' for available modalities: {self.available_modalities}")

                        # Call validation step without metric recorder to prevent wrong pattern recording
                        output = self.local_model.validation_step(
                            batch,
                            loss_functions=self.lm_loss,
                            device=self.device,
                            metric_recorder=None,
                            return_test_info=True,
                        )

                        # Manually record metrics with correct pattern
                        from experiment_utils.utils import safe_detach
                        predictions = safe_detach(output["preds"])
                        labels = safe_detach(output["targets"])
                        corrected_miss_types = [available_pattern] * len(predictions)

                        self.metric_recorder.update_group_all(
                            "classification",
                            predictions=predictions,
                            targets=labels,
                            m_types=corrected_miss_types
                        )

                        # Override the miss_type in output for consistency
                        output["miss_type"] = corrected_miss_types
                    else:
                        # Non-baseline mode: regular metric recording
                        output = self.local_model.validation_step(
                            batch,
                            loss_functions=self.lm_loss,
                            device=self.device,
                            metric_recorder=self.metric_recorder,
                        )

                    for key in ["miss_type", "sample_ids"]:
                        if key not in output:
                            raise KeyError(f"Key '{key}' not found in test output. Available keys: {list(output.keys())}")

                    safe_extend(output, targets, "targets")
                    safe_extend(output, preds, "preds")
                    safe_extend(output, logits, "logits")
                    safe_extend(output, ground_truth_logits, "ground_truth_logits")
                    safe_extend(output, ids, "sample_ids")
                    safe_extend(output, miss_types, "miss_type")
                    for modality in output["ground_truth_embeddings"]:
                        safe_extend(
                            output["ground_truth_embeddings"],
                            ground_truth_embds[modality],
                            modality,
                            error=True,
                        )

                    if "reconstructed_logits" in output:
                        safe_extend(output, reconstructed_logits, "reconstructed_logits")
                        for modality in output["reconstructed_embeddings"]:
                            safe_extend(
                                output["reconstructed_embeddings"],
                                reconstructed_embds[modality],
                                modality,
                                error=True,
                            )
                    losses["loss"].append(output["loss"])
                    for k, v in output.get("other_losses", {}).items():
                        losses[k].append(v.item())
                    console.update_task(f"Client {self.id} -- Testing", advance=1)
                avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
                test_metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=None, loss=avg_loss))
                test_metrics["loss"] = avg_loss

                if len(reconstructed_logits) == 0:
                    reconstructed_logits = None
                    reconstructed_embds = None

                self.write_metrics(test_metrics, epoch=None, split=DataSplitType.TEST, available_modalities=self.available_modalities)
                try:
                    self.write_logits_and_embeddings(
                    targets=targets,
                    preds=preds,
                    logits=logits,
                    miss_types=miss_types,
                    ids=ids,
                    ground_truth_logits=logits,
                    reconstructed_logits=reconstructed_logits,
                    ground_truth_embds=ground_truth_embds,
                    reconstructed_embds=reconstructed_embds,
                )
                except Exception as e:
                    raise e

                print_success(console, f"✓ Test metrics written for Client {self.id}")
                console.complete_task(f"Client {self.id} -- Testing")
                return test_metrics

    def test_client_model_post_round(self) -> dict[str, Any]:
        """
        Test the client model after local training within each federated round.
        This ensures clients are tested per round, not just at the end.
        """
        print_info(console, f"Testing Client {self.id} after round {self.current_round}")

        # Load the best checkpoint from this round
        self.model.checkpoint_manager.load_checkpoint(
            model=self.local_model,
            load_best=True,
            round=self.current_round,
            ignore="cmams"
        )

        # Ensure model is on the correct device for testing
        self.local_model.to(self.device)

        # Run the test using existing test method
        test_metrics = self._test_lm()

        # Move model back to CPU to save memory
        self.local_model.to("cpu")

        # Write metrics with round information
        self.write_metrics(test_metrics, f"round_{self.current_round}", DataSplitType.TEST, self.available_modalities)

        print_success(console, f"✓ Client {self.id} post-round testing completed")
        return test_metrics

    def test_client_cmams_post_round(self) -> dict[str, Any]:
        """
        Test the client C-MAMs after local C-MAM training within each federated round.
        This ensures C-MAMs are tested per round, not just at the end.
        """
        if not self.cmam_data:
            print_warning(console, f"Client {self.id} has no C-MAM data, skipping C-MAM testing")
            return {}

        print_info(console, f"Testing Client {self.id} C-MAMs after round {self.current_round}")

        # Load the base model checkpoint for C-MAM testing
        self.model.checkpoint_manager.load_checkpoint(
            model=self.local_model,
            load_best=True,
            round=self.current_round,
            ignore="cmams"
        )

        # Ensure base model is on the correct device for C-MAM testing
        self.local_model.to(self.device)

        combined_metrics = {}
        for cmam_data in self.cmam_data:
            # Load the best C-MAM checkpoint from this round
            cmam_data.checkpoint_manager.load_checkpoint(
                model=cmam_data.cmam,
                load_best=True,
                round=self.current_round
            )

            # Ensure C-MAM is on the correct device for testing
            cmam_data.cmam.to(self.device)

            # Test this specific C-MAM
            test_metrics = self._test_single_cmam(
                cmam=cmam_data.cmam,
                loss_functions=cmam_data.loss_function,
                target_modality=cmam_data.cmam.target_modality.value
            )

            # Write metrics with round information
            self.write_cmam_metrics(
                test_metrics,
                f"round_{self.current_round}",
                DataSplitType.TEST,
                cmam_data.cmam.target_modality.value
            )

            combined_metrics[cmam_data.cmam.target_modality.value] = test_metrics

        # Move models back to CPU to save memory
        self.local_model.to("cpu")
        for cmam_data in self.cmam_data:
            cmam_data.cmam.to("cpu")

        print_success(console, f"✓ Client {self.id} post-round C-MAM testing completed")
        return combined_metrics

    @property
    def cmam_parameter_size_bytes(self) -> int:
        """
        Returns the size of the CMAM model's parameters in bytes.
        This is useful for tracking CMAM model size.
        """
        return self.cmam.parameters_size_bytes if self.cmam else 0

    def _train_cmams(self) -> Optional[ClientMetricsResult]:
        """
        Train the client's C-MAM models following the centralized C-MAM training pattern.
        This method performs C-MAM training with the client's local data.

        Returns:
            Optional[ClientMetricsResult]: C-MAM training results, or None if no C-MAMs
        """
        if not self.cmam_data:
            print_warning(console, f"Client {self.id} has no C-MAM data to train")
            return None

        console.rule(f"[bold magenta]Client {self.id} -- Training C-MAMs")
        print_info(console, f"Client {self.id} training {len(self.cmam_data)} C-MAM(s)")

        # Move models to device
        self.local_model.to(self.device)
        self.local_model.eval()  # Base model should be in eval mode during C-MAM training

        cmam_results = {}

        for i, cmam_data in enumerate(self.cmam_data):
            correct_selected_patterns = get_correct_cmam_dataset_selected_patterns(
                condition=CONDITION, cmam=cmam_data.cmam
            )
            old_patterns = self.val_dataloader.dataset.patterns
            self.val_dataloader.dataset.patterns = [correct_selected_patterns]
            cmam = cmam_data.cmam
            optimizer = cmam_data.cmam_optimizer
            loss_functions = cmam_data.cmam_loss_function

            target_modality = cmam.target_modality
            print_info(
                console, f"Client {self.id} training C-MAM {i+1}/{len(self.cmam_data)} to reconstruct {target_modality}"
            )

            # Train single C-MAM
            cmam_result = self._train_single_cmam(
                cmam=cmam,
                optimizer=optimizer,
                loss_functions=loss_functions,
                target_modality=target_modality,
                cmam_data=cmam_data,
            )
            self.val_dataloader.dataset.patterns = old_patterns  # Restore original patterns

            old_patterns = self.test_dataloader.dataset.patterns
            self.test_dataloader.dataset.patterns = [correct_selected_patterns]

            # Test the trained C-MAM
            cmam_data.checkpoint_manager.load_checkpoint(model=cmam, load_best=True, round=self.current_round)
            cmam_test_result = self._test_single_cmam(
                cmam=cmam,
                loss_functions=loss_functions,
                target_modality=target_modality,
            )

            cmam_results[str(target_modality)] = cmam_test_result

            # cmam_results[str(target_modality)].update(cmam_test_result)
            self.test_dataloader.dataset.patterns = old_patterns  # Restore original patterns

        # Move models back to CPU
        self.local_model.to("cpu")
        for cmam_data in self.cmam_data:
            cmam_data.cmam.to("cpu")

        print_success(console, f"✓ Client {self.id} C-MAM training completed")

        # Return consolidated results - for now just return the first C-MAM's results
        # TODO: Better aggregation of multiple C-MAM results
        if cmam_results:
            first_result = next(iter(cmam_results.values()))
            return first_result
        return None

    def _train_single_cmam(
        self,
        cmam: SimpleCMAM,
        optimizer: Optimizer,
        loss_functions: LossFunctionGroup,
        target_modality: str,
        cmam_data: ClientCMAMData,
    ) -> ClientMetricsResult:
        """
        Train a single C-MAM following the centralized training pattern.

        Args:
            cmam: The C-MAM model to train
            optimizer: Optimizer for the C-MAM
            loss_functions: Loss function group for training
            target_modality: Target modality for logging

        Returns:
            ClientMetricsResult: Training results for this C-MAM
        """
        # Get C-MAM training configuration
        epochs = (
            self.exp_data.config.fed_config.client_training.cmam_epochs
            if hasattr(self.exp_data.config.fed_config.client_training, "cmam_epochs")
            else 1
        )

        # Set models to appropriate device and modes
        cmam.train()
        cmam.to(self.device)

        # Initialize training state
        best_metrics = None
        wait = 0

        console.start_task(
            f"Client {self.id} C-MAM Reconstructing {target_modality} -- Epoch", total=epochs, style="cyan"
        )

        train_metrics = {}
        val_metrics = {}

        # Training loop
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # === Training Phase ===
            cmam.train()
            epoch_train_metrics = self._run_cmam_train_epoch(cmam, loss_functions, optimizer, epoch, target_modality)

            save_name = f"{"-".join([str(m) for m in cmam.input_modalities])}_{str(target_modality)}"

            # Save training metrics
            self.write_cmam_metrics(
                epoch_train_metrics,
                epoch,
                DataSplitType.TRAIN,
                f"{"-".join([str(m) for m in cmam.input_modalities])}_{str(target_modality)}",
            )
            print_metric_summary(
                console=console,
                epoch=epoch,
                metrics=epoch_train_metrics,
                split="train",
                skip_conditions=[str(target_modality)[0]],
            )
            log_epoch_time(
                console,
                logger,
                f"Client{self.id}-CMAM-{target_modality}",
                epoch,
                epoch_train_metrics["train_time_secs"],
            )

            # === Validation Phase ===
            cmam.eval()
            epoch_val_metrics = self._run_cmam_validation_epoch(cmam, loss_functions, epoch, target_modality)

            # Save validation metrics
            self.write_cmam_metrics(epoch_val_metrics, epoch, DataSplitType.VALIDATION, save_name)
            print_metric_summary(
                console,
                epoch,
                epoch_val_metrics,
                f"Client {self.id} C-MAM Reconstructing {target_modality} Validation",
                skip_conditions=[str(target_modality)[0]],
            )
            log_epoch_time(
                console,
                logger,
                f"Client{self.id}-CMAM-Reconstructing-{target_modality}",
                epoch,
                epoch_val_metrics["validation_time_secs"],
            )

            # === Early Stopping & Checkpointing ===
            is_best, should_continue, wait = check_early_stopping(
                val_metrics=epoch_val_metrics,
                best_metrics=best_metrics,
                patience=self.exp_data.config.fed_config.client_training["early_stopping_patience"],
                min_delta=self.exp_data.config.fed_config.client_training["early_stopping_min_delta"],
                wait=wait,
                mode="minimize"
                if "loss" in self.exp_data.config.fed_config.client_training["early_stopping_metric"]
                else "maximize",
                target_metric=self.exp_data.config.fed_config.client_training["early_stopping_metric"],
            )

            if is_best or epoch == 1:
                best_metrics = epoch_val_metrics.copy()
                # Save best C-MAM checkpoint
                cmam_data.checkpoint_manager.save_checkpoint(
                    model=cmam,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=epoch_val_metrics,
                    is_best=is_best,
                    scheduler=None,
                    round=self.current_round,
                )
                print_success(
                    console,
                    f">> New best C-MAM For Reconstructing {target_modality} saved at epoch {epoch} for Client {self.id}",
                )

            if not should_continue:
                print_warning(
                    console, f"Early stopping triggered for Client {self.id} C-MAM {target_modality} at epoch {epoch}"
                )
                break

            # Store last epoch metrics for return
            train_metrics = epoch_train_metrics
            val_metrics = epoch_val_metrics

            log_epoch_time(console, logger, f"Client{self.id}-CMAM-{target_modality}", epoch, time.time() - epoch_start)

        console.complete_task(f"Client {self.id} C-MAM Reconstructing {target_modality} -- Epoch")

        # Move back to CPU to save memory
        cmam.to("cpu")

        print_success(console, f"✓ Client {self.id} C-MAM {target_modality} training completed")

        # Return result in expected format
        result = ClientMetricsResult.from_dict(
            {
                DataSplitType.TRAIN: train_metrics,
                DataSplitType.VALIDATION: val_metrics,
                DataSplitType.TEST: {},  # No test phase during training
            },
            client_id=self.id,
        )

        return result

    def _run_cmam_train_epoch(
        self,
        cmam: SimpleCMAM,
        loss_functions: LossFunctionGroup,
        optimizer: Optimizer,
        epoch: int,
        target_modality: str,
    ) -> dict[str, Any]:
        """
        Run one training epoch for a specific C-MAM.

        Args:
            cmam: The C-MAM model to train
            loss_functions: Loss function group for training
            optimizer: Optimizer for the C-MAM
            epoch: Current epoch number
            target_modality: Target modality for logging

        Returns:
            Dict containing training metrics
        """
        self.metric_recorder.reset()
        losses = defaultdict(list)
        timings = []

        # Get pattern-filtered dataloader for this specific C-MAM using context manager
        cmam_dataloaders = self.get_cmam_dataloaders(cmam)

        with cmam_dataloaders[DataSplitType.TRAIN] as train_dataloader:
            console.start_task(
                f"Client {self.id} C-MAM {target_modality} Training", total=len(train_dataloader), style="green"
            )

            for batch in train_dataloader:
                start = time.time()
                try:
                    train_output = cmam.train_step(
                        batch=batch,
                        loss_functions=loss_functions,
                        optimizer=optimizer,
                        device=torch.device(self.device),
                        epoch=epoch,
                        metric_recorder=self.metric_recorder,
                        trained_model=self.local_model,
                    )

                    loss = train_output["loss"]
                    losses["loss"].append(loss)

                    # Record other losses if present
                    other_losses = train_output.get("losses", {})
                    for key, value in other_losses.items():
                        losses[key].append(value.item() if hasattr(value, "item") else value)

                except Exception as e:
                    print_error(console, f"Error training C-MAM for {target_modality} on Client {self.id}: {e}")
                    import traceback

                    traceback.print_exc()
                    raise e

                timings.append(time.time() - start)
                console.update_task(f"Client {self.id} C-MAM {target_modality} Training", advance=1)

        console.complete_task(f"Client {self.id} C-MAM {target_modality} Training")

        # Calculate metrics
        avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
        duration = np.sum(timings)
        per_sample = np.mean(timings) / self.train_dataloader.batch_size

        metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=epoch, loss=avg_loss))
        metrics.update(
            {
                "loss": avg_loss,
                "train_time_secs": duration,
                "per_sample_time": per_sample,
            }
        )

        # Add other losses to metrics
        for key, values in losses.items():
            if key != "loss":
                metrics[key] = np.mean(values)

        return metrics

    def _run_cmam_validation_epoch(
        self, cmam: SimpleCMAM, loss_functions: LossFunctionGroup, epoch: int, target_modality: str
    ) -> dict[str, Any]:
        """
        Run one validation epoch for a specific C-MAM.

        Args:
            cmam: The C-MAM model to validate
            loss_functions: Loss function group for validation
            epoch: Current epoch number
            target_modality: Target modality for logging

        Returns:
            Dict containing validation metrics
        """
        self.metric_recorder.reset()
        losses = defaultdict(list)
        timings = []

        # Get pattern-filtered dataloader for this specific C-MAM using context manager
        cmam_dataloaders = self.get_cmam_dataloaders(cmam)

        with cmam_dataloaders[DataSplitType.VALIDATION] as val_dataloader:
            console.start_task(
                f"Client {self.id} C-MAM {target_modality} Validation", total=len(val_dataloader), style="yellow"
            )

            with torch.no_grad():
                for batch in val_dataloader:
                    start = time.time()
                    try:
                        val_output = cmam.validation_step(
                            batch=batch,
                            loss_functions=loss_functions,
                            device=torch.device(self.device),
                            metric_recorder=self.metric_recorder,
                            trained_model=self.local_model,
                        )

                        loss = val_output["loss"]
                        losses["loss"].append(loss)

                        # Record other losses if present
                        other_losses = val_output.get("losses", {})
                        for key, value in other_losses.items():
                            losses[key].append(value.item() if hasattr(value, "item") else value)

                    except Exception as e:
                        print_error(console, f"Error validating C-MAM for {target_modality} on Client {self.id}: {e}")
                        import traceback

                        traceback.print_exc()
                        raise e

                    timings.append(time.time() - start)
                    console.update_task(f"Client {self.id} C-MAM {target_modality} Validation", advance=1)

        console.complete_task(f"Client {self.id} C-MAM {target_modality} Validation")

        # Calculate metrics
        avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
        duration = np.sum(timings)
        per_sample = np.mean(timings) / self.val_dataloader.batch_size

        metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=epoch, loss=avg_loss))
        metrics.update(
            {
                "loss": avg_loss,
                "validation_time_secs": duration,
                "per_sample_time": per_sample,
            }
        )

        # Add other losses to metrics
        for key, values in losses.items():
            if key != "loss":
                metrics[key] = np.mean(values)

        return metrics

    def write_cmam_metrics(self, metrics: dict[str, Any], epoch: int, split: DataSplitType, modality: str,) -> None:
        """
        Write C-MAM metrics to JSON file.

        Args:
            metrics: Dictionary of metrics to save
            epoch: Current epoch number
            split: Data split type (train/validation/test)
            target_modality: Target modality for the C-MAM
        """
        metrics = prepare_metrics_for_json([metrics])[0]

        # Create C-MAM specific metrics directory
        cmam_metrics_fp = self.metrics_fp / "cmams" / modality
        cmam_metrics_fp.mkdir(parents=True, exist_ok=True)

        metrics_fp_split = cmam_metrics_fp / split.value
        metrics_fp_split.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_fp_split / f"{epoch}.json"

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        print_info(console, f"Client {str(self.id)} C-MAM {modality} metrics for epoch {epoch} saved to {metrics_file}")


    def _test_single_cmam(
        self,
        cmam: SimpleCMAM,
        loss_functions: LossFunctionGroup,
        target_modality: str,
        num_epochs: int = None,
        phase: str = "test",
    ) -> dict[str, Any]:
        """
        Test a single C-MAM following the centralized testing pattern.

        Args:
            cmam: The C-MAM model to test
            loss_functions: Loss function group for testing
            target_modality: Target modality for logging
            num_epochs: Unused, kept for compatibility with training call signature
            phase: Testing phase identifier

        Returns:
            Dict containing test metrics
        """
        console.start_task(f"Client {self.id} -- Testing C-MAM {target_modality}", total=1, style="purple")

        # Set models to appropriate modes
        cmam.eval()
        self.local_model.eval()

        # Reset metrics for fresh test evaluation
        self.metric_recorder.reset()
        losses = defaultdict(list)
        timings = []

        # Collect test data
        targets, preds, logits = [], [], []
        ground_truth_logits, reconstructed_logits = [], []
        ground_truth_embds, reconstructed_embds = [], []
        ids, miss_types = [], []

        # Get pattern-filtered dataloader for this specific C-MAM using context manager
        cmam_dataloaders = self.get_cmam_dataloaders(cmam)

        with cmam_dataloaders[DataSplitType.TEST] as test_dataloader:
            console.start_task(
                f"Client {self.id} C-MAM {target_modality} Testing", total=len(test_dataloader), style="purple"
            )

            with torch.no_grad():
                for batch in test_dataloader:
                    start = time.time()
                    try:
                        test_output = cmam.validation_step(
                            batch=batch,
                            loss_functions=loss_functions,
                            device=torch.device(self.device),
                            metric_recorder=self.metric_recorder,
                            trained_model=self.local_model,
                        )

                        loss = test_output["loss"]
                        losses["loss"].append(loss)

                        # Record other losses if present
                        other_losses = test_output.get("losses", {})
                        for key, value in other_losses.items():
                            losses[key].append(value.item() if hasattr(value, "item") else value)

                        # Collect test data for analysis
                        for key in ["miss_type", "sample_ids"]:
                            if key not in test_output:
                                print_warning(
                                    console, f"Key '{key}' not found in C-MAM test output for Client {self.id}"
                                )

                        # Safely extend arrays with test outputs
                        safe_extend(test_output, targets, "targets")
                        safe_extend(test_output, preds, "preds")
                        safe_extend(test_output, logits, "logits")
                        safe_extend(test_output, ground_truth_logits, "ground_truth_logits")
                        safe_extend(test_output, ids, "sample_ids")
                        safe_extend(test_output, miss_types, "miss_type")

                        # Handle embeddings if present
                        if "ground_truth_embeddings" in test_output:
                            safe_extend(
                                test_output,
                                ground_truth_embds,
                                "ground_truth_embeddings",
                                error=True,
                            )

                        if "reconstructed_logits" in test_output:
                            safe_extend(test_output, reconstructed_logits, "reconstructed_logits")
                            if "reconstructed_embeddings" in test_output:
                                safe_extend(
                                    test_output,
                                    reconstructed_embds,
                                    "reconstructed_embeddings",
                                    error=True,
                                )

                    except Exception as e:
                        print_error(console, f"Error testing C-MAM for {target_modality} on Client {self.id}: {e}")
                        import traceback

                        traceback.print_exc()
                        raise e

                    timings.append(time.time() - start)
                    console.update_task(f"Client {self.id} C-MAM {target_modality} Testing", advance=1)

        console.complete_task(f"Client {self.id} C-MAM {target_modality} Testing")

        # Calculate test metrics
        avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
        duration = np.sum(timings)
        per_sample = np.mean(timings) / self.test_dataloader.batch_size if len(self.test_dataloader) > 0 else 0.0

        test_metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=None, loss=avg_loss))
        test_metrics.update(
            {
                "loss": avg_loss,
                "test_time_secs": duration,
                "per_sample_time": per_sample,
            }
        )

        # Add other losses to metrics
        for key, values in losses.items():
            if key != "loss":
                test_metrics[key] = np.mean(values)

        # Save test metrics
        self.write_cmam_metrics(
            test_metrics,
            f"{phase}_metrics",
            DataSplitType.TEST,
            f"{'-'.join([str(m) for m in cmam.input_modalities])}_{str(target_modality)}",
        )

        # Clean up reconstructed data if empty
        if len(reconstructed_logits) == 0:
            reconstructed_logits = None
            reconstructed_embds = None

        print_metric_summary(
            console=console,
            epoch="test",
            metrics=test_metrics,
            split="test",
        )
        # Save logits and embeddings for analysis
        self.write_cmam_logits_and_embeddings(
            targets=targets,
            preds=preds,
            logits=logits,
            miss_types=miss_types,
            ids=ids,
            ground_truth_logits=ground_truth_logits,
            reconstructed_logits=reconstructed_logits,
            ground_truth_embds=ground_truth_embds,
            reconstructed_embds=reconstructed_embds,
            target_modality=target_modality,
            cmam_input_modalities=cmam.input_modalities,
        )

        console.complete_task(f"Client {self.id} -- Testing C-MAM {target_modality}")
        print_success(console, f"✓ Client {self.id} C-MAM {target_modality} testing completed")

        return test_metrics

    def write_cmam_logits_and_embeddings(
        self,
        targets: list[Any],
        preds: list[Any],
        logits: list[Any],
        miss_types: list[Any],
        ids: list[Any],
        ground_truth_logits: list[Any],
        ground_truth_embds: dict[str, list[Any]],
        target_modality: str,
        cmam_input_modalities: list[str],
        reconstructed_logits: Optional[list[Any]] = None,
        reconstructed_embds: Optional[dict[str, list[Any]]] = None,
    ) -> None:
        """
        Write C-MAM test results (logits and embeddings) to npz archive for analysis.

        Args:
            targets: Ground truth targets
            preds: Model predictions
            logits: Output logits
            miss_types: Missing modality patterns
            ids: Sample IDs
            ground_truth_logits: Ground truth logits from base model
            ground_truth_embds: Ground truth embeddings by modality
            target_modality: Target modality for this C-MAM
            cmam_input_modalities: Input modalities for this C-MAM
            reconstructed_logits: Reconstructed logits (optional)
            reconstructed_embds: Reconstructed embeddings (optional)
        """
        targets = np.array(targets)
        preds = np.array(preds)
        logits = np.array(logits)
        ground_truth_logits = np.array(ground_truth_logits)
        ids = np.array(ids)
        miss_types = np.array(miss_types)

        if reconstructed_logits is not None:
            reconstructed_logits = np.array(reconstructed_logits)

        # Create entry for this C-MAM's test results
        entry = {
            "targets": targets,
            "preds": preds,
            "logits": logits,
            "ids": ids,
            "ground_truth_logits": ground_truth_logits,
            "miss_types": miss_types,
            "target_modality": str(target_modality),
            "input_modalities": [str(m) for m in cmam_input_modalities],
        }

        ground_truth_embds = np.array(ground_truth_embds)
        entry["ground_truth_embeddings"] = ground_truth_embds

        # Add reconstructed data if available
        if reconstructed_logits is not None:
            entry["reconstructed_logits"] = np.array(reconstructed_logits)

        if reconstructed_embds is not None:
            reconstructed_embds = np.array(reconstructed_embds)
            entry["reconstructed_embeddings"] = reconstructed_embds

        # Create C-MAM specific output directory
        cmam_identifier = f"{'-'.join([str(m) for m in cmam_input_modalities])}_{str(target_modality)}"
        output_dir = self.metrics_fp / "cmams" / cmam_identifier
        output_dir.mkdir(parents=True, exist_ok=True)
        print_info(console, f"Saving C-MAM {target_modality} test results to {output_dir}")
        output_path = output_dir / f"client_{self.id}_cmam_test_results.npz"

        try:
            np.savez_compressed(output_path, **entry)
            print_success(
                console, f"✓ C-MAM {target_modality} test results saved for Client {self.id} at {output_path}"
            )
        except Exception as e:
            print_error(console, f"Failed to save C-MAM {target_modality} test results for Client {self.id}: {e}")
            raise

    def write_logits_and_embeddings(
        self,
        targets: list[Any],
        preds: list[Any],
        logits: list[Any],
        miss_types: list[Any],
        ids: list[Any],
        ground_truth_logits: list[Any],
        ground_truth_embds: list[Any],
        reconstructed_logits: Optional[list[Any]] = None,
        reconstructed_embds: Optional[list[Any]] = None,
    ) -> None:
        """
        Write the logits and embeddings to a npz archive for future analysis, grouped by available modality.
        Each entry is a dictionary of arrays, stored as an array of dicts.
        """

        targets = np.array(targets)
        preds = np.array(preds)
        logits = np.array(logits)
        ids = np.array(ids)
        miss_types = np.array(miss_types)
        
        # Convert ground_truth_logits to array
        ground_truth_logits = np.array(ground_truth_logits)
            
        # Handle reconstructed_logits
        if reconstructed_logits is not None and len(reconstructed_logits) > 0:
            reconstructed_logits = np.array(reconstructed_logits)
        else:
            reconstructed_logits = None

        tracking: dict[str, list[dict[str, np.ndarray]]] = {modality: [] for modality in self.tracked_modalities}

        print_info(console, f"Client {self.id} tracked_modalities: {self.tracked_modalities}")
        print_info(console, f"Client {self.id} available ground_truth_embds keys: {list(ground_truth_embds.keys()) if isinstance(ground_truth_embds, dict) else 'Not a dict'}")
        print_info(console, f"Client {self.id} available reconstructed_embds keys: {list(reconstructed_embds.keys()) if isinstance(reconstructed_embds, dict) else 'Not a dict'}")
        
        for modality in self.tracked_modalities:
            print_info(console, f"Processing modality '{modality}' for Client {self.id}")
            if np.unique(miss_types).size == 1:
                mask = np.ones_like(miss_types, dtype=bool)
            else:
                mask = miss_types == modality
                console.info(f"Client {self.id} modality '{modality}' mask sum: {np.sum(mask)} out of {len(mask)}")
                console.info(
                    f"Client {self.id} modality '{modality}'- mask {mask}"
                )

            if not np.any(mask):
                print_warning(console, f"No data found for modality '{modality}' in test batch. ")
                print_info(console, f"Available modalities: {self.tracked_modalities}")
                continue

            # Flatten 2D arrays that should be 1D for boolean indexing
            if isinstance(ids, np.ndarray) and ids.ndim > 1:
                print_info(console, f"Client {self.id}: Flattening ids array from shape {ids.shape} to 1D")
                ids = ids.flatten()
            if isinstance(miss_types, np.ndarray) and miss_types.ndim > 1:
                print_info(console, f"Client {self.id}: Flattening miss_types array from shape {miss_types.shape} to 1D")
                miss_types = miss_types.flatten()

            # Validate array lengths before boolean indexing
            arrays_to_check = {
                "targets": targets,
                "preds": preds,
                "logits": logits,
                "ids": ids,
                "ground_truth_logits": ground_truth_logits,
                "miss_types": miss_types
            }

            # Check that all arrays have the same length as the mask
            mask_length = len(mask)
            for name, array in arrays_to_check.items():
                array_length = len(array)
                if array_length != mask_length:
                    error_msg = f"Array length mismatch in Client {self.id}: {name} has length {array_length} but mask has length {mask_length}"
                    print_error(console, error_msg)
                    print_error(console, f"Available modalities: {self.tracked_modalities}")
                    print_error(console, f"Miss types: {np.unique(miss_types)}")
                    print_error(console, f"Mask: {mask}")
                    print_error(console, f"DEBUG: Array contents - {name}: {array}")
                    print_error(console, f"DEBUG: Array type - {name}: {type(array)}")
                    if hasattr(array, 'shape'):
                        print_error(console, f"DEBUG: Array shape - {name}: {array.shape}")
                    print_error(console, "CRITICAL: Data aggregation is STILL broken. Exiting immediately.")
                    import sys
                    sys.exit(1)

            entry = {
                "targets": targets[mask],
                "preds": preds[mask],
                "logits": logits[mask],
                "ids": ids[mask],
                "ground_truth_logits": ground_truth_logits[mask],
            }

            # Handle ground truth embeddings with proper validation
            if isinstance(ground_truth_embds, dict):
                for m, m_embd in ground_truth_embds.items():
                    if m == modality and m_embd is not None and len(m_embd) > 0:
                        try:
                            m_embd_array = np.array(m_embd)
                            if len(m_embd_array) == len(mask):
                                masked = m_embd_array[mask]
                                entry[f"ground_truth_embd_{m}"] = masked
                                print_info(console, f"Added ground truth embeddings for {m}: shape {masked.shape}")
                            else:
                                print_warning(console, f"Embedding length mismatch for {m}: {len(m_embd_array)} vs mask {len(mask)}")
                        except Exception as e:
                            print_warning(console, f"Error processing ground truth embeddings for {m}: {e}")
            
            # Handle reconstructed embeddings with proper validation  
            if reconstructed_embds is not None and isinstance(reconstructed_embds, dict):
                for m, m_embd in reconstructed_embds.items():
                    if m == modality and m_embd is not None and len(m_embd) > 0:
                        try:
                            m_embd_array = np.array(m_embd)
                            if len(m_embd_array) == len(mask):
                                masked = m_embd_array[mask]
                                entry[f"reconstructed_embd_{m}"] = masked
                                print_info(console, f"Added reconstructed embeddings for {m}: shape {masked.shape}")
                            else:
                                print_warning(console, f"Reconstructed embedding length mismatch for {m}: {len(m_embd_array)} vs mask {len(mask)}")
                        except Exception as e:
                            print_warning(console, f"Error processing reconstructed embeddings for {m}: {e}")
            
            # Handle reconstructed logits
            if reconstructed_logits is not None:
                try:
                    entry["reconstructed_logits"] = reconstructed_logits[mask]
                    print_info(console, f"Added reconstructed logits: shape {reconstructed_logits[mask].shape}")
                except Exception as e:
                    print_warning(console, f"Error processing reconstructed logits: {e}")

            tracking[modality].append(entry)

        # Final packaging: convert list of dicts into numpy array of dicts
        packaged = {
            modality: np.array(modality_entries) for modality, modality_entries in tracking.items() if modality_entries
        }

        if not packaged:
            print_error(console, f"No valid entries to write for Client {self.id}")
            print_info(console, f"Available modalities: {self.tracked_modalities}")
            print_info(console, f"Data collected: {tracking}")
            print_info(
                console, f"Logits shape: {logits.shape}, Targets shape: {targets.shape}, Preds shape: {preds.shape}"
            )
            print_info(console, f"Ground Truth Logits shape: {ground_truth_logits.shape}")
            print_info(
                console,
                f"Reconstructed Logits shape: {reconstructed_logits.shape if reconstructed_logits is not None else 'N/A'}",
            )
            print_info(
                console, f"Ground Truth Embeddings: {ground_truth_embds.keys() if ground_truth_embds else 'N/A'}"
            )
            print_info(
                console, f"Reconstructed Embeddings: {reconstructed_embds.keys() if reconstructed_embds else 'N/A'}"
            )
            print_info(console, f"Miss Types: {miss_types}")

            # raise ValueError(f"No valid entries to write for Client {self.id}. Check tracked modalities and data.")
        print_info(console, f"Writing logits and embeddings for Client {self.id} with modalities: {self.tracked_modalities}")
        output_path = self.metrics_fp / f"client_{self.id}_logits_embeddings.npz"
        try:
            np.savez_compressed(output_path, **packaged)
            print_success(console, f"✓ Logits and embeddings saved for Client {self.id} at {output_path}")
        except Exception as e:
            print_error(console, f"Failed to save logits and embeddings: {e}")
            raise

    def update_bytes_received_cmams(self, bytes_received: int) -> None:
        """
        Update the total bytes received by the client for CMAM data.
        This is useful for tracking data transfer during federated learning.
        """
        if self.cmam_data is None:
            print_error(console, f"Client {self.id} has no CMAM data to update bytes received.")
            return

        self.cmam_bytes_received += bytes_received
        print_info(
            console,
            f"Client {self.id} received {bytes_received} bytes for CMAM. Total: {self.cmam_bytes_received} bytes.",
        )

    def update_bytes_sent_cmams(self, bytes_sent: int) -> None:
        """
        Update the total bytes sent by the client for CMAM data.
        This is useful for tracking data transfer during federated learning.
        """
        if self.cmam_data is None:
            # print_error(console, f"Client {self.id} has no CMAM data to update bytes sent.")
            return

        self.cmam_bytes_sent += bytes_sent
        print_info(console, f"Client {self.id} sent {bytes_sent} bytes for CMAM. Total: {self.cmam_bytes_sent} bytes.")

    def update_bytes_received(self, bytes_received: int) -> None:
        """
        Update the total bytes received by the client.
        This is useful for tracking data transfer during federated learning.
        """
        self.bytes_received += bytes_received
        print_info(console, f"Client {self.id} received {bytes_received} bytes. Total: {self.bytes_received} bytes.")

    def update_bytes_sent(self, bytes_sent: int) -> None:
        """
        Update the total bytes sent by the client.
        This is useful for tracking data transfer during federated learning.
        """
        self.bytes_sent += bytes_sent
        print_info(console, f"Client {self.id} sent {bytes_sent} bytes. Total: {self.bytes_sent} bytes.")

    def write_metrics(self, metrics: dict[str, Any], epoch: Optional[int], split: DataSplitType, available_modalities: Optional[str]= None) -> None:
        if epoch is None:
            epoch = "test_metrics"

        metrics_fp = self.metrics_fp
        metrics_fp_split = metrics_fp / split.value
        metrics_fp_split.mkdir(parents=True, exist_ok=True)
        if available_modalities:
            metrics_fp_split = metrics_fp_split / available_modalities
            metrics_fp_split.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_fp_split / f"{epoch}.json"

        metrics = prepare_metrics_for_json([metrics])[0]

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        print_info(console, f"Metrics for Client {self.id} at epoch {epoch} saved to {metrics_file}")

    def info(self) -> str:
        s = ""

        s += f"Client ID: {self.id}\n"
        s += f"Device: {self.device}\n"
        s += f"Epochs: {self.epochs}\n"
        s += f"Local Model: {self.local_model.__class__.__name__}\n"
        s += f"Optimizer: {self.optimizer.__class__.__name__}\n"
        s += f"Loss Function: {self.loss_function.__class__.__name__}\n"
        if self.cmam:
            s += f"CMAM: {self.cmam.__class__.__name__}\n"
            s += f"CMAM Optimizer: {self.cmam_optimizer.__class__.__name__}\n"
            s += f"CMAM Loss Function: {self.cmam_loss_function.__class__.__name__}\n"
        else:
            s += "CMAM: Not used\n"

        for split, loader in self.dataloaders.items():
            s += f"{split.value.capitalize()} DataLoader: {loader.dataset.__class__.__name__} - Length: {len(loader.dataset)}\n"
