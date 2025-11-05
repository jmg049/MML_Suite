import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import traceback
from typing import Any, Literal, Optional

import numpy as np
import torch
from config.base_config import BaseConfig
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
from experiment_utils.utils import clean_checkpoints, ensure_dir, flatten_dict, prepare_metrics_for_json, safe_extend, get_correct_cmam_dataset_selected_patterns, CONDITION
from fed import DataSplitType
from fed.client import Client, ClientMetricsResult
from fed.simple_convergence import SimpleConvergenceMonitor
from modalities import Modality
from models.cmams import SimpleCMAM
from models.protocols import MultimodalModelProtocol
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from train_multimodal import check_early_stopping

console = get_console()
logger = get_logger()


@dataclass
class GlobalModelData:
    config: FederatedExperimentConfig
    model: MultimodalModelProtocol
    optimizer: Optimizer
    loss_function: LossFunctionGroup
    scheduler: Optional[LRScheduler] = None


@dataclass
class GlobalCMAMData:
    config: BaseConfig
    cmam: SimpleCMAM
    optimizer: Optimizer
    loss_function: LossFunctionGroup
    checkpoint_manager: CheckpointManager
    save_metric: str = "loss"


@dataclass(kw_only=True)
class FederatedTrainer:
    base: GlobalModelData
    dataloaders: dict[DataSplitType, DataLoader]
    metric_recorder: MetricRecorder
    checkpoint_manager: CheckpointManager
    clients: list[Client]
    tracked_modalities: list[str]
    cmams: dict[Modality, GlobalCMAMData] = None
    train_cmams: bool = True
    device: str = "cpu"
    epochs: int = 1
    experiment_data: dict[str, Any] = field(
        default_factory=lambda: {
            "metrics_history": {"train": [], "val": [], "test": []},
            "timing_history": {"train": [], "val": [], "test": []},
        }
    )
    id: int = -1  # Default ID for global trainer, can be overridden
    current_round: int = 0  # Track current federated learning round
    
    # Simple convergence monitoring
    convergence_monitor: SimpleConvergenceMonitor = None
    client_assignments: dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.train_cmams:
            assert self.cmams is not None, "C-MAMs must be provided if training is enabled."
            # Initialize client C-MAMs to match global C-MAMs
            self._initialize_client_cmams()

        assert len(self.clients) > 0, "At least one client must be provided for global training."
        
        # Initialize client assignments for metrics tracking
        for client in self.clients:
            self.client_assignments[client.id] = client.available_modalities
        
        # Initialize convergence monitoring
        self.convergence_monitor = SimpleConvergenceMonitor(
            metrics_dir=self.metrics_fp / "convergence",
            patience=10,  # Can be made configurable
            min_delta=0.001
        )

    def get_cmam_dataloaders(self, cmam: SimpleCMAM) -> dict[DataSplitType, DataLoader]:
        """
        Get pattern-filtered dataloaders for a specific global C-MAM using dynamic patterns.
        
        This uses the new context manager approach to avoid dataset duplication while
        ensuring C-MAMs only receive samples with their required pattern.
        
        Args:
            cmam: The C-MAM model requiring filtered data
            
        Returns:
            Dict of context managers that yield filtered dataloaders for each data split
        """
        from fed.data_utils import create_cmam_context_dataloader
        
        filtered_dataloaders = {}
        for split_type, base_dataloader in self.dataloaders.items():
            # Use the new context manager approach for dynamic pattern switching
            filtered_dataloader = create_cmam_context_dataloader(
                base_dataloader=base_dataloader,
                input_modalities=cmam.input_modalities,
                target_modality=cmam.target_modality,
            )
            filtered_dataloaders[split_type] = filtered_dataloader
        return filtered_dataloaders

    def run(self, round: int, training_phase: str = "base_model"):
        """
        Executes federated training for the specified phase.
        
        Args:
            round: Current round number
            training_phase: Either "base_model", "cmam", or "incongruent" to indicate training phase
        """
        # Set round number for all participants to ensure proper directory structure
        self.set_current_round(round)
        for client in self.clients:
            client.set_current_round(round)
        
        console.rule(f"[bold cyan]FEDERATED ROUND {round} - {training_phase.upper()} PHASE")
        
        if training_phase == "base_model":
            return self._run_base_model_round(round)
        elif training_phase == "cmam":
            if not self.train_cmams:
                print_warning(console, "C-MAM training requested but train_cmams=False")
                return None, [], None
            return self._run_cmam_round(round)
        elif training_phase == "incongruent":
            return self._run_incongruent_round(round)
        else:
            raise ValueError(f"Invalid training_phase: {training_phase}. Use 'base_model', 'cmam', or 'incongruent'.")

    def _run_base_model_round(self, round: int,  ):
        """Execute a single round of base model federated training."""
        # Train global model first
        training_metrics = self._train_global_model(round=round, do_test=False)
        
        self.checkpoint_manager.load_checkpoint(model=self.global_model, load_best=True, round=round, ignore="cmams")
        # Test global model before aggregation (pre-aggregation metrics)
        pre_aggregation_metrics = self._test_global_model(round=round, pre_agg=True)

        self._distribute_base_model_to_clients()

        client_metrics = []
        for client in self.clients:
            # Train the client
            base_result = client.run_base_model_round()
            base_result.client_id = client.id
            client_metrics.append((base_result, None))  # Format as (model_result, cmam_result) tuple

            # Test the client immediately after training
            client.test_client_model_post_round()

        self._aggregate_client_base_model_updates("selective")

        # Test global model after aggregation (post-aggregation metrics)
        post_aggregation_test_metrics = self._test_global_model(round=round, pre_agg=False)
        
        # Collect communication metrics from all clients
        communication_metrics = self._collect_communication_metrics()
        
        # Track convergence with communication data and both pre/post aggregation metrics
        convergence_info = self.convergence_monitor.add_round_metrics(
            round_num=round,
            global_metrics=post_aggregation_test_metrics,
            pre_aggregation_metrics=pre_aggregation_metrics,
            client_metrics=client_metrics,
            communication_metrics=communication_metrics,
            phase="base_model"      
        )
        clean_checkpoints(self.checkpoint_manager.model_dir, round=round, keep_best=True, keep_last=True)
        console.rule("[bold green]BASE MODEL ROUND COMPLETED")
        return pre_aggregation_metrics, client_metrics, post_aggregation_test_metrics

    def _run_cmam_round(self, round: int):
        """Execute a single round of C-MAM federated training."""
        # Train global C-MAMs first
        print_info(console, "Training global C-MAMs")
        print_info(console, f"{self.checkpoint_manager} - {self.checkpoint_manager.model_dir} - {self.global_model}")
        self.checkpoint_manager.load_checkpoint(model=self.global_model, load_best=True, round=round, ignore="cmams")
        print_info(console, "Loading best global model for C-MAM training")

        training_metrics = self._train_cmams()
        
        # Test global C-MAMs before aggregation (pre-aggregation metrics) if not the first round - no point testing a fresh model
        if round != 1:
            pre_aggregation_metrics = self._test_global_cmams(round=round, pre_agg=True)
        else:
            pre_aggregation_metrics = {}
        self._distribute_cmams_to_clients()

        client_metrics = []
        for client in self.clients:
            # Train client C-MAMs
            cmam_metrics = client.run_cmam_round()
            cmam_metrics['client_id'] = client.id
            client_metrics.append((None, cmam_metrics))  # No base model metrics, only C-MAM

            # Test client C-MAMs immediately after training
            if client.cmam_data:  # Only test clients that have C-MAMs
                client.test_client_cmams_post_round()

        self._aggregate_client_cmam_updates()
        # Test global C-MAMs after aggregation (post-aggregation metrics)

        post_aggregation_test_metrics = self._test_global_cmams(round=round, pre_agg=False)
        
        # Collect communication metrics from all clients
        communication_metrics = self._collect_communication_metrics()
        
        # Track C-MAM convergence with both pre/post aggregation metrics
        convergence_info = self.convergence_monitor.add_round_metrics(
            round_num=round,
            global_metrics=post_aggregation_test_metrics,
            pre_aggregation_metrics=pre_aggregation_metrics,
            client_metrics=client_metrics,
            communication_metrics=communication_metrics,
            phase="cmam"
        )

        for cmam in self.cmams.values():
            clean_checkpoints(cmam.checkpoint_manager.model_dir, round=round, keep_best=True, keep_last=True)

        console.rule("[bold green]C-MAM ROUND COMPLETED")
        return pre_aggregation_metrics, client_metrics, post_aggregation_test_metrics

    def _run_incongruent_round(self, round: int):
        """
        Execute a single round of incongruent federated training with intertwined base model and C-MAM training.
        
        Based on algorithm specification:
        - Global model trains on complete data (congruent setup)
        - Complete clients train like congruent setting (base model first, then C-MAMs)
        - Incomplete clients train MM and C-MAM simultaneously
        - Smart C-MAM distribution: only send relevant C-MAMs to each client
        - Mixed aggregation: handle both base model and C-MAM updates
        
        Args:
            round: Current round number
            
        Returns:
            Tuple of (pre_aggregation_metrics, client_metrics, post_aggregation_metrics)
        """
        console.rule("[bold purple]INCONGRUENT FEDERATED TRAINING ROUND")
        
        # === Global Training ===
        # Global model trains on complete data (congruent setup)
        print_info(console, "Training global models on complete data")
        global_training_metrics = self._train_global_model(round=round, do_test=False)

        cmam_epochs = self.config.fed_config.global_cmam_training.get("epochs", 1) if isinstance(self.config.fed_config.global_cmam_training, dict) else self.config.fed_config.global_cmam_training.epochs
        if self.train_cmams and cmam_epochs > 0:
            print_info(console, f"Training global C-MAMs for {cmam_epochs} epochs")
            global_cmam_metrics = self._train_cmams()
        else:
            print_info(console, f"Skipping global C-MAM training (train_cmams={self.train_cmams}, cmam_epochs={cmam_epochs})")
        
        # === Pre-Aggregation Testing ===
        print_info(console, "Testing global models before client distribution")
        pre_aggregation_base_metrics = self._test_global_model(round=round, pre_agg=True)
        pre_aggregation_metrics = {f"base_{k}": v for k, v in pre_aggregation_base_metrics.items()}
        
        if self.train_cmams and cmam_epochs > 0:
            print_info(console, "Testing global C-MAMs before client distribution")
            pre_aggregation_cmam_metrics = self._test_global_cmams(round=round, pre_agg=True) if self.train_cmams else {}
            pre_aggregation_metrics.update({f"cmam_{k}": v for k, v in pre_aggregation_cmam_metrics.items()})

        # === Smart Distribution ===
        # Distribute base model to all clients
        self._distribute_base_model_to_clients()
        # Smart C-MAM distribution: only relevant C-MAMs to each client
        if self.train_cmams and cmam_epochs > 0:
            print_info(console, "Distributing C-MAMs to clients")
            self._distribute_cmams_to_clients_smart()
        else:
            print_info(console, f"Skipping C-MAM distribution (train_cmams={False} - baseline mode or cmam_epochs={cmam_epochs})")
        
        # === Client Training ===
        print_info(console, f"Starting incongruent client training for {len(self.clients)} clients")
        client_metrics = []
        complete_clients = []
        incomplete_clients = []
        
        for client in self.clients:
            print_info(console, f"Client {client.id} starting incongruent round")

            try:
                # Each client determines its own training strategy internally
                client_result = client.run_incongruent_round()
                client_metrics.append((client_result, None))  # Format as (combined_result, None) for compatibility

                # Test the client immediately after training
                if client._is_complete_modality_client():
                    # Complete clients: test both base model and C-MAMs
                    client.test_client_model_post_round()
                    if client.cmam_data and not client.baseline:
                        client.test_client_cmams_post_round()
                    complete_clients.append(client)
                else:
                    # Incomplete clients: test according to their available setup
                    if client.cmam_data and not client.baseline:
                        # If they have C-MAMs and not in baseline mode, they did simultaneous training
                        client.test_client_model_post_round()
                        client.test_client_cmams_post_round()
                    else:
                        # Pure baseline training or baseline mode
                        client.test_client_model_post_round()
                    incomplete_clients.append(client)

                print_success(console, f"Client {client.id} completed incongruent training and testing")

            except Exception as e:
                print_error(console, f"Error in Client {client.id} incongruent training: {e}")
                import traceback
                traceback.print_exc()
                # Add empty result for failed client
                client_metrics.append((None, None))
        
        print_info(console, f"Client training completed: {len(complete_clients)} complete, {len(incomplete_clients)} incomplete")
        
        # === Aggregation ===
        print_info(console, "Aggregating client updates from incongruent training")
        self._aggregate_incongruent_client_updates(complete_clients, incomplete_clients)
        
        # === Post-Aggregation Testing ===
        print_info(console, "Testing global models after aggregation")
        post_aggregation_base_metrics = self._test_global_model(round=round, pre_agg=False)
        post_aggregation_metrics = {f"base_{k}": v for k, v in post_aggregation_base_metrics.items()}

        if self.train_cmams and cmam_epochs > 0:
            print_info(console, "Testing global C-MAMs after aggregation")
            post_aggregation_cmam_metrics = self._test_global_cmams(round=round, pre_agg=False) if self.train_cmams else {}
            post_aggregation_metrics.update({f"cmam_{k}": v for k, v in post_aggregation_cmam_metrics.items()})
        
        # === Communication Metrics ===
        communication_metrics = self._collect_communication_metrics()
        
        # === Convergence Monitoring ===
        # Track convergence for incongruent training
        convergence_info = self.convergence_monitor.add_round_metrics(
            round_num=round,
            global_metrics=post_aggregation_metrics,
            pre_aggregation_metrics=pre_aggregation_metrics,
            client_metrics=client_metrics,
            communication_metrics=communication_metrics,
            phase="incongruent"
        )
        
        console.rule("[bold green]INCONGRUENT ROUND COMPLETED")
        return pre_aggregation_metrics, client_metrics, post_aggregation_metrics

    def _distribute_cmams_to_clients_smart(self) -> None:
        """
        Smart C-MAM distribution: Only send relevant C-MAMs to each client based on their modality configuration.
        
        For incongruent training:
        - Complete clients get all C-MAMs (they can train all of them)
        - Incomplete clients get only C-MAMs for their missing modalities
        """
        console.rule("[bold green]Smart C-MAM Distribution for Incongruent Training")

        if not self.train_cmams or not self.cmams:
            print_warning(console, "No C-MAMs to distribute")
            return

        for client in self.clients:
            if client.cmam_data is None:
                print_info(console, f"Client {client.id} has no C-MAM data structure (expected for train_cmams=False)")
                continue

            # Determine if client is complete or incomplete
            is_complete = client._is_complete_modality_client()
            available_modalities = client._parse_available_modalities()
            
            if is_complete:
                # Complete clients get all C-MAMs (can train all of them)
                print_info(console, f"Client {client.id} is complete - distributing all C-MAMs")
                self._distribute_all_cmams_to_client(client)
            else:
                # Incomplete clients get only relevant C-MAMs for their missing modalities
                missing_modalities = self._get_missing_modalities(available_modalities)
                print_info(console, f"Client {client.id} missing {missing_modalities} - distributing relevant C-MAMs")
                self._distribute_relevant_cmams_to_client(client, available_modalities, missing_modalities)

    def _get_missing_modalities(self, available_modalities: set) -> set:
        """Get modalities that are missing from the available set."""
        from modalities import Modality, add_modality
        add_modality("VIDEO")
        
        all_modalities = {Modality.AUDIO, Modality.VIDEO, Modality.TEXT}
        return all_modalities - available_modalities

    def _distribute_all_cmams_to_client(self, client) -> None:
        """Distribute all C-MAMs to a complete client."""
        for modality_key, cmam_data in self.cmams.items():
            # Find the client's C-MAM that matches this global C-MAM
            matching_client_cmam = self._find_client_cmam_match(client, cmam_data.cmam)
            print_debug(console, f"Distributing C-MAM {modality_key} to Client {client.id}")
            print_debug(console, f"  → Client available modalities: {client.available_modalities}")
            print_debug(console, f"  → Global C-MAM input modalities: {cmam_data.cmam.input_modalities}, target: {cmam_data.cmam.target_modality}")
            print_debug(console, f"Matching client C-MAM: {matching_client_cmam}")
            if matching_client_cmam is not None:
                client.load_cmam_state({
                    "model_state_dict": cmam_data.cmam.state_dict(),
                }, idx=matching_client_cmam)
                client.update_bytes_received_cmams(cmam_data.cmam.parameters_size_bytes)
                print_success(console, f"Distributed C-MAM {modality_key} to complete Client {client.id}")
            else:
                print_warning(console, f"No matching client C-MAM found for global C-MAM {modality_key} on Client {client.id}")

    def _distribute_relevant_cmams_to_client(self, client, available_modalities: set, missing_modalities: set) -> None:
        """Distribute only relevant C-MAMs to an incomplete client."""
        distributed_count = 0
        
        for modality_key, cmam_data in self.cmams.items():
            global_cmam = cmam_data.cmam
            print_debug(console, f"Checking C-MAM {modality_key} for Client {client.id}")
            # Check if this C-MAM is relevant for this client
            cmam_input_modalities = set(global_cmam.input_modalities)
            cmam_target_modality = global_cmam.target_modality
            print_debug(console, f"  → Input modalities: {cmam_input_modalities}, Target: {cmam_target_modality}")
            
            
            # 1. Client has the input modalities needed to train it
            # 2. Client is missing the target modality that this C-MAM reconstructs
            has_input_modalities = cmam_input_modalities.issubset(available_modalities)
            missing_target_modality = cmam_target_modality in missing_modalities
            
            if has_input_modalities and missing_target_modality:
                # This C-MAM is relevant for this client
                matching_client_cmam = self._find_client_cmam_match(client, global_cmam)

                print_debug(console, f"C-MAM {modality_key} relevant for Client {client.id}: has_input={has_input_modalities}, missing_target={missing_target_modality}")
                print_debug(console, f"Matching client C-MAM index: {matching_client_cmam}")
                if matching_client_cmam is not None:
                    client.load_cmam_state({
                        "model_state_dict": cmam_data.cmam.state_dict()
                    },idx=matching_client_cmam)
                    client.update_bytes_received_cmams(cmam_data.cmam.parameters_size_bytes)
                    distributed_count += 1
                    print_success(console, f"Distributed relevant C-MAM {modality_key} to incomplete Client {client.id}")
                    print_info(console, f"  → Input modalities: {cmam_input_modalities}, Target: {cmam_target_modality}")
                else:
                    print_warning(console, f"No matching client C-MAM found for relevant global C-MAM {modality_key} on Client {client.id}")
                    # print_debug(console, f"Client {client.id} has C-MAMs: {[cmam.cmam.input_modalities for cmam in client.cmam_data]}")
                    # print_debug(console, f"Global C-MAM input modalities: {cmam_input_modalities}, target: {cmam_target_modality}")
                    # raise ValueError(f"Client {client.id} has no matching C-MAM for global C-MAM {modality_key} with input {cmam_input_modalities} and target {cmam_target_modality}")
            else:
                # This C-MAM is not relevant for this client
                print_debug(console, f"C-MAM {modality_key} not relevant for Client {client.id}: has_input={has_input_modalities}, missing_target={missing_target_modality}")
        
        if distributed_count == 0:
            print_warning(console, f"No relevant C-MAMs distributed to incomplete Client {client.id}")
            raise ValueError(f"No relevant C-MAMs found for incomplete Client {client.id} with available modalities {available_modalities} and missing modalities {missing_modalities}")
        else:
            print_info(console, f"Distributed {distributed_count} relevant C-MAM(s) to incomplete Client {client.id}")

    def _find_client_cmam_match(self, client, global_cmam) -> Optional[int]:
        """Find the index of client C-MAM that matches the global C-MAM configuration."""
        if not client.cmam_data:
            return None
            
        for i, client_cmam_data in enumerate(client.cmam_data):
            client_cmam = client_cmam_data.cmam
            
            # Check if input and target modalities match
            if (set(client_cmam.input_modalities) == set(global_cmam.input_modalities) and
                client_cmam.target_modality == global_cmam.target_modality):
                print_debug(console, f"Found matching C-MAM at index {i} for Client {client.id}")
                print_debug(console, f"  → Client C-MAM input modalities: {client_cmam.input_modalities}, target: {client_cmam.target_modality}")
                print_debug(console, f"  → Global C-MAM input modalities: {global_cmam.input_modalities}, target: {global_cmam.target_modality}")
                return i
        
        # No matching C-MAM found
        print_info(console, f"No matching C-MAM found for global C-MAM with input {global_cmam.input_modalities} and target {global_cmam.target_modality} on Client {client.id}")
        print_debug(console, f"Client {client.id} has C-MAMs: {[cmam.cmam.input_modalities for cmam in client.cmam_data]}")
        print_debug(console, f"Global C-MAM input modalities: {global_cmam.input_modalities}, target: {global_cmam.target_modality}")
        return None

    def _aggregate_incongruent_client_updates(self, complete_clients: list, incomplete_clients: list) -> None:
        """
        Aggregate client updates from incongruent training with different strategies for complete vs incomplete clients.
        
        Args:
            complete_clients: Clients that have all modalities (trained congruently)  
            incomplete_clients: Clients that are missing modalities (trained incongruently)
        """
        console.rule("[bold blue]Aggregating Incongruent Client Updates")
        
        print_info(console, f"Aggregating updates from {len(complete_clients)} complete + {len(incomplete_clients)} incomplete clients")
        
        # === Base Model Aggregation ===
        # Use selective aggregation for base model (modality-aware)
        print_info(console, "Aggregating base model updates with selective approach")
        self._aggregate_client_base_model_updates(method="selective")
        
        # === C-MAM Aggregation ===
        if self.train_cmams:
            # Use weighted aggregation for C-MAMs (complete clients get higher weights)
            print_info(console, "Aggregating C-MAM updates with weighting (complete clients prioritized)")
            self._aggregate_client_cmam_updates(method="weighted")
            
            # Additional weighted aggregation based on client types
            self._apply_client_type_weighting(complete_clients, incomplete_clients)
        else:
            print_info(console, "Skipping C-MAM aggregation (train_cmams=False - baseline mode)")
        
        print_success(console, "Incongruent client aggregation completed")

    def _apply_client_type_weighting(self, complete_clients: list, incomplete_clients: list) -> None:
        """
        Apply additional weighting based on client type (complete vs incomplete).
        Complete clients get higher influence due to having ground truth for all modalities.
        """
        if not complete_clients:
            print_info(console, "No complete clients available for additional weighting")
            return
            
        total_clients = len(complete_clients) + len(incomplete_clients)
        complete_ratio = len(complete_clients) / total_clients
        incomplete_ratio = len(incomplete_clients) / total_clients
        
        print_info(console, f"Client type distribution: {complete_ratio:.2%} complete, {incomplete_ratio:.2%} incomplete")
        
        # The weighted aggregation in _aggregate_client_cmam_updates already handles this
        # through the _calculate_cmam_client_weights method which gives higher weights to complete clients
        print_info(console, "Client type weighting applied through existing weighted aggregation mechanism")

    def _train_global_model(self, round: int, do_test: bool = True) -> None | ClientMetricsResult:
        console.rule("[bold cyan]Training Global Base Model")
        console.start_task("Global Model -- Epoch", total=self.epochs, style="blue")
        best_metrics = None
        wait = 0

        self.global_model.to(self.device)
        self.global_model.train()  # Ensure model is in training mode
        for param in self.global_model.parameters():
            param.requires_grad_(True)
        self.metric_recorder.reset()
        print_info(console, f"=== Round {round} ===")
        print_info(console, f"Model training mode: {self.global_model.training}")
        print_info(console, f"Device: {next(self.global_model.parameters()).device}")

        for name, param in self.global_model.named_parameters():
            if not param.requires_grad:
                print(f"[WARN] Param {name} requires_grad = False")
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # === Training ===
            self.global_model.train()

            train_metrics = self._run_train_epoch(epoch)
            self.write_metrics(train_metrics, epoch, DataSplitType.TRAIN)
            self.experiment_data["metrics_history"]["train"].append(train_metrics.copy())
            self.experiment_data["timing_history"]["train"].append(train_metrics["train_time_secs"])
            print_metric_summary(console, epoch, train_metrics, "train")
            log_epoch_time(console, logger, self.id, epoch, train_metrics["train_time_secs"])

            # === Validation ===
            self.global_model.eval()

            val_metrics = self._run_validation_epoch(epoch)
            self.write_metrics(val_metrics, epoch, DataSplitType.VALIDATION)
            self.experiment_data["metrics_history"]["val"].append(val_metrics.copy())
            self.experiment_data["timing_history"]["val"].append(val_metrics["validation_time_secs"])
            print_metric_summary(console, epoch, val_metrics, "val")
            log_epoch_time(console, logger, self.id, epoch, val_metrics["validation_time_secs"])

            # === Checkpointing & Early Stopping ===
            is_best, should_continue, wait = check_early_stopping(
                val_metrics=val_metrics,
                best_metrics=best_metrics,
                patience=self.config.fed_config.global_training.early_stopping_patience,
                min_delta=self.config.fed_config.global_training.early_stopping_min_delta,
                wait=wait,
                mode="minimize"
                if "loss" in self.config.fed_config.global_training.early_stopping_metric
                else "maximize",
                target_metric=self.config.fed_config.global_training.early_stopping_metric,
            )

            if is_best:
                best_metrics = val_metrics.copy()
                self.checkpoint_manager.save_checkpoint(
                    model=self.global_model,
                    optimizer=self.gm_optimizer,
                    scheduler=self.gm_scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=True,
                    round=round
                )
                print_success(console, f">> New best model saved at epoch {epoch}")

            if self.do_early_stopping and not should_continue:
                print_warning(console, f"Early stopping triggered at epoch {epoch}")
                break

            if self.gm_scheduler:
                target = val_metrics[self.config.fed_config.global_training.early_stopping_metric]
                if isinstance(self.gm_scheduler, ReduceLROnPlateau):
                    self.gm_scheduler.step(target)
                else:
                    self.gm_scheduler.step()
                print_info(console, f"Learning rate: {self.gm_optimizer.param_groups[0]['lr']:.2e}")

            log_epoch_time(console, logger, self.id, epoch, time.time() - epoch_start)

        console.complete_task("Global Model -- Epoch")

        # Final testing phase
        test_metrics = self._test_global_model(round, pre_agg=True) if do_test else {}
        result = ClientMetricsResult.from_dict(
            {
                DataSplitType.TRAIN: train_metrics,
                DataSplitType.VALIDATION: val_metrics,
                DataSplitType.TEST: test_metrics,
            },
            client_id=-1,  # -1 to indicate global model
        )

        self.global_model.to("cpu")  # Move model back to CPU after training to save memory

        log_total_time(console, logger, self.id, train_metrics["train_time_secs"])
        return result
    
    def get_dataloaders(self, cmam: SimpleCMAM) -> dict[DataSplitType, DataLoader]:
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
            )

            filtered_dataloaders[split_type] = filtered_dataloader

        return filtered_dataloaders
    
    def _run_train_epoch(self, epoch: int) -> dict[str, Any]:
        self.metric_recorder.reset()
        losses = defaultdict(list)
        timings = []

        console.start_task("Training", total=len(self.train_dataloader), style="green")

        for batch in self.train_dataloader:
            start = time.time()
            output = self.global_model.train_step(
                batch,
                optimizer=self.gm_optimizer,
                loss_functions=self.gm_loss_function,
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
        per_sample = np.mean(timings) / self.train_dataloader.batch_size

        metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=epoch, loss=avg_loss))
        metrics.update(
            {
                "loss": avg_loss,
                "train_time_secs": duration,
                "per_sample_time": per_sample,
            }
        )

        return metrics

    def _run_validation_epoch(self, epoch: int) -> dict[str, Any]:
        self.metric_recorder.reset()
        losses = defaultdict(list)
        timings = []

        console.start_task("Validation", total=len(self.val_dataloader), style="yellow")

        targets, preds, logits = [], [], []
        for batch in self.val_dataloader:
            start = time.time()
            output = self.global_model.validation_step(
                batch,
                loss_functions=self.gm_loss_function,
                device=self.device,
                metric_recorder=self.metric_recorder,
                epoch=epoch,
            )
            timings.append(time.time() - start)

            losses["loss"].append(output["loss"])
            for k, v in output.get("other_losses", {}).items():
                losses[k].append(v.item())

            safe_extend(output, targets, "targets")
            safe_extend(output, preds, "preds")
            safe_extend(output, logits, "logits")

            console.update_task("Validation", advance=1)

        console.complete_task("Validation")

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
        return metrics

    def _test_global_model(self, round: int, pre_agg: bool = True) -> dict[str, Any]:
        console.start_task("Global -- Testing", total=len(self.test_dataloader), style="purple")
        self.metric_recorder.reset()
        losses = defaultdict(list)

        targets, preds, logits = [], [], []
        ground_truth_logits, reconstructed_logits = [], []
        ground_truth_embds, reconstructed_embds = defaultdict(list), defaultdict(list)
        ids, miss_types = [], []

        self.global_model.to(self.device)

        for batch in self.test_dataloader:
            output = self.global_model.validation_step(
                batch,
                loss_functions=self.gm_loss_function,
                device=self.device,
                metric_recorder=self.metric_recorder,
            )
            for key in ["miss_type", "sample_ids"]:
                if key not in output:
                    raise KeyError(f"Key '{key}' not found in test output. Available keys: {list(output.keys())}")

            safe_extend(output, targets, "targets", error=True)
            safe_extend(output, preds, "preds", error=True)
            safe_extend(output, logits, "logits", error=True)
            safe_extend(output, ground_truth_logits, "ground_truth_logits", error=True)

            for modality in output["ground_truth_embeddings"]:
                safe_extend(
                    output["ground_truth_embeddings"],
                    ground_truth_embds[modality],
                    modality,
                    error=True,
                )

            safe_extend(output, ids, "sample_ids", error=True)
            safe_extend(output, miss_types, "miss_type", error=True)
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
            console.update_task("Global -- Testing", advance=1)
        avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
        test_metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=None, loss=avg_loss))
        test_metrics["loss"] = avg_loss

        if len(reconstructed_logits) == 0:
            reconstructed_logits = None
            reconstructed_embds = None

        self.write_metrics(
            test_metrics, epoch=f"{round}_{'pre_agg' if pre_agg else 'post_agg'}", split=DataSplitType.TEST
        )
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

        print_success(console, f"Test metrics written for Client {self.id}")
        console.complete_task("Global -- Testing")
        self.global_model.to("cpu")  # Move model back to CPU after testing to save memory

        return test_metrics

    def _test_global_cmams(self, round: int, pre_agg: bool = True) -> dict[str, Any]:
        """
        Test all global C-MAMs after aggregation, similar to _test_global_model.
        
        Args:
            round: Current round number
            pre_agg: Whether this is pre-aggregation (True) or post-aggregation (False) testing
            
        Returns:
            dict containing combined test metrics from all C-MAMs
        """
        if not self.cmams:
            print_warning(console, "No global C-MAMs to test")
            return {}
        
        console.rule("[bold purple]Testing Global C-MAMs")
        combined_metrics = {}
        
        # Move base model to device for C-MAM testing (C-MAMs need trained model)
        self.global_model.to(self.device)
        

        for modality_key, cmam_data in self.cmams.items():
            console.start_task(f"Global C-MAM {modality_key} -- Testing", total=len(self.test_dataloader), style="purple")
            
            cmam = cmam_data.cmam
            cmam_data.checkpoint_manager.load_checkpoint(
                model=cmam, load_best=True, round=round
            )
            loss_functions = cmam_data.loss_function
            
            # Move C-MAM to device
            cmam.to(self.device)
            cmam.eval()
            
            self.metric_recorder.reset()
            losses = defaultdict(list)
            
            # Get pattern-filtered dataloader for this specific C-MAM using context manager
            cmam_dataloaders = self.get_cmam_dataloaders(cmam)
            
            with cmam_dataloaders[DataSplitType.TEST] as test_dataloader:
                targets, preds, logits = [], [], []
                ground_truth_logits, reconstructed_logits = [], []
                ground_truth_embds, reconstructed_embds = [], []
                ids, miss_types = [], []
                
                with torch.no_grad():
                    for batch in test_dataloader:
                        try:
                            output = cmam.validation_step(
                                batch=batch,
                                loss_functions=loss_functions,
                                device=torch.device(self.device),
                                metric_recorder=self.metric_recorder,
                                trained_model=self.global_model,
                            )
                            
                            # Extract required keys with error checking
                            for key in ["miss_type", "sample_ids"]:
                                if key not in output:
                                    raise KeyError(f"Key '{key}' not found in C-MAM test output. Available keys: {list(output.keys())}")

                            safe_extend(output, targets, "targets", error=True)
                            safe_extend(output, preds, "preds", error=True)
                            safe_extend(output, logits, "logits", error=True)
                            safe_extend(output, ground_truth_logits, "ground_truth_logits", error=True)

                            # Handle embeddings
                            if "ground_truth_embeddings" in output:
                                for modality in output["ground_truth_embeddings"]:
                                    safe_extend(
                                        output,
                                        ground_truth_embds,
                                        "ground_truth_embeddings",
                                        error=True,
                                    )

                            safe_extend(output, ids, "sample_ids", error=True)
                            safe_extend(output, miss_types, "miss_type", error=True)
                            
                            if "reconstructed_logits" in output:
                                safe_extend(output, reconstructed_logits, "reconstructed_logits")
                                if "reconstructed_embeddings" in output:
                                    safe_extend(
                                        output,
                                        reconstructed_embds,
                                        "reconstructed_embeddings",
                                        error=True,
                                    )

                            losses["loss"].append(output["loss"])
                            for k, v in output.get("other_losses", {}).items():
                                losses[k].append(v.item() if hasattr(v, 'item') else v)
                                
                        except Exception as e:
                            err_traceback = traceback.format_exc()
                            print_error(console, f"Error testing global C-MAM {modality_key}: {e}\n{err_traceback}")
                            raise e
                        
                        console.update_task(f"Global C-MAM {modality_key} -- Testing", advance=1)
            
            console.complete_task(f"Global C-MAM {modality_key} -- Testing")
            
            # Calculate metrics for this C-MAM
            avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
            test_metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=None, loss=avg_loss))
            test_metrics["loss"] = avg_loss
            
            # Add other losses to metrics
            for key, values in losses.items():
                if key != "loss":
                    test_metrics[key] = np.mean(values)

            if len(reconstructed_logits) == 0:
                reconstructed_logits = None
                reconstructed_embds = None

            print_metric_summary(console, round, test_metrics, "test")

            # Write metrics with C-MAM specific naming
            epoch_suffix = f"{round}_{'pre_agg' if pre_agg else 'post_agg'}"
            name = "-".join([str(m) for m in cmam.input_modalities])
            name = f"{name}_to_{str(cmam.target_modality)}"
            self.write_cmam_metrics(
                test_metrics, 
                epoch=epoch_suffix, 
                split=DataSplitType.TEST, 
                modality=f"{name}", 
                target_modality=str(cmam.target_modality)
            )
            
            # Write logits and embeddings for this C-MAM
            self._write_cmam_logits_and_embeddings(
                targets=targets,
                preds=preds,
                logits=logits,
                miss_types=miss_types,
                ids=ids,
                ground_truth_logits=ground_truth_logits,
                reconstructed_logits=reconstructed_logits,
                ground_truth_embds=ground_truth_embds,
                reconstructed_embds=reconstructed_embds,
                modality_key=modality_key,
                cmam=cmam
            )

            # Add to combined metrics with C-MAM prefix
            for metric_name, value in test_metrics.items():
                combined_metrics[f"cmam_{modality_key}_{metric_name}"] = value

            print_success(console, f"Test metrics written for Global C-MAM {modality_key}")
            
            # Move C-MAM back to CPU to save memory
            cmam.to("cpu")

        # Move base model back to CPU
        self.global_model.to("cpu")
        
        console.rule("[bold green]Global C-MAMs Testing Complete")
        return combined_metrics

    def _write_cmam_logits_and_embeddings(
        self,
        targets: list[Any],
        preds: list[Any],
        logits: list[Any],
        miss_types: list[Any],
        ids: list[Any],
        ground_truth_logits: list[Any],
        ground_truth_embds: list[Any],
        modality_key: str,
        cmam: SimpleCMAM,
        reconstructed_logits: Optional[list[Any]] = None,
        reconstructed_embds: list[Any] = None,
    ) -> None:
        """
        Write C-MAM logits and embeddings to a npz archive for future analysis.
        
        Args:
            targets: Target labels
            preds: Predictions
            logits: Model logits
            miss_types: Missing modality types
            ids: Sample IDs
            ground_truth_logits: Ground truth logits
            ground_truth_embds: Ground truth embeddings
            modality_key: Key identifying the C-MAM modality
            cmam: The C-MAM model being tested
            reconstructed_logits: Reconstructed logits (optional)
            reconstructed_embds: Reconstructed embeddings (optional)
        """
        if not targets:
            print_warning(console, f"No data to write for Global C-MAM {modality_key}")
            return

        targets = np.array(targets)
        preds = np.array(preds)
        logits = np.array(logits)
        ground_truth_logits = np.array(ground_truth_logits)
        ids = np.array(ids)
        miss_types = np.array(miss_types)

        if reconstructed_logits is not None:
            reconstructed_logits = np.array(reconstructed_logits)

        # Create entry for this C-MAM
        entry = {
            "targets": targets,
            "preds": preds,
            "logits": logits,
            "ids": ids,
            "ground_truth_logits": ground_truth_logits,
            "miss_types": miss_types,
            "input_modalities": [str(m) for m in cmam.input_modalities],
            "target_modality": str(cmam.target_modality),
        }

        # Add ground truth embeddings
        # for modality, embeddings in ground_truth_embds.items():
            # if embeddings:
                # entry[f"ground_truth_embd_{modality}"] = np.array(embeddings)

        ground_truth_embds = np.array(ground_truth_embds)
        entry["ground_truth_embeddings"] = ground_truth_embds

        # Add reconstructed data if available
        if reconstructed_logits is not None:
            entry["reconstructed_logits"] = np.array(reconstructed_logits)

        if reconstructed_embds is not None:
            entry["reconstructed_embeddings"] = np.array(reconstructed_embds)

        # Save to C-MAM specific file
        output_path = self.metrics_fp / "cmams" / f"global_cmam_{modality_key}_logits_embeddings.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            np.savez_compressed(output_path, **entry)
            print_success(console, f"C-MAM {modality_key} logits and embeddings saved to {output_path}")
        except Exception as e:
            print_error(console, f"Failed to save C-MAM {modality_key} logits and embeddings: {e}")
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
        ground_truth_logits = np.array(ground_truth_logits)
        reconstructed_logits = np.array(reconstructed_logits)
        ids = np.array(ids)

        tracking: dict[str, list[dict[str, np.ndarray]]] = {modality: [] for modality in self.tracked_modalities}
        miss_types = np.array(miss_types)

        print_info(console, f"Ground Truth Embeddings: {len(ground_truth_embds)} - {list(ground_truth_embds.keys())}")

        # Group entries by missing modality type
        for modality in self.tracked_modalities:
            mask = miss_types == modality
            if not np.any(mask):
                print_warning(console, f"No data found for modality '{modality}' in test batch.")
                continue

            entry = {
                "targets": targets[mask],
                "preds": preds[mask],
                "logits": logits[mask],
                "ids": ids[mask],
                "ground_truth_logits": ground_truth_logits[mask],
            }

            for m, m_embd in ground_truth_embds.items():
                if m == modality:
                    masked = ground_truth_embds[m][mask]
                    print_info(console, f"Masked ground truth embeddings for {modality}: {masked}")
                    entry[f"ground_truth_embd_{m}"] = masked
            if reconstructed_embds is not None:
                for m_embd in reconstructed_embds:
                    masked = reconstructed_embds[modality]
                    print_info(console, f"{masked}")
                    masked = masked[mask]
                    entry[f"reconstructed_embd_{modality}"] = masked
            if reconstructed_logits:
                entry["reconstructed_logits"] = reconstructed_logits[mask]

            tracking[modality].append(entry)

        # Final packaging: convert list of dicts into numpy array of dicts
        packaged = {
            modality: np.array(modality_entries) for modality, modality_entries in tracking.items() if modality_entries
        }

        if not packaged:
            print_error(console, "No valid entries to write for Global Model.")
            return

        output_path = self.metrics_fp / "global_logits_embeddings.npz"
        try:
            np.savez_compressed(output_path, **packaged)
            print_success(console, f"Logits and embeddings saved for Global Model at {output_path}")
        except Exception as e:
            print_error(console, f"Failed to save logits and embeddings: {e}")
            raise

    def write_metrics(self, metrics: dict[str, Any], epoch: Optional[int], split: DataSplitType) -> None:
        if epoch is None:
            epoch = "test_metrics"

        metrics = prepare_metrics_for_json([metrics])[0]

        metrics_fp = self.metrics_fp
        metrics_fp_split = metrics_fp / split.value
        metrics_fp_split.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_fp_split / f"{epoch}.json"

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        print_info(console, f"Metrics for Global Model at epoch {epoch} saved to {metrics_file}")

    def _train_cmams(self) -> None:
        console.rule("[bold magenta]Training Global C-MAMs")
        
        if not self.cmams:
            print_warning(console, "No global C-MAMs configured, skipping C-MAM training")
            return

        self._train_global_cmams_with_data()
        # Collect and aggregate client C-MAM updates from previous round
        cmam_updates = self._collect_cmam_updates()
        # print_info(console, f"Collected C-MAM updates from clients: {len(cmam_updates)} modalities")
        # print_info(console, f"Collected C-MAM updates for {cmam_updates.keys()}")
        if cmam_updates:
            aggregated_updates = self._aggregate_cmam_updates(cmam_updates)
            self._apply_cmam_updates(aggregated_updates)

    def _collect_cmam_updates(self) -> dict[Modality, list[dict[str, Any]]]:
        """
        Collect C-MAM parameters from all clients that have trained C-MAMs.
        
        Returns:
            dict mapping target modalities to lists of client C-MAM state dicts
        """
        cmam_updates = defaultdict(list)
        
        for client in self.clients:
            for cmam_data in client.cmam_data:
                cmam = cmam_data.cmam
                # target_modality = cmam.target_modality
                inputs = "".join([str(m)[0] for m in cmam.input_modalities]).lower()

                cmam_state = cmam.state_dict()
                if cmam_state is not None:
                    cmam_updates[inputs].append(cmam_state)
                    print_info(console, f"Collected C-MAM update from Client {client.id} for {inputs}")
        
        return dict(cmam_updates)

    def _aggregate_cmam_updates(self, cmam_updates: dict[Modality, list[dict[str, Any]]]) -> dict[Modality, dict[str, Any]]:
        """
        Aggregate C-MAM updates using federated averaging.
        
        Args:
            cmam_updates: dict mapping modalities to lists of client C-MAM state dicts
            
        Returns:
            dict mapping modalities to aggregated C-MAM state dicts
        """
        aggregated_updates = {}
        
        for modality, client_states in cmam_updates.items():
            if not client_states:
                continue
                
            # Federated averaging of parameters
            aggregated_state = {}
            
            # Get parameter names from first client
            param_names = client_states[0].keys()
            
            for param_name in param_names:
                # Stack all client parameters for this parameter name
                param_tensors = [state[param_name] for state in client_states]
                # Average across clients
                aggregated_param = torch.stack(param_tensors).float().mean(dim=0)
                aggregated_state[param_name] = aggregated_param
            
            aggregated_updates[modality] = aggregated_state
            print_info(console, f"Aggregated C-MAM updates for {modality} from {len(client_states)} clients")
        
        return aggregated_updates

    def _apply_cmam_updates(self, aggregated_updates: dict[Modality, dict[str, Any]]) -> None:
        """
        Apply aggregated C-MAM updates to global C-MAM models.
        
        Args:
            aggregated_updates: dict mapping modalities to aggregated state dicts
        """
        for modality, aggregated_state in aggregated_updates.items():
            modality_key = "".join(sorted([str(m)[0] for m in modality])).lower()

            if modality_key in self.cmams:
                self.cmams[modality_key].cmam.load_state_dict(aggregated_state)
                print_success(console, f"Updated global C-MAM for {modality_key}")
            else:
                print_error(console, f"No global C-MAM found for modality {modality}, skipping update")
                print_error(console, f"Available C-MAMs: {list(self.cmams.keys())}")
                raise ValueError(f"No global C-MAM found for modality {modality_key}")



    def _train_global_cmams_with_data(self) -> None:
        """
        Perform actual C-MAM training using global server data.
        This includes forward/backward passes with train_step and validation_step.
        Each C-MAM trains for its configured number of epochs with metrics and early stopping.
        """
        if DataSplitType.TRAIN not in self.dataloaders:
            print_warning(console, "No training data available for global C-MAM training")
            return
        
        print_info(console, f"Training {len(self.cmams)} global C-MAMs")

        for modality, cmam_data in self.cmams.items():
            print_info(console, f"Training global C-MAM for input modality: {modality} - target modality: {cmam_data.cmam.target_modality}")
            
            self._train_single_global_cmam(str(modality), cmam_data)

    def _train_single_global_cmam(self, modality: str, cmam_data: GlobalCMAMData) -> None:
        """
        Train a single global C-MAM following the centralized C-MAM training pattern.
        
        Args:
            modality: Target modality for the C-MAM
            cmam_data: Global C-MAM data containing model, optimizer, config, etc.
        """
        cmam = cmam_data.cmam
        optimizer = cmam_data.optimizer
        loss_functions = cmam_data.loss_function
        inputs = "-".join([str(m) for m in cmam.input_modalities]).lower()
        modality = f"{inputs}_{str(cmam.target_modality).lower()}"

        
        # Get training configuration - check for epochs in config
        epochs = self.config.fed_config.global_cmam_training.get("epochs", 1) if isinstance(self.config.fed_config.global_cmam_training, dict) else self.config.fed_config.global_cmam_training.epochs
        print_info(console, f"Training C-MAM {modality} for {epochs} epochs")
        # Set models to appropriate device
        cmam.to(self.device)
        self.global_model.to(self.device)
        
        # Initialize training state
        best_metrics = None
        wait = 0
        
        console.start_task(f"C-MAM {modality} -- Epoch", total=epochs, style="cyan")
        
        old_patterns = self.val_dataloader.dataset.patterns
        # Training loop for multiple epochs
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # === Training Phase ===
            cmam.train()
            train_metrics = self._run_cmam_train_epoch(cmam, loss_functions, optimizer, epoch, modality)
            
            # Save training metrics
            self.write_cmam_metrics(train_metrics, epoch, DataSplitType.TRAIN, modality, cmam_data.cmam.target_modality)
            print_metric_summary(console, epoch, train_metrics, f"C-MAM {modality} train")
            log_epoch_time(console, logger, f"CMAM-{modality}", epoch, train_metrics["train_time_secs"])
            
            # === Validation Phase ===
            cmam.eval()
            val_metrics = self._run_cmam_validation_epoch(cmam, loss_functions, epoch, modality)
            
            # Save validation metrics
            self.write_cmam_metrics(metrics=val_metrics, epoch=epoch, split=DataSplitType.VALIDATION, modality=modality, target_modality=cmam_data.cmam.target_modality)
            print_metric_summary(console, epoch, val_metrics, f"C-MAM {modality} val", skip_conditions=[str(cmam_data.cmam.target_modality).upper()[0]])
            log_epoch_time(console, logger, f"CMAM-{modality}", epoch, val_metrics["validation_time_secs"])
            

            # === Early Stopping & Checkpointing ===
            is_best, should_continue, wait = check_early_stopping(
                val_metrics=val_metrics,
                best_metrics=best_metrics,
                patience=self.config.fed_config.global_cmam_training["early_stopping_patience"],
                min_delta= self.config.fed_config.global_cmam_training.get("early_stopping_min_delta", 0.01),
                wait=wait,
                mode="minimize" if "loss" in cmam_data.save_metric else "maximize",
            )
            
            if is_best:
                best_metrics = val_metrics.copy()
                # Save best C-MAM checkpoint
                cmam_data.checkpoint_manager.save_checkpoint(
                    model=cmam,
                    optimizer=optimizer,
                    scheduler=None,  # C-MAMs typically don't use schedulers
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=True,
                    round=self.current_round,
                )
                print_success(console, f">> New best C-MAM {modality} saved at epoch {epoch}")
            
            if  self.config.fed_config.global_cmam_training["early_stopping"] and not should_continue:
                print_warning(console, f"Early stopping triggered for C-MAM {modality} at epoch {epoch}")
                break
        
            log_epoch_time(console, logger, f"CMAM-{modality}", epoch, time.time() - epoch_start)
        
        self.val_dataloader.dataset.patterns = old_patterns  # Restore original patterns
        console.complete_task(f"C-MAM {modality} -- Epoch")
        
        # Move back to CPU to save memory
        cmam.to("cpu")
        if epochs > 0:
            print_success(console, f"✓ Global C-MAM {modality} training completed")
        else:
            print_success(console, f"✓ Global C-MAM {modality} training skipped successfully! (0 epochs)")
    
    def _run_cmam_train_epoch(self, cmam: SimpleCMAM, loss_functions: LossFunctionGroup, 
                             optimizer: Optimizer, epoch: int, modality: Modality) -> dict[str, Any]:
        """
        Run one training epoch for a specific C-MAM.
        
        Args:
            cmam: The C-MAM model to train
            loss_functions: Loss function group for training
            optimizer: Optimizer for the C-MAM
            epoch: Current epoch number
            modality: Target modality for logging
            
        Returns:
            dict containing training metrics
        """
        self.metric_recorder.reset()
        losses = defaultdict(list)
        timings = []
        
        # Get pattern-filtered dataloader for this specific C-MAM using context manager
        cmam_dataloaders = self.get_cmam_dataloaders(cmam)
        
        with cmam_dataloaders[DataSplitType.TRAIN] as train_dataloader:
            console.start_task(f"C-MAM {modality} Training", total=len(train_dataloader), style="green")
            
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
                        trained_model=self.global_model,
                    )
                    
                    loss = train_output["loss"]
                    losses["loss"].append(loss)
                    
                    # Record other losses if present
                    other_losses = train_output.get("losses", {})
                    for key, value in other_losses.items():
                        losses[key].append(value.item() if hasattr(value, 'item') else value)
                        
                except Exception as e:
                    err_traceback = traceback.format_exc()
                    print_error(console, f"Error training C-MAM for {modality}: {e}\n{err_traceback}")
                    raise e
                
                timings.append(time.time() - start)
            console.update_task(f"C-MAM {modality} Training", advance=1)
        
        console.complete_task(f"C-MAM {modality} Training")
        
        # Calculate metrics
        avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
        duration = np.sum(timings)
        per_sample = np.mean(timings) / train_dataloader.batch_size
        
        metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=epoch, loss=avg_loss))
        metrics.update({
            "loss": avg_loss,
            "train_time_secs": duration,
            "per_sample_time": per_sample,
        })
        
        # Add other losses to metrics
        for key, values in losses.items():
            if key != "loss":
                metrics[key] = np.mean(values)
        
        return metrics

    def _run_cmam_validation_epoch(self, cmam: SimpleCMAM, loss_functions: LossFunctionGroup,
                                  epoch: int, modality: Modality) -> dict[str, Any]:
        """
        Run one validation epoch for a specific C-MAM.
        
        Args:
            cmam: The C-MAM model to validate
            loss_functions: Loss function group for validation
            epoch: Current epoch number
            modality: Target modality for logging
            
        Returns:
            dict containing validation metrics
        """
        self.metric_recorder.reset()
        losses = defaultdict(list)
        timings = []
        
        # Get pattern-filtered dataloader for this specific C-MAM using context manager
        cmam_dataloaders = self.get_cmam_dataloaders(cmam)
        
        with cmam_dataloaders[DataSplitType.VALIDATION] as val_dataloader:
            console.start_task(f"C-MAM {modality} Validation", total=len(val_dataloader), style="yellow")
            
            with torch.no_grad():
                for batch in val_dataloader:
                    start = time.time()
                    try:
                        val_output = cmam.validation_step(
                            batch=batch,
                            loss_functions=loss_functions,
                            device=torch.device(self.device),
                            metric_recorder=self.metric_recorder,
                            trained_model=self.global_model,
                        )
                        
                        loss = val_output["loss"]
                        losses["loss"].append(loss)
                        
                        # Record other losses if present
                        other_losses = val_output.get("losses", {})
                        for key, value in other_losses.items():
                            losses[key].append(value.item() if hasattr(value, 'item') else value)
                            
                    except Exception as e:
                        err_traceback = traceback.format_exc()
                        print_error(console, f"Error validating C-MAM for {modality}: {e}\n{err_traceback}")
                        raise e
                    
                    timings.append(time.time() - start)
                    console.update_task(f"C-MAM {modality} Validation", advance=1)
        
        console.complete_task(f"C-MAM {modality} Validation")
        
        # Calculate metrics
        avg_loss = np.mean(losses["loss"]) if losses["loss"] else 0.0
        duration = np.sum(timings)
        per_sample = np.mean(timings) / val_dataloader.batch_size
        
        metrics = flatten_dict(self.metric_recorder.calculate_all_groups(epoch=epoch, loss=avg_loss))
        metrics.update({
            "loss": avg_loss,
            "validation_time_secs": duration,
            "per_sample_time": per_sample,
        })
        
        # Add other losses to metrics
        for key, values in losses.items():
            if key != "loss":
                metrics[key] = np.mean(values)
        
        return metrics

    def write_cmam_metrics(self, metrics: dict[str, Any], epoch: int, split: DataSplitType, modality: str, target_modality: str) -> None:
        """
        Write C-MAM metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            epoch: Current epoch number
            split: Data split type (train/validation/test)
            modality: Target modality for the C-MAM
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
        
        print_info(console, f"C-MAM {modality} metrics for epoch {epoch} saved to {metrics_file}")

    def _initialize_client_cmams(self) -> None:
        """
        Initialize clients with C-MAM data that matches the global C-MAMs.
        Each client will get C-MAMs for the target modalities they need.
        """
        console.rule("[bold yellow]Initializing Client C-MAMs")
        
        if not self.cmams:
            print_warning(console, "No global C-MAMs to initialize on clients")
            return
        
        for client in self.clients:
            # Skip clients that already have C-MAMs configured
            if client.cmam_data is not None and len(client.cmam_data) > 0:
                print_info(console, f"Client {client.id} already has {len(client.cmam_data)} C-MAM(s) configured, skipping initialization")
                continue
                
            print_info(console, f"Initializing C-MAMs for Client {client.id}")
            
            # Initialize empty C-MAM data list if None
            if client.cmam_data is None:
                client.cmam_data = []
            
            # For now, give each client the first global C-MAM
            # TODO: This should be smarter - assign based on client's modality configuration
            if len(self.cmams) > 0:
                # Get the first global C-MAM modality
                target_modality = next(iter(self.cmams.keys()))
                global_cmam_data = self.cmams[target_modality]
                
                # Create a copy of the global C-MAM for the client
                client_cmam = self._create_client_cmam_copy(global_cmam_data, target_modality)
                
                # Create optimizer for client C-MAM
                client_optimizer = self._create_cmam_optimizer_copy(global_cmam_data.optimizer, client_cmam)
                
                # Create independent loss function copy
                client_loss_function = self._create_cmam_loss_function_copy(global_cmam_data.loss_function)
                
                # Create ClientCMAMData
                from fed.client import ClientCMAMData
                client_cmam_data = ClientCMAMData(
                    cmam=client_cmam,
                    cmam_optimizer=client_optimizer,
                    cmam_loss_function=client_loss_function
                )
                
                client.cmam_data.append(client_cmam_data)
                # print_success(console, f"Client {client.id} initialized with C-MAM for {target_modality}")
            
        clients_with_cmams = sum(1 for client in self.clients if client.cmam_data and len(client.cmam_data) > 0)
        print_success(console, f"C-MAM initialization complete: {clients_with_cmams}/{len(self.clients)} clients have C-MAMs")

    def _create_client_cmam_copy(self, global_cmam_data: GlobalCMAMData, target_modality: Modality) -> SimpleCMAM:
        """
        Create a completely independent copy of the global C-MAM for a client.
        Ensures NO shared state between global and client C-MAMs.
        
        Args:
            global_cmam_data: Global C-MAM data to copy
            target_modality: Target modality for the C-MAM
            
        Returns:
            New SimpleCMAM instance for the client with no shared state
        """
        from models.cmams import AssociationNetwork
        
        global_cmam = global_cmam_data.cmam
        
        # Extract architecture parameters without sharing references
        assoc_net = global_cmam.association_network
        input_size = int(assoc_net.input_size)  # Ensure new int object
        
        # Safely extract hidden and output sizes from the Sequential layers
        layers = list(assoc_net.assoc_net.children())
        from torch.nn import Linear
        linear_layers = [layer for layer in layers if isinstance(layer, Linear)]
        
        if len(linear_layers) >= 2:
            # First linear layer: Linear(input_size, hidden_size)
            first_linear = linear_layers[0]
            hidden_size = int(first_linear.out_features)
            
            # Last linear layer: Linear(hidden_size, output_size)  
            last_linear = linear_layers[-1]
            output_size = int(last_linear.out_features)
            
            print_info(console, f"Creating client C-MAM with architecture: input={input_size}, hidden={hidden_size}, output={output_size}")
        else:
            # Fallback if architecture is different
            hidden_size = input_size
            output_size = input_size
            print_warning(console, f"Using fallback architecture: input={input_size}, hidden={hidden_size}, output={output_size}")
        
        # Create completely new AssociationNetwork - no shared state
        new_association_network = AssociationNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            batch_norm=self._has_batch_norm(assoc_net),
            dropout=self._get_dropout_rate(assoc_net)
        )
        
        # Create completely new modality lists - no shared references
        input_modalities = [mod for mod in global_cmam.input_modalities]  # New list with same values
        target_modality_copy = global_cmam.target_modality  # Modality enums should be safe to share
        
        # Create new SimpleCMAM with no shared state
        client_cmam = SimpleCMAM(
            association_network=new_association_network,
            input_modalities=input_modalities,
            target_modality=target_modality_copy,
            fusion_fn=self._get_fusion_fn_name(global_cmam),
            grad_clip=float(global_cmam.grad_clip),  # Ensure new float object
            labels_key=str(global_cmam.labels_key)   # Ensure new string object
        )
        
        # Copy weights using state dict (this creates new tensors)
        global_state_dict = global_cmam.state_dict()
        # Deep copy the state dict to ensure no tensor sharing
        client_state_dict = {k: v.clone().detach() for k, v in global_state_dict.items()}
        client_cmam.load_state_dict(client_state_dict)
        
        return client_cmam

    def _has_batch_norm(self, association_network) -> bool:
        """Check if association network has batch normalization."""
        for module in association_network.assoc_net:
            if hasattr(module, 'num_features'):  # BatchNorm1d indicator
                return True
        return False

    def _get_dropout_rate(self, association_network) -> float:
        """Extract dropout rate from association network."""
        for module in association_network.assoc_net:
            if hasattr(module, 'p'):  # Dropout indicator
                return float(module.p)
        return 0.0

    def _get_fusion_fn_name(self, cmam) -> str:
        """Safely extract fusion function name."""
        if hasattr(cmam.fusion_fn, '__name__'):
            name = cmam.fusion_fn.__name__
            if name == 'cat':
                return "concat"
            return name
        # Default fallback
        return "concat"

    def _create_cmam_optimizer_copy(self, global_optimizer: Optimizer, client_cmam: SimpleCMAM) -> Optimizer:
        """
        Create a copy of the global C-MAM optimizer for a client.
        
        Args:
            global_optimizer: Global optimizer to copy
            client_cmam: Client C-MAM to optimize
            
        Returns:
            New optimizer instance for the client
        """
        # Create new optimizer of the same type with same parameters
        optimizer_class = type(global_optimizer)
        optimizer_params = global_optimizer.param_groups[0].copy()
        
        # Remove 'params' key as it will be set by the optimizer constructor
        if 'params' in optimizer_params:
            del optimizer_params['params']
        
        # Create optimizer with client C-MAM parameters
        return optimizer_class(client_cmam.parameters(), **optimizer_params)

    def _find_matching_global_cmam_key(self, client_cmam: SimpleCMAM) -> Optional[str]:
        """
        Find the global C-MAM config key that matches the client C-MAM's input and target modalities.
        
        Args:
            client_cmam: Client C-MAM to find a match for
            
        Returns:
            Config key of matching global C-MAM, or None if no match found
        """

        print_info(console, f"Trying to find matching using {client_cmam}")
        if isinstance(client_cmam, SimpleCMAM):
            client_input_modalities = set(client_cmam.input_modalities)
            client_target_modality = client_cmam.target_modality
        else :
            return client_cmam
    
        for config_key, global_cmam_data in self.cmams.items():
            global_input_modalities = set(global_cmam_data.cmam.input_modalities)
            global_target_modality = global_cmam_data.cmam.target_modality
            
            if (global_input_modalities == client_input_modalities and 
                global_target_modality == client_target_modality):
                return config_key
        
        return None

    def _create_cmam_loss_function_copy(self, global_loss_function: LossFunctionGroup) -> LossFunctionGroup:
        """
        Create a completely independent copy of the global C-MAM loss function for a client.
        Ensures NO shared state between global and client loss functions.
        
        Args:
            global_loss_function: Global loss function to copy
            
        Returns:
            New LossFunctionGroup instance for the client with no shared state
        """
        import copy
        
        # Deep copy the entire loss function group to ensure no shared references
        # This includes all loss function instances, weights, and any internal state
        client_loss_function = copy.deepcopy(global_loss_function)
        
        # Verify the copy is completely independent by checking object IDs
        if id(client_loss_function) == id(global_loss_function):
            raise RuntimeError("Loss function copy failed - objects share the same ID")
        
        # Check that any loss function instances within the group are also independent
        if hasattr(global_loss_function, 'loss_functions'):
            global_funcs = global_loss_function.loss_functions
            client_funcs = client_loss_function.loss_functions
            
            for i, (global_func, client_func) in enumerate(zip(global_funcs, client_funcs)):
                if id(global_func) == id(client_func):
                    raise RuntimeError(f"Loss function {i} copy failed - objects share the same ID")
        
        return client_loss_function

    def _distribute_base_model_to_clients(self) -> None:
        """Distribute only the base model to clients."""
        console.rule("[bold green]Distributing Base Model to Clients")

        model_state = {
            "model_state_dict": self.global_model.state_dict(),
            "optimizer_state_dict": self.gm_optimizer.state_dict(),
        }
        
        # Include scheduler state if available
        if self.gm_scheduler is not None:
            model_state["scheduler_state_dict"] = self.gm_scheduler.state_dict()

        for client in self.clients:
            client.load_lm_state(model_state)
            client.update_bytes_received(self.model_parameters_size_bytes)
            print_success(console, f"Distributed base model to Client {client.id}")

    def _distribute_cmams_to_clients(self) -> None:
        """Distribute only the C-MAMs to clients."""
        console.rule("[bold green]Distributing C-MAMs to Clients")

        if not self.train_cmams or not self.cmams:
            print_warning(console, "No C-MAMs to distribute")
            return

        for client in self.clients:
            if client.cmam is None:
                if client.cmam_data is not None and len(client.cmam_data) == 0:
                    raise RuntimeError(
                        f"Client {client.id} has empty C-MAM data list. "
                        f"When train_cmams=True, all clients must have properly initialized C-MAMs. "
                        f"Expected C-MAM for one of these modalities: {list(self.cmams.keys())}"
                    )
                else:
                    raise RuntimeError(
                        f"Client {client.id} has no C-MAM data. "
                        f"When train_cmams=True, all clients must have C-MAM instances."
                    )

            # Generate client C-MAM key and find matching global key
            client_cmam_key = "".join([str(m)[0] for m in client.cmam.input_modalities]).lower()
            matching_key = self._find_matching_global_cmam_key(client_cmam_key)
            logger.info(f"Client {client.id} C-MAM key: {client_cmam_key}, matching global key: {matching_key}")
            print_info(console, f"Client {client.id} C-MAM key: {client_cmam_key}, matching global key: {matching_key}")
            
            if matching_key is None:
                available_cmam_configs = {
                    k: f"input={[str(m) for m in v.cmam.input_modalities]} -> target={v.cmam.target_modality}" 
                    for k, v in self.cmams.items()
                }
                client_config = f"input={[str(m) for m in client.cmam.input_modalities]} -> target={client.cmam.target_modality}"
                raise KeyError(f"No global C-MAM found matching client configuration: {client_config}. "
                             f"Available: {available_cmam_configs}")
            
            cmam_data = self.cmams[matching_key]
            print_info(console, f"Distributing C-MAM {matching_key} to Client {client.id}")
            print_info(console, f"Client C-MAM input modalities: {cmam_data}")
            
            print_info(console, f"Found matching global C-MAM: config key '{matching_key}' -> {[str(m) for m in client.cmam.input_modalities]} → {client.cmam.target_modality}")
            client.load_cmam_state(
                {
                    "model_state_dict": cmam_data.cmam.state_dict(),
                }
            )
            client.update_bytes_received_cmams(cmam_data.cmam.parameters_size_bytes)

            print_success(console, f"Distributed C-MAM for inputs={[str(m) for m in client.cmam.input_modalities] } - {client.cmam.target_modality} to Client {client.id}")

    def _distribute_to_clients(self) -> None:
        """Legacy method - distribute both base model and C-MAMs (for backward compatibility)."""
        self._distribute_base_model_to_clients()
        if self.train_cmams:
            self._distribute_cmams_to_clients()

    def _aggregate_client_base_model_updates(self, method: Literal["fedavg", "selective"] = "selective") -> None:
        """Aggregate only base model updates from clients with selective parameter aggregation."""
        console.rule("[bold blue]Aggregating Client Base Model Updates")
        
        if method == "selective":
            self._aggregate_selective_base_model()
        elif method == "fedavg":
            # Fallback to simple FedAvg for all parameters
            model_state = []
            for client in self.clients:
                state = client.model_paramters
                model_state.append(state)
                client.update_bytes_sent(client.model_parameters_size_bytes)
            
            aggregated_model = self._aggregate_fedavg(model_state)
            self.base.model.load_state_dict(aggregated_model)
            self.base.model.to(self.device)  # Ensure model is on correct device
            print_success(console, "Base model aggregation completed (FedAvg)")
        else:
            raise NotImplementedError(f"Aggregation method '{method}' is not implemented.")

    def _aggregate_client_cmam_updates(self, method: Literal["fedavg", "weighted"] = "weighted") -> None:
        """Aggregate C-MAM updates from clients with weighted aggregation based on modality completeness."""
        console.rule("[bold blue]Aggregating Client C-MAM Updates")
        cmams_state = defaultdict(list)
        client_participation = defaultdict(list)  # Track which clients contributed to each C-MAM
        client_objects = defaultdict(list)  # Track client objects for weight calculation

        for client in self.clients:
            if client.cmam_data and len(client.cmam_data) > 0:
                # Handle multiple C-MAMs per client (incongruent training)
                for cmam_data in client.cmam_data:
                    cmam = cmam_data.cmam
                    if cmam is not None:
                        # Use the same key generation as the old system
                        cmam_key = "".join([str(m)[0] for m in cmam.input_modalities]).lower()
                        
                        cmam_params = cmam.state_dict()
                        if cmam_params is not None:
                            cmams_state[cmam_key].append(cmam_params)
                            client_participation[cmam_key].append(client.id)
                            client_objects[cmam_key].append(client)
                            client.update_bytes_sent_cmams(cmam.parameters_size_bytes)
                            print_info(console, f"Collected C-MAM update from Client {client.id} for {cmam_key}")
            elif client.cmam and client.cmam_parameters is not None:
                # Handle single C-MAM per client (legacy/congruent training)
                cmam_key = "".join([str(m)[0] for m in client.cmam.input_modalities]).lower()
                
                cmams_state[cmam_key].append(client.cmam_parameters)
                client_participation[cmam_key].append(client.id)
                client_objects[cmam_key].append(client)
                client.update_bytes_sent_cmams(client.cmam_parameter_size_bytes)
                print_info(console, f"Collected C-MAM update from Client {client.id} for {cmam_key}")

        # Selective aggregation: only aggregate C-MAMs with sufficient participation
        min_clients_for_aggregation = max(1, 1)  # At least 25% of clients must participate
        
        if method == "weighted":
            for modality_key, cmam_states in cmams_state.items():
                participating_clients = client_participation[modality_key]
                participating_client_objects = client_objects[modality_key]
                
                # Find matching global C-MAM by target modality
                # global_cmam_key = self._find_matching_global_cmam_key(m
                global_cmam_key = modality_key
                print_debug(console, f"Finding global C-MAM for client C-MAM key: {modality_key}")
                if global_cmam_key is None:
                    print_warning(console, f"No matching global C-MAM found for client C-MAM key: {modality_key}")
                    continue
                global_cmam_key = "".join(sorted([x for x in global_cmam_key]))
                if len(cmam_states) >= min_clients_for_aggregation:
                    # Sufficient participation - perform weighted aggregation
                    aggregated_cmam = self._aggregate_weighted_cmam(cmam_states, participating_client_objects, modality_key)
                    self.cmams[global_cmam_key].cmam.load_state_dict(aggregated_cmam)
                    print_success(console, f"Weighted C-MAM aggregation completed for {modality_key} -> {global_cmam_key} from {len(participating_clients)} clients: {participating_clients}")
                elif len(cmam_states) > 0:
                    # Some participation but not enough - still do weighted aggregation
                    print_warning(console, f"Insufficient participation for C-MAM {modality_key}: {len(participating_clients)} clients (min: {min_clients_for_aggregation})")
                    print_info(console, f"Participating clients: {participating_clients}")
                    
                    aggregated_cmam = self._aggregate_weighted_cmam(cmam_states, participating_client_objects, modality_key)
                    self.cmams[global_cmam_key].cmam.load_state_dict(aggregated_cmam)
                    print_warning(console, f"C-MAM partial weighted aggregation completed for {modality_key} -> {global_cmam_key} from {len(participating_clients)} clients")
                else:
                    print_warning(console, f"No updates received for C-MAM {modality_key}")
        elif method == "fedavg":
            for modality_key, cmam_states in cmams_state.items():
                participating_clients = client_participation[modality_key]
                
                # Find matching global C-MAM by target modality
                global_cmam_key = self._find_matching_global_cmam_key(modality_key)
                if global_cmam_key is None:
                    print_warning(console, f"No matching global C-MAM found for client C-MAM key: {modality_key}")
                    continue
                
                if len(cmam_states) >= min_clients_for_aggregation:
                    # Sufficient participation - perform simple aggregation
                    aggregated_cmam = self._aggregate_fedavg(cmam_states)
                    self.cmams[global_cmam_key].cmam.load_state_dict(aggregated_cmam)
                    print_success(console, f"FedAvg C-MAM aggregation completed for {modality_key} -> {global_cmam_key} from {len(participating_clients)} clients: {participating_clients}")
                elif len(cmam_states) > 0:
                    # Some participation but not enough - still aggregate
                    print_warning(console, f"Insufficient participation for C-MAM {modality_key}: {len(participating_clients)} clients (min: {min_clients_for_aggregation})")
                    print_info(console, f"Participating clients: {participating_clients}")
                    
                    aggregated_cmam = self._aggregate_fedavg(cmam_states)
                    self.cmams[global_cmam_key].cmam.load_state_dict(aggregated_cmam)
                    print_warning(console, f"C-MAM partial aggregation completed for {modality_key} -> {global_cmam_key} from {len(participating_clients)} clients")
                else:
                    print_warning(console, f"No updates received for C-MAM {modality_key}")
        else:
            raise NotImplementedError(f"Aggregation method '{method}' is not implemented.")

    def _aggregate_client_updates(self, method: Literal["fedavg"] = "fedavg") -> None:
        """Legacy method - aggregate both base model and C-MAMs (for backward compatibility)."""
        self._aggregate_client_base_model_updates(method)
        if self.train_cmams:
            self._aggregate_client_cmam_updates(method)
    
    def _collect_communication_metrics(self) -> dict[str, Any]:
        """
        Collect communication overhead metrics from all clients.
        
        Returns:
            Dictionary containing aggregated communication metrics
        """
        total_bytes_sent = 0
        total_bytes_received = 0
        total_cmam_bytes_sent = 0
        total_cmam_bytes_received = 0
        
        client_communication = {}
        
        for client in self.clients:
            client_data = {
                "bytes_sent": client.bytes_sent,
                "bytes_received": client.bytes_received,
                "cmam_bytes_sent": client.cmam_bytes_sent,
                "cmam_bytes_received": client.cmam_bytes_received,
                "total_bytes": client.bytes_sent + client.bytes_received + client.cmam_bytes_sent + client.cmam_bytes_received
            }
            client_communication[f"client_{client.id}"] = client_data
            
            # Aggregate totals
            total_bytes_sent += client.bytes_sent
            total_bytes_received += client.bytes_received
            total_cmam_bytes_sent += client.cmam_bytes_sent
            total_cmam_bytes_received += client.cmam_bytes_received
        
        communication_metrics = {
            "total_bytes_sent": total_bytes_sent,
            "total_bytes_received": total_bytes_received, 
            "total_cmam_bytes_sent": total_cmam_bytes_sent,
            "total_cmam_bytes_received": total_cmam_bytes_received,
            "total_communication": total_bytes_sent + total_bytes_received + total_cmam_bytes_sent + total_cmam_bytes_received,
            "avg_bytes_per_client": (total_bytes_sent + total_bytes_received) / len(self.clients) if self.clients else 0,
            "avg_cmam_bytes_per_client": (total_cmam_bytes_sent + total_cmam_bytes_received) / len(self.clients) if self.clients else 0,
            "client_communication": client_communication
        }
        
        print_info(console, f"Communication metrics: {total_bytes_sent + total_bytes_received + total_cmam_bytes_sent + total_cmam_bytes_received} total bytes across {len(self.clients)} clients")
        
        return communication_metrics

    def run_final_comprehensive_testing(self):

        # We need to test all the clients 
        client_results = []
        for client in self.clients:
            # check if the client has all modalities or just one set
            cmam_results = None
            is_complete = client._is_complete_modality_client()
            available_modalities = client.available_modalities
            if is_complete:
                print_info(console, f"Running final comprehensive testing for Client {client.id} (complete modalities)")
                cmam_results = {}

                # Only test C-MAMs if this is not baseline mode
                if self.train_cmams and client.cmam_data:
                    print_info(console, f"Testing C-MAMs for complete Client {client.id}")
                    for i, cmam_data in enumerate(client.cmam_data):
                        cmam = cmam_data.cmam

                        correct_selected_patterns = get_correct_cmam_dataset_selected_patterns(
                            condition=CONDITION, cmam=cmam
                        )
                        old_patterns = self.test_dataloader.dataset.patterns
                        self.test_dataloader.dataset.patterns = [correct_selected_patterns]
                        loss_functions = cmam_data.cmam_loss_function
                        target_modality = cmam.target_modality
                        cmam_data.checkpoint_manager.load_checkpoint(model=cmam, load_best=True)
                        cmam_test_result = client._test_single_cmam(
                            cmam=cmam,
                            loss_functions=loss_functions,
                            target_modality=target_modality,
                        )
                        print_metric_summary(console, epoch="test", metrics=cmam_test_result, split="test")
                        cmam_results[str(target_modality)] = cmam_test_result

                        # cmam_results[str(target_modality)].update(cmam_test_result)
                        self.test_dataloader.dataset.patterns = old_patterns  # Restore original patterns
                else:
                    # Baseline mode: test only the base model for complete clients
                    print_info(console, f"Baseline mode: Testing base model only for complete Client {client.id}")
                    base_test_result = client._test_incongruent()  # Use the same testing logic as incomplete clients
                    cmam_results = base_test_result

                client_results.append({
                    "client_id": client.id,
                    "is_complete": is_complete,
                    "available_modalities": available_modalities,
                    "results": cmam_results
                })

            else:
                try:
                    results = client._test_incongruent()
                except Exception as e:
                    raise e
                client_results.append({
                    "client_id": client.id,
                    "is_complete": is_complete,
                    "available_modalities": available_modalities,
                    "results": results
                })
        return client_results

    def _aggregate_selective_base_model(self) -> None:
        """
        Selectively aggregate base model parameters based on client modality availability.
        Only aggregate encoder parameters from clients that had access to that modality.
        """
        from modalities import Modality
        
        print_info(console, "Performing selective base model parameter aggregation")
        
        # Group clients by available modalities
        modality_client_mapping = self._group_clients_by_modalities()
        
        # Get current global model state
        global_state = self.base.model.state_dict()
        aggregated_state = {}
        
        # Track which parameters were updated
        updated_params = set()
        
        for param_name, param_tensor in global_state.items():
            # Determine which modality this parameter belongs to
            responsible_modality = self._get_parameter_modality(param_name)
            
            if responsible_modality is not None:
                # Get clients that have this modality
                eligible_clients = modality_client_mapping.get(responsible_modality, [])
                
                if eligible_clients:
                    # Aggregate only from eligible clients
                    client_params = []
                    for client in eligible_clients:
                        client_state = client.model_paramters
                        if param_name in client_state:
                            client_params.append(client_state[param_name])
                            client.update_bytes_sent(client.model_parameters_size_bytes // len(global_state))
                    
                    if client_params:
                        # Weighted average based on client completeness
                        weights = self._calculate_client_weights(eligible_clients)
                        aggregated_state[param_name] = self._weighted_average(client_params, weights)
                        updated_params.add(param_name)
                        print_info(console, f"Updated {param_name} from {len(eligible_clients)} clients with {responsible_modality}")
                    else:
                        # Keep global parameter unchanged
                        aggregated_state[param_name] = param_tensor
                        print_warning(console, f"No client updates for {param_name} ({responsible_modality}), keeping global")
                else:
                    # No clients have this modality, keep global parameter
                    aggregated_state[param_name] = param_tensor
                    print_warning(console, f"No clients available for {param_name} ({responsible_modality}), keeping global")
            else:
                # Shared parameters (fusion layers, classifiers) - aggregate from all clients
                client_params = []
                for client in self.clients:
                    client_state = client.model_paramters
                    if param_name in client_state:
                        client_params.append(client_state[param_name])
                
                if client_params:
                    # Equal weight aggregation for shared parameters
                    aggregated_state[param_name] = sum(client_params) / len(client_params)
                    updated_params.add(param_name)
                    print_info(console, f"Updated shared parameter {param_name} from {len(client_params)} clients")
                else:
                    aggregated_state[param_name] = param_tensor
        
        # Load the aggregated state
        self.base.model.load_state_dict(aggregated_state)
        self.base.model.to(self.device)  # Ensure model is on correct device
        
        # Save aggregation report
        self._save_aggregation_report("base_model", updated_params, global_state, modality_client_mapping)
        
        print_success(console, f"Selective base model aggregation completed: {len(updated_params)}/{len(global_state)} parameters updated")
    
    def _group_clients_by_modalities(self) -> dict[Modality, list]:
        """
        Group clients by the modalities they have access to.
        
        Returns:
            dict mapping Modality to list of clients that have that modality
        """
        from modalities import Modality
        
        modality_mapping = defaultdict(list)
        
        for client in self.clients:
            # Parse client's available modalities string
            available_modalities = self._parse_client_modalities(client.available_modalities)
            
            for modality in available_modalities:
                modality_mapping[modality].append(client)
        
        print_info(console, "Client modality distribution:")
        for modality, clients in modality_mapping.items():
            client_ids = [c.id for c in clients]
            print_info(console, f"  {modality}: {len(clients)} clients {client_ids}")
        
        return dict(modality_mapping)
    
    def _parse_client_modalities(self, available_modalities_str: str) -> list[Modality]:
        """
        Parse client's available modalities string into Modality objects.
        
        Args:
            available_modalities_str: String like "missing_audio", "missing_video", "complete", "ai", "av", etc.
            
        Returns:
            list of Modality objects that the client has access to
        """
        from modalities import Modality
        
        # Get all possible modalities for this model
        all_modalities = [Modality.AUDIO, Modality.VIDEO, Modality.TEXT, Modality.IMAGE]
        
        if available_modalities_str == "complete":
            # Client has all modalities
            return all_modalities
        elif available_modalities_str.startswith("missing_"):
            # Client is missing specific modality
            missing_modality_str = available_modalities_str.replace("missing_", "")
            try:
                missing_modality = Modality.from_str(missing_modality_str.upper())
                return [m for m in all_modalities if m != missing_modality]
            except:
                print_warning(console, f"Unknown missing modality: {missing_modality_str}")
                return all_modalities
        else:
            # Handle short patterns like "ai", "av", "tv", etc.
            available_modalities = []
            pattern = available_modalities_str.lower()
            
            # Map common pattern letters to modalities
            modality_map = {
                'a': Modality.AUDIO,
                'v': Modality.VIDEO, 
                'i': Modality.IMAGE,
                't': Modality.TEXT
            }
            
            for char in pattern:
                if char in modality_map:
                    available_modalities.append(modality_map[char])
            
            if available_modalities:
                # print_info(console, f"Parsed modality pattern '{pattern}' as: {[str(m) for m in available_modalities]}")
                return available_modalities
            else:
                print_warning(console, f"Unknown modality configuration: {available_modalities_str}, assuming complete")
                return all_modalities
    
    def _get_parameter_modality(self, param_name: str) -> Optional[Modality]:
        """
        Determine which modality a parameter belongs to based on its name.
        
        Args:
            param_name: Name of the parameter (e.g., "audio_encoder.net.0.weight")
            
        Returns:
            Modality that owns this parameter, or None for shared parameters
        """
        from modalities import Modality
        
        param_name_lower = param_name.lower()
        
        # Audio encoder parameters
        if any(keyword in param_name_lower for keyword in ["audio_encoder", "audio_model", "neta", "audio"]):
            return Modality.AUDIO
        
        # Video encoder parameters  
        if any(keyword in param_name_lower for keyword in ["video_encoder", "video_model", "netv", "video"]):
            return Modality.VIDEO
        
        # Text encoder parameters
        if any(keyword in param_name_lower for keyword in ["text_encoder", "text_model", "nett", "text"]):
            return Modality.TEXT
        
        # Image encoder parameters
        if any(keyword in param_name_lower for keyword in ["image_encoder", "image_model", "neti", "image"]):
            return Modality.IMAGE
        
        # Shared parameters (fusion layers, classifiers, etc.)
        return None
    
    def _calculate_client_weights(self, clients: list) -> list[float]:
        """
        Calculate weights for clients based on their modality completeness.
        Clients with more available modalities get higher weights.
        
        Args:
            clients: List of clients to calculate weights for
            
        Returns:
            List of weights (normalized to sum to 1.0)
        """
        weights = []
        
        for client in clients:
            available_modalities = self._parse_client_modalities(client.available_modalities)
            # Weight based on number of available modalities
            weight = len(available_modalities)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Equal weights if all clients have 0 modalities (shouldn't happen)
            weights = [1.0 / len(clients)] * len(clients)
        
        return weights
    
    def _weighted_average(self, tensors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
        """
        Compute weighted average of tensors with proper dtype handling.
        
        Args:
            tensors: List of tensors to average
            weights: List of weights (should sum to 1.0)
            
        Returns:
            Weighted average tensor
        """
        if not tensors:
            raise ValueError("Cannot compute weighted average of empty tensor list")
        
        # Get the dtype and device of the first tensor
        reference_tensor = tensors[0]
        original_dtype = reference_tensor.dtype
        result_device = reference_tensor.device
        
        # For integer tensors, we need to compute in float and then cast back
        is_integer_type = original_dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
        
        if is_integer_type:
            # Convert to float for computation
            compute_dtype = torch.float32
        else:
            # Use original dtype for computation
            compute_dtype = original_dtype
        
        # Initialize result tensor with computation dtype
        result = torch.zeros_like(reference_tensor, dtype=compute_dtype, device=result_device)
        
        for tensor, weight in zip(tensors, weights):
            # Convert tensor to computation dtype and device
            tensor = tensor.to(device=result_device, dtype=compute_dtype)
            result += tensor * weight
        
        # Convert back to original dtype if needed
        if is_integer_type:
            result = result.round().to(dtype=original_dtype)
        
        return result
    
    def _aggregate_weighted_cmam(self, cmam_states: list[dict[str, Any]], clients: list, modality_key: str) -> dict[str, Any]:
        """
        Aggregate C-MAM parameters using weighted averaging based on client modality completeness.
        Clients with more available modalities (especially complete clients) get higher weights.
        
        Args:
            cmam_states: List of C-MAM state dictionaries from participating clients
            clients: List of client objects that contributed the C-MAM states
            modality_key: Key identifying which C-MAM is being aggregated
            
        Returns:
            Aggregated C-MAM state dictionary
        """
        if not cmam_states:
            raise ValueError("Cannot aggregate empty C-MAM states list")
        
        # Calculate weights based on client modality completeness
        weights = self._calculate_cmam_client_weights(clients, modality_key)
        
        print_info(console, f"C-MAM {modality_key} aggregation weights:")
        for client, weight in zip(clients, weights):
            modalities = self._parse_client_modalities(client.available_modalities)
            print_info(console, f"  Client {client.id} ({client.available_modalities}): {weight:.3f} (has {len(modalities)} modalities)")
        
        # Weighted aggregation of all parameters
        aggregated_state = {}
        
        for param_name in cmam_states[0].keys():
            # Get all client parameters for this parameter name
            client_params = [state[param_name] for state in cmam_states]
            
            # Compute weighted average
            aggregated_param = self._weighted_average(client_params, weights)
            aggregated_state[param_name] = aggregated_param
        
        # Save C-MAM aggregation report
        cmam_weights_info = {
            modality_key: {
                "client_weights": {
                    clients[i].id: {
                        "weight": weights[i],
                        "available_modalities": clients[i].available_modalities,
                        "modality_count": len(self._parse_client_modalities(clients[i].available_modalities))
                    }
                    for i in range(len(clients))
                },
                "aggregation_summary": {
                    "total_clients": len(clients),
                    "weight_distribution": weights,
                    "max_weight": max(weights),
                    "min_weight": min(weights),
                    "weight_variance": np.var(weights)
                }
            }
        }
        
        # Save report using all C-MAM parameters as the "all_params" reference
        self._save_aggregation_report("cmam", set(cmam_states[0].keys()), cmam_states[0], cmam_weights=cmam_weights_info)
        
        print_success(console, f"Weighted C-MAM aggregation for {modality_key}: {len(clients)} clients with weights {[f'{w:.3f}' for w in weights]}")
        return aggregated_state
    
    def _calculate_cmam_client_weights(self, clients: list, modality_key: str) -> list[float]:
        """
        Calculate weights for C-MAM aggregation based on client modality completeness.
        
        The weighting strategy prioritizes:
        1. Complete clients (have all modalities) - highest weight
        2. Clients with more available modalities - higher weight  
        3. Clients that can actually benefit from this specific C-MAM - moderate weight
        
        Args:
            clients: List of client objects
            modality_key: The C-MAM being aggregated (e.g., "av" for audio-video to missing)
            
        Returns:
            List of normalized weights (sum to 1.0)
        """
        weights = []
        
        # Find the global C-MAM that matches this modality key
        # global_cmam_key = self._find_matching_global_cmam_key(modality_key)
        global_cmam_key = modality_key  # Use the same key as the client C-MAM key
        if global_cmam_key is None:
            # Fallback to equal weights
            print_warning(console, "No matching global C-MAM for weight calculation, using equal weights")
            return [1.0 / len(clients)] * len(clients)
        
        global_cmam_key = "".join(sorted([x for x in global_cmam_key]))
        global_cmam_data = self.cmams[global_cmam_key]
        target_modality = global_cmam_data.cmam.target_modality
        input_modalities = global_cmam_data.cmam.input_modalities
        
        for client in clients:
            available_modalities = self._parse_client_modalities(client.available_modalities)
            
            # Base weight: number of available modalities (more complete = higher weight)
            base_weight = len(available_modalities)
            
            # Bonus for complete clients (they have ground truth for everything)
            if client.available_modalities == "complete":
                completeness_bonus = 2.0  # Double weight for complete clients
            else:
                completeness_bonus = 1.0
            
            # Bonus for clients that actually need this C-MAM (missing the target modality)
            if target_modality not in available_modalities:
                relevance_bonus = 1.5  # 50% bonus for clients that benefit from this C-MAM
            else:
                relevance_bonus = 1.0
            
            # Bonus for clients that have the input modalities (can train this C-MAM effectively)
            input_modalities_available = sum(1 for mod in input_modalities if mod in available_modalities)
            input_coverage = input_modalities_available / len(input_modalities)
            input_bonus = 0.5 + input_coverage  # 0.5 to 1.5 range
            
            # Combined weight
            total_weight = base_weight * completeness_bonus * relevance_bonus * input_bonus
            weights.append(total_weight)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Fallback to equal weights
            weights = [1.0 / len(clients)] * len(clients)
        
        return weights
    
    def _calculate_parameter_bytes(self, params: dict) -> int:
        """
        Calculate the total size in bytes of PyTorch parameters.
        
        Args:
            params: Dictionary of parameter names to parameter tensors
            
        Returns:
            Total size in bytes
        """
        total_bytes = 0
        for param_name, param_tensor in params.items():
            if hasattr(param_tensor, 'element_size') and hasattr(param_tensor, 'numel'):
                total_bytes += param_tensor.element_size() * param_tensor.numel()
        return total_bytes
    
    def _calculate_parameter_bytes_from_names(self, param_names: list, all_params: dict) -> int:
        """
        Calculate the total size in bytes for specific parameter names.
        
        Args:
            param_names: List of parameter names to calculate size for
            all_params: Dictionary of all available parameters
            
        Returns:
            Total size in bytes for the specified parameters
        """
        total_bytes = 0
        for param_name in param_names:
            if param_name in all_params:
                param_tensor = all_params[param_name]
                if hasattr(param_tensor, 'element_size') and hasattr(param_tensor, 'numel'):
                    total_bytes += param_tensor.element_size() * param_tensor.numel()
        return total_bytes
    
    def _save_aggregation_report(self, model_type: str, updated_params: set, all_params: dict, 
                                modality_mapping: dict = None, cmam_weights: dict = None) -> None:
        """
        Save a detailed report of what was aggregated in this round.
        
        Args:
            model_type: "base_model" or "cmam" 
            updated_params: Set of parameter names that were updated
            all_params: Dict of all parameters in the model
            modality_mapping: Dict mapping modalities to clients (for base model)
            cmam_weights: Dict mapping C-MAM keys to weight information (for C-MAMs)
        """
        import json
        from datetime import datetime
        
        # Calculate byte sizes
        total_bytes = self._calculate_parameter_bytes(all_params)
        updated_bytes = self._calculate_parameter_bytes_from_names(list(updated_params), all_params)
        skipped_params = [name for name in all_params.keys() if name not in updated_params]
        skipped_bytes = self._calculate_parameter_bytes_from_names(skipped_params, all_params)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "round": self.current_round,
            "model_type": model_type,
            "total_parameters": len(all_params),
            "updated_parameters": len(updated_params),
            "update_rate": len(updated_params) / len(all_params) if all_params else 0.0,
            "total_bytes": total_bytes,
            "updated_bytes": updated_bytes,
            "skipped_bytes": skipped_bytes,
            "update_rate_bytes": updated_bytes / total_bytes if total_bytes > 0 else 0.0,
            "updated_parameter_list": list(updated_params),
            "skipped_parameter_list": skipped_params
        }
        
        if model_type == "base_model" and modality_mapping:
            # Add modality-specific information
            modality_info = {}
            for modality, clients in modality_mapping.items():
                modality_params = [
                    name for name in updated_params 
                    if self._get_parameter_modality(name) == modality
                ]
                modality_bytes = self._calculate_parameter_bytes_from_names(modality_params, all_params)
                
                modality_info[str(modality)] = {
                    "client_count": len(clients),
                    "client_ids": [c.id for c in clients],
                    "parameters_updated": modality_params,
                    "parameters_updated_count": len(modality_params),
                    "parameters_updated_bytes": modality_bytes
                }
            report["modality_distribution"] = modality_info
            
        elif model_type == "cmam" and cmam_weights:
            # Add C-MAM weight information
            report["cmam_weights"] = cmam_weights
        
        # Save to metrics directory
        report_path = self.metrics_fp / f"aggregation_report_{model_type}_round_{self.current_round}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print_info(console, f"Aggregation report saved: {report_path}")
    
    def _aggregate_fedavg(self, model_state: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Aggregates model parameters using Federated Averaging (FedAvg).
        """
        console.print("[blue]Aggregating model parameters using FedAvg...")
        total_clients = len(model_state)
        aggregated_state = {}

        for key in model_state[0].keys():
            # Compute average
            aggregated_param = sum(client_state[key] for client_state in model_state) / total_clients
            
            # Ensure proper device placement and gradient requirements
            if torch.is_tensor(aggregated_param):
                aggregated_param = aggregated_param.to(self.device)
                if aggregated_param.requires_grad:
                    aggregated_param = aggregated_param.detach().requires_grad_(True)
            
            aggregated_state[key] = aggregated_param

        return aggregated_state

    def info(self) -> str:
        s = "[GlobalTrainer Info]\n"
        s += f"  Base model: {self.global_model.__class__.__name__}\n"
        s += f"  Optimizer: {self.optimizer.__class__.__name__}\n"
        s += f"  Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'}\n"
        s += f"  Epochs: {self.epochs}\n"
        s += f"  Training C-MAMs: {self.train_cmams}\n"
        if self.train_cmams:
            s += f"  C-MAM targets: {[str(mod) for mod in self.cmams.keys()]}\n"
        s += f"  Device: {self.device}\n"
        return s

    @property
    def config(self) -> FederatedExperimentConfig:
        """
        Returns the configuration for the federated trainer.
        """
        return self.base.config

    @property
    def global_model(self) -> MultimodalModelProtocol:
        """
        Returns the global model for the federated trainer.
        """
        return self.base.model

    @property
    def gm_optimizer(self) -> Optimizer:
        """
        Returns the optimizer for the local model.
        """
        return self.base.optimizer

    @property
    def gm_loss_function(self) -> LossFunctionGroup:
        """
        Returns the loss function for the local model.
        """
        return self.base.loss_function

    @property
    def gm_scheduler(self) -> Optional[LRScheduler]:
        """
        Returns the learning rate scheduler for the local model, if any.
        """
        return self.base.scheduler

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
    def do_early_stopping(self) -> bool:
        """
        Returns whether early stopping is enabled for the client.
        """
        return self.config.fed_config.global_training.early_stopping

    @property
    def metrics_fp(self) -> Path:
        """Returns the file path for the global model with round information."""
        return ensure_dir(Path(self.config.logging.metrics_path) / "global" / f"round_{self.current_round}")

    @property
    def model_fp(self) -> Path:
        """Returns the file path for the global model with round information."""
        return ensure_dir(Path(self.config.logging.model_output_path) / "global" / f"round_{self.current_round}")

    @property
    def logging_fp(self) -> Path:
        """Returns the file path for the global model's logging with round information."""
        return ensure_dir(Path(self.config.logging.logging_path) / "global" / f"round_{self.current_round}")

    @property
    def model_paramters(self) -> dict[str, Any]:
        """
        Returns the parameters of the local model.
        This is useful for saving and loading model parameters.
        """
        return self.global_model.state_dict()

    @property
    def model_parameters_size_bytes(self) -> int:
        """
        Returns the size of the local model's parameters in bytes.
        This is useful for tracking model size.
        """
        return sum(p.numel() * p.element_size() for p in self.global_model.parameters())

    def set_current_round(self, round_number: int) -> None:
        """
        Set the current federated learning round number for the global model.
        This affects the directory structure for saving metrics, models, and logs.
        
        Args:
            round_number: The current round number (0-indexed or 1-indexed depending on convention)
        """
        self.current_round = round_number
        print_info(console, f"Global model set to round {round_number}")
    
    def finalize_convergence_monitoring(self) -> None:
        """
        Finalize convergence monitoring and save summary.
        Call this at the end of federated training.
        """
        if self.convergence_monitor:
            self.convergence_monitor.save_final_summary()
            print_success(console, "Convergence monitoring finalized")
    
    def has_converged(self) -> bool:
        """Check if the model has converged."""
        return self.convergence_monitor.converged if self.convergence_monitor else False
    
    def get_convergence_info(self) -> dict:
        """Get current convergence information."""
        if not self.convergence_monitor:
            return {"converged": False}
        
        return {
            "converged": self.convergence_monitor.converged,
            "convergence_round": self.convergence_monitor.convergence_round,
            "wait_count": self.convergence_monitor.wait_count,
            "best_metric": self.convergence_monitor.best_metric,
            "primary_metric": self.convergence_monitor.primary_accuracy_key
        }

