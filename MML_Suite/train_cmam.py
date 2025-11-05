import atexit
import json
import os
import subprocess
import sys
import time
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from train_multimodal import count_parameters
from config.cmam_config import CMAMConfig
from config.multimodal_training_config import StandardMultimodalConfig
from config.resolvers import resolve_init_fn, resolve_model_name
from experiment_utils.checkpoints import CheckpointManager
from experiment_utils.experiment_report import (
    EmbeddingVisualizationReport,
    ExperimentReportGenerator,
    MetricsReport,
    ModelReport,
    TimingReport,
)
from experiment_utils.logging import LoggerSingleton, configure_logger, get_logger
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.monitoring import ExperimentMonitor
from experiment_utils.printing import EnhancedConsole, get_console
from experiment_utils.utils import (
    PARAMETER_SIZE_BYTES,
    AccessError,
    NestedDictAccess,
    SafeDict,
    clean_checkpoints,
    gpu_memory,
    prepare_metrics_for_json,
    safe_detach,
)
from modalities import add_modality
from models.protocols import MultimodalModelProtocol
from rich import box
from rich.panel import Panel
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

warnings.filterwarnings(
    "error",
    message="Degrees of freedom <= 0 for slice",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "error",
    message="divide by zero encountered in divide",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "error",
    message="invalid value encountered in multiply",
    category=RuntimeWarning,
)


def shutdown_cursor_reset_hook() -> None:
    subprocess.run(["tput", "cnorm"])


atexit.register(shutdown_cursor_reset_hook)

# Add modalities and initialize utilities
add_modality("video")
console: EnhancedConsole = get_console()
logger: Optional[LoggerSingleton] = None


def setup_experiment(config_path: str, run_id: int) -> CMAMConfig:
    """
    Set up the experiment configuration and logging.

    Args:
        config_path (str): Path to the experiment configuration file.
        run_id (int): Unique identifier for this experiment run.

    Returns:
        StandardMultimodalConfig: Loaded experiment configuration.
    """
    global logger
    config = CMAMConfig.load(config_path, run_id)

    # Configure logging
    configure_logger(log_path=config.logging.log_path, suffix="")
    logger = get_logger()

    logger.info(f"Starting experiment with run ID {run_id}")
    logger.info(f"Configuration:\n{config}")
    console.rule(f"Starting experiment with run ID {run_id}")

    return config


def setup_dataloaders(config: StandardMultimodalConfig) -> Dict[str, DataLoader]:
    """
    Set up the data loaders for training and evaluation.

    Args:
        config (StandardMultimodalConfig): Experiment configuration.

    Returns:
        Dict[str, DataLoader]: Dictionary of data loaders for different splits.
    """
    logger.debug("Building dataloaders...")

    dataloaders = config.data.build_all_dataloaders(
        is_train=config.experiment.is_train, is_test=config.experiment.is_test
    )
    console.print(f"Finished building dataloaders. Created: {list(dataloaders.keys())}")

    for split, loader in dataloaders.items():
        console.print(f"Loader batch size {loader.batch_size}")
        dataset_size = len(loader.dataset)
        logger.debug(f"{split} dataset size: {dataset_size}")
        if split in ["train", "validation"]:
            total_iterations = config.training.epochs * (dataset_size // loader.batch_size)
            logger.debug(f"Total {split} iterations: {total_iterations}")

    return dataloaders


def setup_model_components(
    config: CMAMConfig,
    dataloaders: Optional[DataLoader | Dict[str, DataLoader]] = None,
    fold:int = None,

) -> Tuple[
    MultimodalModelProtocol,
    MultimodalModelProtocol,
    Optimizer,
    LossFunctionGroup,
    Optional[LRScheduler],
    torch.device,
    MetricRecorder,
]:
    """
    Set up the model and its training components.

    Args:
        config (StandardMultimodalConfig): Experiment configuration.
        dataloaders (Optional[Union[DataLoader, Dict[str, DataLoader]]]): Optional dataloaders for initialization.

    Returns:
        Tuple[Module, Optimizer, LossFunctionGroup, Optional[LRScheduler], torch.device, MetricRecorder]:
            Model, optimizer, criterion, scheduler, device, and metric recorder.
    """

    metric_recorder = MetricRecorder(
        config.metrics,
        tensorboard_path=config.logging.tensorboard_path,
        tb_record_only=config.logging.tb_record_only,
    )

    logger.debug("Building model...")

    base_model_cls: MultimodalModelProtocol = resolve_model_name(config.model.name)
    cmam_model_cls: MultimodalModelProtocol = resolve_model_name(config.cmam.name)

    base_model = base_model_cls(**config.model.kwargs)
    cmam_model = cmam_model_cls(**config.cmam.kwargs)

    if (
        hasattr(base_model, "post_init_with_dataloaders")
        and callable(base_model.post_init_with_dataloaders)
        and dataloaders
    ):
        console.print("Initializing model with dataloaders")
        base_model.post_init_with_dataloaders(dataloaders)

    if (
        hasattr(cmam_model, "post_init_with_dataloaders")
        and callable(cmam_model.post_init_with_dataloaders)
        and dataloaders
    ):
        console.print("Initializing model with dataloaders")
        cmam_model.post_init_with_dataloaders(dataloaders)

    if config.model.init_fn is not None:
        init_fn = resolve_init_fn(config.model.init_fn)
        init_fn(base_model)
        console.print(f"[green]✓[/] Initialized model with {config.model.init_fn}")

    if config.cmam.init_fn is not None:
        init_fn = resolve_init_fn(config.cmam.init_fn)
        init_fn(cmam_model)
        console.print(f"[green]✓[/] Initialized model with {config.cmam.init_fn}")

    console.print("[green]✓[/] Model created successfully")
    console.print(
        Panel(str(base_model), box=box.SQUARE, highlight=True, expand=True, title="[heading]Model Architecture[/]")
    )
    console.print(
        Panel(str(cmam_model), box=box.SQUARE, highlight=True, expand=True, title="[heading]CMAM Model Architecture[/]")
    )

    logger.info(f"Model: {base_model}")
    logger.info(f"CMAM Model: {cmam_model}")

    device = config.experiment.device
    base_model.to(device)
    cmam_model.to(device)

    base_model.eval()

    optimizer = config.get_optimizer(cmam_model)
    criterion = config.training.loss_functions
    console.print("[green]✓[/] Optimizer and criterion created")
    logger.info(f"Optimizer and criterion created\n{optimizer}\n{criterion}")

    scheduler = None
    if config.training.scheduler:
        scheduler = config.get_scheduler(optimizer=optimizer)
        console.print("[green]✓[/] Scheduler created")
        logger.info(f"Scheduler created\n{scheduler}")
    else:
        console.print("[bold yellow]![/] No scheduler")

    if config.model.pretrained_path:
        pt_path = config.model.pretrained_path
        if fold:
            model_path = config.model.pretrained_path.replace(f"models/{config.experiment.run_id}/", f"models/{config.experiment.run_id}/fold_{fold}/")
            pt_path = model_path

        logger.info(f"Loading pretrained model from {pt_path}")
        console.print(f"[bold green] >> [/] Loading pretrained model from {pt_path}")
        base_model.load_state_dict(torch.load(pt_path, weights_only=True, map_location="cpu")["model_state_dict"])

    else:
        console.print("No model weights provided")
        exit(-1)
    if (
        "load_pretrained_encoder_state_for" in config.cmam.kwargs
        and len(config.cmam.kwargs["load_pretrained_encoder_state_for"]) > 0
    ):
        data = {
            modality: base_model.get_encoder(modality).state_dict().copy()
            for modality in config.cmam.kwargs["load_pretrained_encoder_state_for"]
        }

        console.print(f"> Loading encoder state for {config.cmam.kwargs['load_pretrained_encoder_state_for']}")
        cmam_model.load_encoder_state_for(data)

    elif config.cmam.pretrained_path:
        logger.info(f"Loading pretrained C-MAM model from {config.cmam.pretrained_path}")
        console.print(f"Loading pretrained C-MAM model from {config.cmam.pretrained_path}")
        cmam_model.load_state_dict(torch.load(config.cmam.pretrained_path, weights_only=True)["model_state_dict"])
    else:
        # If no explicit C-MAM checkpoint path is provided and we're in test mode,
        # try to find the checkpoint automatically
        if not config.experiment.is_train:
            logger.info("No explicit C-MAM checkpoint path provided. Attempting to find checkpoint automatically.")
            console.print("[yellow]Warning:[/] No explicit C-MAM checkpoint path provided.")
            console.print("[yellow]Will attempt to load checkpoint automatically during testing.[/]")
            
            # Try to find if there's a checkpoint available
            temp_checkpoint_manager = CheckpointManager(
                model_dir=config.logging.model_output_path,
                save_metric=config.logging.save_metric,
                mode="minimize" if config.logging.save_metric == "loss" else "maximize",
                device=config.experiment.device,
            )
            try:
                checkpoint_path = temp_checkpoint_manager.get_best_checkpoint()
                if temp_checkpoint_manager.validate_checkpoint(checkpoint_path):
                    logger.info(f"Found and validated C-MAM checkpoint at: {checkpoint_path}")
                    console.print(f"[green]✓[/] Found C-MAM checkpoint: {checkpoint_path}")
                    # Load it immediately to ensure consistency
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    cmam_model.load_state_dict(checkpoint["model_state_dict"])
                    console.print(f"[green]✓[/] Loaded C-MAM checkpoint for test-only mode")
                    cmam_model.eval()  # Ensure model is in eval mode for testing
                else:
                    logger.warning(f"Found but failed to validate C-MAM checkpoint at: {checkpoint_path}")
                    console.print(f"[red]✗[/] C-MAM checkpoint validation failed: {checkpoint_path}")
            except FileNotFoundError as e:
                logger.error(f"No C-MAM checkpoint found: {e}")
                console.print("[red]Error:[/] No C-MAM checkpoint found. Model will use random weights!")
                console.print("[red]This will likely produce incorrect evaluation results.[/]")
                # The CheckpointManager.get_best_checkpoint() already provided recovery suggestions
                raise RuntimeError(
                    "C-MAM checkpoint is required for testing but was not found. "
                    "Please ensure training was completed or provide explicit checkpoint path."
                ) from e

    console.print(f"> Loss Functions: {(criterion)}")
    console.print(f"{type(criterion)}")
    return cmam_model, base_model, optimizer, criterion, scheduler, device, metric_recorder


def check_early_stopping(
    val_metrics: Dict[str, Any],
    best_metrics: Optional[Dict[str, Any]],
    patience: int,
    min_delta: float,
    wait: int,
    mode: Literal["minimize", "maximize"] = "minimize",
    target_metric: str = "loss",
) -> Tuple[bool, bool, int]:
    """
    Check early stopping conditions based on validation metrics.

    Args:
        val_metrics (Dict[str, Any]): Current validation metrics.
        best_metrics (Optional[Dict[str, Any]]): Best metrics recorded so far.
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum improvement threshold to reset patience.
        wait (int): Current wait count since the last improvement.
        mode (str): "minimize" to minimize the metric, or "maximize" to maximize.

    Returns:
        Tuple[bool, bool, int]:
            is_best (bool): Whether the current metrics are the best so far.
            should_continue (bool): Whether training should continue.
            wait (int): Updated wait count.
    """
    if best_metrics is None:
        # No best metrics yet; current metrics are the best by default
        return True, True, 0

    metric_value = val_metrics.get("classification").get(target_metric, None)
    best_value = best_metrics.get("classification").get(target_metric, None)


    # Check for improvement
    if (mode == "minimize" and metric_value < best_value - min_delta) or (
        mode == "maximize" and metric_value > best_value + min_delta
    ):
        console.print(f"[bold green]>>[/] Improvement detected: {best_value:.4f} -> {metric_value:.4f}")
        return True, True, 0  # Improvement detected, reset wait

    # No improvement
    wait += 1
    should_continue = wait < patience
    return False, should_continue, wait


def setup_tracking(
    config: CMAMConfig, output_dir: Path, model: Module
) -> Tuple[CheckpointManager, Dict[str, Any], ExperimentReportGenerator, Optional[ExperimentMonitor]]:
    """
    Set up tracking components for the experiment.

    Args:
        config (StandardMultimodalConfig): Experiment configuration.
        output_dir (Path): Directory to store outputs.
        model (Module): Model being trained.

    Returns:
        Tuple[CheckpointManager, Dict[str, Any], ExperimentReportGenerator, Optional[ExperimentMonitor]]:
            Checkpoint manager, experiment data dictionary, report generator, and optional monitor.
    """
    checkpoint_manager = CheckpointManager(
        model_dir=config.logging.model_output_path,
        save_metric=config.logging.save_metric,
        mode="minimize" if config.logging.save_metric == "loss" else "maximize",
        device=config.experiment.device,
    )

    experiment_data = {
        "metrics_history": {"train": [], "validation": [], "test": []},
        "timing_history": {"train": [], "validation": []},
        "embeddings": None,
        "model_info": {},
    }

    subreports = {
        "metrics": MetricsReport(
            output_dir=config.logging.metrics_path,
            metric_keys=list(config.metrics.metrics.keys()),
        ),
        "embeddings": EmbeddingVisualizationReport(
            output_dir=config.logging.metrics_path / "embeddings",
            visualization_fn=model.mm_config.visualize_embeddings
            if hasattr(model, "visualize_embeddings")
            else lambda x, y: (x, y),
        ),
        "model": ModelReport(output_dir=config.logging.metrics_path),
        "timing": TimingReport(output_dir=config.logging.metrics_path),
    }

    report_generator = ExperimentReportGenerator(output_dir=output_dir, config=config, subreports=subreports)
    monitor = None
    if config.monitoring.enabled:
        monitor = ExperimentMonitor(config.monitoring, model=model, log_dir=config.logging.monitor_path)
        model.attach_monitor(monitor)
        console.print(f"Monitor: {monitor}")

    console.print(f"Checkpoints Manager: {checkpoint_manager}")
    console.print(f"Report Generator: {report_generator}")
    return checkpoint_manager, experiment_data, report_generator, monitor


def train_epoch(
    cmam: MultimodalModelProtocol,
    model: MultimodalModelProtocol,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_functions: LossFunctionGroup,
    device: torch.device,
    epoch: int,
    metric_recorder: MetricRecorder,
    monitor: Optional[ExperimentMonitor] = None,
) -> Tuple[float, float, Dict[str, List[float]]]:
    """
    Run one training epoch.

    Args:
        model (MultimodalModelProtocol): Model to train.
        train_loader (DataLoader): Data loader for training data.
        optimizer (Optimizer): Optimizer for model parameters.
        criterion (LossFunctionGroup): Loss function group.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.
        metric_recorder (MetricRecorder): Recorder for metrics.
        monitor (Optional[ExperimentMonitor]): Optional monitor for experiment progress.

    Returns:
        Tuple[float, float]: Average loss and time per batch for this epoch.
    """
    model.eval()
    cmam.train()

    losses = defaultdict(list)

    console.start_task("Training", total=len(train_loader), style="light slate_blue")

    start_time = time.time()
    for i, batch in enumerate(train_loader):
        train_output: Dict[str, Any] = cmam.train_step(
            batch=batch,
            loss_functions=loss_functions,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            metric_recorder=metric_recorder,
            trained_model=model,
        )

        loss: float = train_output["loss"]
        other_losses = train_output.get("losses", None)

        losses["loss"].append(loss)
        if other_losses:
            for key, value in other_losses.items():
                losses[key].append(value)
        if monitor:
            monitor.step()
        console.update_task("Training", advance=1)
    time_taken = time.time() - start_time

    console.complete_task("Training")

    losses = {key: np.mean(value) for key, value in losses.items()}
    return losses["loss"], time_taken, losses


def validate_epoch(
    cmam: MultimodalModelProtocol,
    model: MultimodalModelProtocol,
    val_loader: DataLoader,
    loss_functions: LossFunctionGroup,
    device: torch.device,
    console: EnhancedConsole,
    metric_recorder: MetricRecorder,
    monitor: Optional[ExperimentMonitor] = None,
    task_name: str = "Validation",
) -> Tuple[float, float, Dict[str, List[float]]]:
    """
    Run one validation epoch.

    Args:
        model (MultimodalModelProtocol): Model to validate.
        val_loader (DataLoader): Data loader for validation data.
        loss_functions (LossFunctionGroup): Loss function group.
        device (torch.device): Device to run validation on.
        console (EnhancedConsole): Console for displaying progress.
        metric_recorder (MetricRecorder): Recorder for metrics.
        monitor (Optional[ExperimentMonitor]): Optional monitor for experiment progress.
        task_name (str): Task name for console display.

    Returns:
        Tuple[float, float]: Average loss and time per batch for this epoch.
    """
    model.eval()
    start_time = time.time()
    losses = defaultdict(list)
    targets = []
    preds = []
    logits = [] # gt logits from predictions with missing data
    rec_logits = [] # logits from using reconstructed modalities in place of the missing data
    miss_types = []
    modality_logits = []
    sample_ids = []
    console.start_task(task_name, total=len(val_loader), style="bright yellow")
    start_time = time.time()

    with torch.no_grad():
        console.print(f"Size of Loader: {len(val_loader)}")
        for batch in val_loader:
            validation_output = cmam.validation_step(
                batch,
                loss_functions=loss_functions,
                device=device,
                metric_recorder=metric_recorder,
                trained_model=model,
            )
            validation_target = validation_output["targets"] if "targets" in validation_output else None
            validation_pred = validation_output["predictions"]
            validation_logits = validation_output["logits"] if "logits" in validation_output else None
            validation_rec_logits = validation_output["rec_logits"] if "rec_logits" in validation_output else None
            validation_miss_types = validation_output["miss_type"] if "miss_type" in validation_output else None
            validation_modality_logits = validation_output["modality_logits"] if "modality_logits" in validation_output else None
            validation_sample_ids = validation_output.get("sample_ids")

            if validation_target is not None:
                targets.append(validation_target)
            preds.append(validation_pred)
            if validation_logits is not None:
                logits.append(validation_logits)
            if validation_miss_types is not None:
                miss_types.extend(validation_miss_types)
            if validation_modality_logits is not None:
                modality_logits.append(validation_modality_logits)
            if validation_sample_ids is not None:
                sample_ids.extend(validation_sample_ids)
            if validation_rec_logits is not None:
                rec_logits.append(validation_rec_logits)
            loss = validation_output["loss"]
            other_losses = validation_output.get("losses", None)

            losses["loss"].append(loss)

            if other_losses:
                for key, value in other_losses.items():
                    losses[key].append(value)

            if monitor:
                monitor.step()
            console.update_task(task_name, advance=1)
    time_taken = time.time() - start_time
    time_taken /= len(val_loader) if len(val_loader) != 0 else 1.0
    time_taken = round(time_taken, 8)
    console.complete_task(task_name)
    if len(targets) > 0:
        targets = np.concat(targets)
    if len(preds) > 0:
        preds = np.concat([safe_detach(p) for p in preds])
    if len(logits) > 0:
        logits = np.concat([safe_detach(l) for l in logits])
    if len(rec_logits) > 0:
        rec_logits = np.concat([safe_detach(l) for l in rec_logits])

    
    losses = {key: np.mean(value) for key, value in losses.items()}
    return losses["loss"], time_taken, losses, (targets, preds, logits, miss_types, modality_logits, sample_ids, rec_logits)


def _train_loop(
    config: CMAMConfig,
    cmam: MultimodalModelProtocol,
    model: MultimodalModelProtocol,
    dataloaders: Dict[str, DataLoader],
    optimizer: Optimizer,
    loss_functions: LossFunctionGroup,
    device: torch.device,
    metric_recorder: MetricRecorder,
    checkpoint_manager: CheckpointManager,
    scheduler: Optional[LRScheduler] = None,
    experiment_data: Optional[Dict[str, Any]] = None,
    monitor: Optional[ExperimentMonitor] = None,
    checkpoint_mode: Literal["minimize", "maximize"] = "minimize",
) -> Dict[str, Any]:
    """
    Perform the training loop over all epochs.

    Args:
        config (CMAMConfig): Experiment configuration.
        model (MultimodalModelProtocol): Model to train.
        dataloaders (Dict[str, DataLoader]): Data loaders for train/validation splits.
        optimizer (Optimizer): Optimizer for model parameters.
        loss_functions (LossFunctionGroup): Loss function group.
        device (torch.device): Device to train on.
        metric_recorder (MetricRecorder): Recorder for metrics.
        checkpoint_manager (CheckpointManager): Manager for saving/loading checkpoints.
        scheduler (Optional[LRScheduler]): Learning rate scheduler.
        experiment_data (Optional[Dict[str, Any]]): Dictionary to store experiment data.
        monitor (Optional[ExperimentMonitor]): Optional experiment monitor.

    Returns:
        Dict[str, Any]: Dictionary containing the best metrics achieved during training.
    """
    best_metrics = None
    wait = 0
    console.start_task("Epoch", total=config.training.epochs)
    console.print(loss_functions)

    for epoch in range(1, config.training.epochs + 1):
        if monitor:
            monitor.start_epoch(epoch)

        metric_recorder.reset()
        train_loss, train_time, train_loss_info = train_epoch(
            cmam=cmam,
            model=model,
            train_loader=dataloaders["train"],
            optimizer=optimizer,
            loss_functions=loss_functions,
            device=device,
            epoch=epoch,
            metric_recorder=metric_recorder,
            monitor=monitor,
        )
        train_metrics: Dict[str, Dict[str, float]] = metric_recorder.calculate_all_groups(epoch=epoch, loss=train_loss)

        train_metrics["loss"] = train_loss
        experiment_data["metrics_history"]["train"].append(train_metrics.copy())
        experiment_data["timing_history"]["train"].append(train_time)


        for l, l_value in train_loss_info.items():
            if "losses" not in train_metrics:
                train_metrics["losses"] = {}    
            train_metrics["losses"][l] = l_value
        
        for group in train_metrics:
            if isinstance(train_metrics[group], dict):
                console.rule(f"Train - {group}")
                console.display_validation_metrics(train_metrics[group])
            else:
                console.print(f"[bold green]{group}[/] - {round(train_metrics[group], 5)}")



        metric_recorder.reset()
        val_loss, val_time, val_loss_info, _ = validate_epoch(
            cmam=cmam,
            model=model,
            val_loader=dataloaders["validation"],
            loss_functions=loss_functions,
            device=device,
            console=console,
            metric_recorder=metric_recorder,
            monitor=monitor,
            task_name="Validation",
        )
        val_metrics = metric_recorder.calculate_all_groups(epoch=epoch, loss=val_loss)
        val_metrics["loss"] = val_loss

        for l, l_value in val_loss_info.items():
            if "losses" not in val_metrics:
                val_metrics["losses"] = {}    
            val_metrics["losses"][l] = l_value

        experiment_data["metrics_history"]["validation"].append(val_metrics.copy())
        experiment_data["timing_history"]["validation"].append(val_time)

        for group in val_metrics:
            if isinstance(val_metrics[group], dict):
                console.rule(f"Validation - {group}")
                console.display_validation_metrics(val_metrics[group])

        if metric_recorder.writer is not None:
            for loss_name in train_loss_info:
                logger.debug(f"Logging {loss_name} loss")
                train_value = train_loss_info[loss_name]
                try:
                    val_value = val_loss_info[loss_name]
                except KeyError as ke:
                    console.print(ke)
                    console.print(val_loss_info.keys())
                    exit(-1)
                metric_recorder.writer.add_scalars(
                    f"{loss_name} Loss", {"Train": train_value, "Validation": val_value}, epoch
                )

        is_best, should_continue, wait = check_early_stopping(
            val_metrics=val_metrics,
            best_metrics=best_metrics,
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            wait=wait,
            mode=checkpoint_mode,
            target_metric=config.logging.save_metric,
        )

        if is_best:
            best_metrics = val_metrics.copy()
            checkpoint_manager.save_checkpoint(
                model=cmam,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                is_best=True,
            )
            console.print(f"[green]>> New best model saved at epoch {epoch}[/]")

        config_do_early_stopping: bool = config.training.early_stopping

        ## Only stop early if the config says to do so AND the check_early_stopping function says to do so.
        if config_do_early_stopping and not should_continue:
            console.print("[bold red]Early stopping triggered. Stopping training.[/]")
            break

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()
            console.print(f"[grey] - Learning rate: {optimizer.param_groups[0]['lr']:.2e}][/]")

        console.update_task("Epoch", advance=1)
        if monitor:
            monitor.end_epoch()

    console.complete_task("Epoch")
    return best_metrics


def test(
    cmam: MultimodalModelProtocol,
    model: MultimodalModelProtocol,
    dataloaders: Dict[str, DataLoader],
    loss_functions: LossFunctionGroup,
    device: torch.device,
    metric_recorder: MetricRecorder,
    checkpoint_manager: CheckpointManager,
    experiment_data: Optional[Dict[str, Any]] = None,
    monitor: Optional[ExperimentMonitor] = None,
    metrics_out_fmt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform testing on the model using the specified data loaders.

    Args:
        model (Module): Model to test.
        dataloaders (Dict[str, DataLoader]): Data loaders for testing splits.
        loss_functions (LossFunctionGroup): Loss function group.
        device (torch.device): Device for computation.
        metric_recorder (MetricRecorder): Recorder for metrics.
        checkpoint_manager (CheckpointManager): Manager for loading the best checkpoint.
        experiment_data (Optional[Dict[str, Any]]): Experiment data storage dictionary.
        monitor (Optional[ExperimentMonitor]): Optional experiment monitor.

    Returns:
        Dict[str, Any]: Metrics recorded during testing.
    """
    # Load checkpoint for testing (but skip if already loaded in setup_model_components)
    # Check if the model has already been loaded by seeing if weights are non-random
    param_sum = sum(p.sum().item() for p in cmam.parameters())
    if abs(param_sum) < 1e-6:  # Likely random initialization
        logger.info("C-MAM model appears to have random weights, loading checkpoint...")
        checkpoint_manager.load_checkpoint(model=cmam, load_best=True)
    else:
        logger.info("C-MAM model appears to already have loaded weights, skipping checkpoint loading.")
    
    # Validate model state for evaluation consistency
    _validate_model_state_for_evaluation(cmam, model, device)
    
    # Set deterministic seeds for evaluation consistency
    _set_deterministic_evaluation_seeds()

    for split_name, loader in dataloaders.items():
        if split_name in ["train", "validation", "embeddings"]:
            continue

        metric_recorder.reset()
        console.print(f"\n[bold cyan]Testing on {split_name} split[/]")

        with torch.no_grad():
            test_loss, test_time, test_loss_info, (targets, preds, logits, miss_types, modality_logits, sample_ids, rec_logits) = validate_epoch(
                cmam=cmam,
                model=model,
                val_loader=loader,
                loss_functions=loss_functions,
                device=device,
                console=console,
                metric_recorder=metric_recorder,
                monitor=monitor,
                task_name=f"Testing {split_name}",
            )

        unique_masks = np.unique(miss_types)                 # e.g. ['av', 'a', 'v']
        console.print(f"Unique miss types: {unique_masks}")


        metrics = metric_recorder.calculate_all_groups(loss=test_loss, skip_tensorboard=False)

        metrics.update({k: np.mean(v) for k, v in test_loss_info.items()})
        experiment_data["metrics_history"][split_name] = metrics
        experiment_data["timing_history"][split_name] = [test_time]

    if dataloaders.get("embeddings", None):
        console.print("[bold cyan]Generating embeddings[/]")
        out_fp = str(metrics_out_fmt)
        checkpoint_manager.load_checkpoint(model=cmam, load_best=True)
        cmam.get_embeddings(dataloader=dataloaders["embeddings"],device=device,  out_fp=out_fp, trained_model=model)

    if metrics_out_fmt:
        metrics_out_fmt = str(metrics_out_fmt)
        targets_path = metrics_out_fmt.format(targ="targets")
        preds_path = metrics_out_fmt.format(targ="preds")
        logits_path = metrics_out_fmt.format(targ="logits")
        rec_logits_path = metrics_out_fmt.format(targ="rec_logits")
        miss_types_path = metrics_out_fmt.format(targ="miss_types")
        timing = metrics_out_fmt.format(targ="params_timing")
        timing = Path(timing).with_suffix(".txt")
        os.makedirs(timing.parent, exist_ok=True)
        with open(timing, "w+") as f:
            parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"parameter_count: {parameter_count}\n")
            f.write(f"test_time: {test_time}")
            console.print(f"> Saved test timing ({test_time}s) to {timing}")

        # console.print(f"Targets: {targets.shape}")


        b_logits = defaultdict(list)
        for batch_logits in modality_logits:
            for m in batch_logits:
                b_logits[m].append(batch_logits[m])
        b_logits = {
            k: np.concat(v) for k, v in b_logits.items()
        }

        for m, l in b_logits.items():
            path = metrics_out_fmt.format(targ=f"{str(m).lower()}_logits")
            print(path)
            np.save(path,l)
            console.print(f"> Saved logits to {path}")

        np.save(targets_path, targets)
        np.save(preds_path, preds)
        np.save(logits_path, logits)
        np.save(rec_logits_path, rec_logits)
        # targets / ids -----------------------------------------------
        fmt = str(metrics_out_fmt)                       # Path pattern

        # np.save(fmt.format(targ=f"targets_{canonical}"), targets_algn)
        # np.save(fmt.format(targ="sample_ids"),           order_ids)
        # # logits per mask ---------------------------------------------
        # for m, arr in aligned_logits.items():
        #     np.save(fmt.format(targ=f"logits_{m}"), arr)
        #     console.print(f"> Saved logits_{m}.npy  shape {arr.shape}")

        # # np.save(miss_types_path, miss_types)
        with open(Path(miss_types_path).with_suffix(".json"), "w+") as jfp:
            json.dump(miss_types, jfp)

        metrics_path = Path(metrics_out_fmt.format(targ="metrics")).with_suffix(".json")

        # write metrics
        with open(metrics_path, "w+") as jfp:
            json.dump(prepare_metrics_for_json([metrics]), jfp)

        # Save evaluation metadata for consistency tracking
        eval_metadata = {
            "cmam_state_hash": CheckpointManager.calculate_model_state_hash(cmam.state_dict()),
            "model_state_hash": CheckpointManager.calculate_model_state_hash(model.state_dict()),
            "evaluation_timestamp": time.time(),
            "targets_shape": list(targets.shape),
            "predictions_shape": list(preds.shape),
            "logits_shape": list(logits.shape),
            "rec_logits_shape": list(rec_logits.shape),
            "num_miss_types": len(miss_types),
            "unique_miss_types": list(np.unique(miss_types)),
        }
        
        eval_metadata_path = Path(metrics_out_fmt.format(targ="evaluation_metadata")).with_suffix(".json")
        with open(eval_metadata_path, "w+") as jfp:
            json.dump(eval_metadata, jfp, indent=2)

        console.print(f"> Saved targets with shape {targets.shape} to {targets_path}")
        console.print(f"> Saved preds with shape {preds.shape} to {preds_path}")
        console.print(f"> Saved logits with shape {logits.shape} to {logits_path}")
        console.print(f"> Saved rec_logits with shape {rec_logits.shape} to {rec_logits_path}")
        console.print(f"> Saved miss_types with len {len(miss_types)} to {miss_types_path}")
        console.print(f"> Saved evaluation metadata to {eval_metadata_path}")

    console.display_validation_metrics(metrics)

    for group in metrics:
        if isinstance(metrics[group], dict):
            console.rule(f"Test - {group}")
            console.display_validation_metrics(metrics[group])

    return experiment_data["metrics_history"]


def _validate_model_state_for_evaluation(cmam: MultimodalModelProtocol, model: MultimodalModelProtocol, device: torch.device) -> None:
    """Validate model state before evaluation to catch potential issues."""
    logger.info("Validating model state for evaluation...")
    
    # Check if models are on the correct device
    cmam_device = next(cmam.parameters()).device
    model_device = next(model.parameters()).device
    
    if cmam_device != device:
        logger.warning(f"C-MAM model is on {cmam_device} but expected {device}")
        console.print(f"[yellow]Warning:[/] C-MAM model device mismatch: {cmam_device} vs {device}")
    
    if model_device != device:
        logger.warning(f"Base model is on {model_device} but expected {device}")
        console.print(f"[yellow]Warning:[/] Base model device mismatch: {model_device} vs {device}")
    
    # Check if models are in eval mode
    if cmam.training:
        logger.warning("C-MAM model is in training mode, setting to eval mode")
        console.print("[yellow]Warning:[/] C-MAM model was in training mode, switching to eval")
        cmam.eval()
    
    if model.training:
        logger.warning("Base model is in training mode, setting to eval mode")
        console.print("[yellow]Warning:[/] Base model was in training mode, switching to eval")
        model.eval()
    
    # Calculate and log model state hashes for reproducibility tracking
    cmam_state_hash = CheckpointManager.calculate_model_state_hash(cmam.state_dict())
    model_state_hash = CheckpointManager.calculate_model_state_hash(model.state_dict())
    
    logger.info(f"Model state validation complete:")
    logger.info(f"  C-MAM state hash: {cmam_state_hash}")
    logger.info(f"  Base model state hash: {model_state_hash}")
    
    console.print(f"[green]✓[/] Model state validation complete")
    console.print(f"  C-MAM state hash: {cmam_state_hash[:16]}...")
    console.print(f"  Base model state hash: {model_state_hash[:16]}...")


def _set_deterministic_evaluation_seeds() -> None:
    """Set deterministic seeds for evaluation consistency."""
    import random
    
    # Set fixed seeds for evaluation reproducibility
    eval_seed = 42  # Fixed seed for evaluation
    
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)
    random.seed(eval_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(eval_seed)
        torch.cuda.manual_seed_all(eval_seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set deterministic evaluation seeds to {eval_seed}")
    console.print(f"[green]✓[/] Set deterministic evaluation seeds")


def main(
    config: CMAMConfig,
) -> Tuple[Module, Dict[str, Any], Path]:
    """
    Main function to perform training, validation, and testing.

    Args:
        config (CMAMConfig): Experiment configuration.

    Returns:
        Tuple[Module, Dict[str, Any], Path]: Final trained model, experiment data, and output directory.
    """
    # Setup output directory
    output_dir = Path(config.logging.log_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean old checkpoints
    logger.debug("Cleaning up old checkpoints...")
    clean_checkpoints(os.path.join(os.path.dirname(config.logging.model_output_path), str(config.experiment.run_id)))

    # Setup components
    dataloaders = setup_dataloaders(config)
    cmam, model, optimizer, loss_functions, scheduler, device, metric_recorder = setup_model_components(
        config=config, dataloaders=dataloaders
    )
    checkpoint_manager, experiment_data, report_generator, monitor = setup_tracking(
        config=config, output_dir=output_dir, model=model
    )

    experiment_data["model_info"]["architecture"] = str(model)
    experiment_data["model_info"]["parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    experiment_data["model_info"]["size"] = (
        sum(p.numel() for p in model.parameters()) * PARAMETER_SIZE_BYTES / 1024 / 1024
    )  # in MB

    experiment_data["cmam_model_info"] = {}
    experiment_data["cmam_model_info"]["architecture"] = str(cmam)
    experiment_data["cmam_model_info"]["parameters"] = sum(p.numel() for p in cmam.parameters() if p.requires_grad)
    experiment_data["cmam_model_info"]["size"] = (
        sum(p.numel() for p in cmam.parameters()) * PARAMETER_SIZE_BYTES / 1024 / 1024
    )  # in MB

    console.print("C-MAM Model Parameter Count Breakdown")
    console.print(f"{cmam.display()}")

    console.print_table(
        "Model Information",
        columns=["Parameter Size", "Trainable Parameters"],
        rows=[
            [
                f"{experiment_data['model_info']['size']:.2f} MB",
                f"{experiment_data['model_info']['parameters']:,}",
            ],
            [
                f"{experiment_data['cmam_model_info']['size']:.2f} MB",
                f"{experiment_data['cmam_model_info']['parameters']:,}",
            ],
        ],
    )
    gpu_mem_info = gpu_memory().replace("\t", " ")
    logger.info(gpu_mem_info)
    console.print(gpu_mem_info)
    console.print(f"[bold green]Model Parameters:[/] [blue]{experiment_data['model_info']['parameters']:,}[/]")
    console.print(count_parameters(cmam))

    if config.experiment.dry_run:
        console.print("[yellow]Dry run completed. Exiting.[/]")
        return model, experiment_data, output_dir

    try:
        # Training
        if config.experiment.is_train:
            _train_loop(
                cmam=cmam,
                config=config,
                model=model,
                dataloaders=dataloaders,
                optimizer=optimizer,
                loss_functions=loss_functions,
                device=device,
                metric_recorder=metric_recorder,
                checkpoint_manager=checkpoint_manager,
                scheduler=scheduler,
                experiment_data=experiment_data,
                monitor=monitor,
                checkpoint_mode="minimize" if config.logging.save_metric == "loss" else "maximize",
            )

        # Testing
        if config.experiment.is_test:
            test(
                cmam=cmam,
                model=model,
                dataloaders=dataloaders,
                loss_functions=loss_functions,
                device=device,
                metric_recorder=metric_recorder,
                checkpoint_manager=checkpoint_manager,
                experiment_data=experiment_data,
                monitor=monitor,
                metrics_out_fmt=config.logging.metrics_path / "test_{targ}.npy",
            )

            # if "embeddings" in dataloaders and hasattr(cmam,"get_embeddings"):
            #     console.print("[bold cyan]Generating embeddings for visualization...[/]")
            #     embeddings = cmam.get_embeddings(dataloaders["embeddings"], trained_model=model, device=device)
            #     labels = embeddings[cmam.labels_key]
            #     rec_embds = safe_detach(embeddings["rec_embd"])
            #     target_embds = safe_detach(embeddings["target_embd"])
            #     labels = safe_detach(labels)
            #     experiment_data["embeddings"] = {"labels": labels, "rec_embd": rec_embds, "target_embd": target_embds}
            #     save_fp = config.logging.metrics_path / "embeddings" / f"{cmam.target_modality}_rec_embeddings.npy"
            #     console.print(f"[green]✓[/] Saved embeddings to: {save_fp}")
            #     np.save(save_fp, rec_embds)

            #     save_fp = config.logging.metrics_path / "embeddings" / f"{cmam.target_modality}_target_embeddings.npy"
            #     np.save(save_fp, target_embds)
            #     console.print(f"[green]✓[/] Saved embeddings to: {save_fp}")

    finally:
        if monitor:
            monitor.close()
            model.detach_monitor()

    # # Generate final report
    report_path = report_generator.generate_report(experiment_data)
    console.print(f"[green]Experiment completed. Report saved at: {report_path}[/]")

    return model, experiment_data, output_dir


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a multimodal model and evaluate using missing data imputation.")

    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--run_id", type=int, default=-1, help="The run ID for this experiment.")

    optional_args = parser.add_argument_group("Optional arguments")
    optional_args.add_argument("--dry-run", action="store_true", help="Run a dry run of the experiment.")
    optional_args.add_argument("--skip-train", action="store_true", help="Skip training phase.")
    optional_args.add_argument("--skip-test", action="store_true", help="Skip testing phase.")
    optional_args.add_argument(
        "--disable_monitoring", action="store_true", help="Disable monitoring of model weights and gradients."
    )

    args = parser.parse_args()

    # Setup experiment
    config = setup_experiment(args.config, args.run_id)
    config.experiment.dry_run = args.dry_run
    config.experiment.is_train = not args.skip_train
    config.experiment.is_test = not args.skip_test

    if args.disable_monitoring:
        config.monitoring.enabled = False

    # if config.experiment.cross_validation:
    #     main_cross_validation(config)
    # else:
    main(config)

    shutdown_cursor_reset_hook()
