import os
from pathlib import Path
from pprint import pformat
import random
import time
import warnings
from argparse import ArgumentParser
from typing import Literal, Optional

import numpy as np
import rich.box as box
import torch
from config.logging_config import LoggingConfig
from config.cmam_config import CMAMConfig
from config.federated_configs import FedDataConfig, FederatedExperimentConfig
from config.metric_config import MetricConfig
from config.model_config import ModelConfig
from config.multimodal_training_config import TrainingConfig
from config.resolvers import resolve_init_fn, resolve_model_name
from data.base_dataset import MultimodalBaseDataset
from experiment_utils.checkpoints import CheckpointManager
from experiment_utils.global_state import set_current_run_id
from experiment_utils.logging import configure_logger, get_logger
from experiment_utils.metric_recorder import MetricRecorder
from experiment_utils.printing import get_console, print_error, print_info, print_success, print_warning
from experiment_utils.utils import ensure_dir
from models.cmams import SimpleCMAM
from models.protocols import MultimodalModelProtocol
from rich.panel import Panel
from torch.utils.data import DataLoader

import random
import logging
from collections import Counter
from fed import DataSplitType
from fed.client import Client, ClientCMAMData, ClientExpData, ClientModelData
from fed.data_utils import FederatedDataset, split_global_client
from fed.global_model import FederatedTrainer, GlobalCMAMData, GlobalModelData

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

console = get_console()
logger = get_logger()
DEBUG = False  # Set to True to enable debug logging


def get_all_possible_patterns(full_modality: str) -> list[str]:
    """Generate all possible modality combinations excluding empty set from a full modality string.

    E.g. for full_modality = "abc", the patterns will be:
    - "a"
    - "b"
    - "c"
    - "ab"
    - "ac"
    - "bc"
    - "abc"
    """

    from itertools import combinations

    patterns = []
    n = len(full_modality)
    for r in range(1, n + 1):
        for combo in combinations(full_modality, r):
            patterns.append("".join(combo))
    return sorted(patterns, key=lambda x: (len(x), x))


def setup_experiment(config_path: str, run_id: int, seed: Optional[int] = None) -> FederatedExperimentConfig:
    """
    Setup the federated experiment configuration.

    Args:
        config_path (str): Path to the configuration file.
        run_id (int): The run ID for this experiment.

    Returns:
        FederatedConfig: The configured federated experiment.
    """
    global logger
    global console

    # Load the configuration from the specified path
    config = FederatedExperimentConfig.load(config_path, run_id=run_id, seed=seed)

    # Log the configuration
    configure_logger(log_path=config.logging.log_path, suffix="")
    logger = get_logger()

    logger.info(f"Starting experiment with run ID {run_id}")
    logger.info(f"Configuration:\n{config}")
    # console.rule(f"Starting experiment with run ID {run_id}")
    console = get_console()
    return config

def assign_modalities(
    num_clients: int,
    modality_conditions: list[str],
    guarantee_at_least_n_full_instances: int = 1,
    guarantee_at_least_n_per_condition: int = 1,
    max_instances_per_condition: dict[str, int] | None = None,
) -> list[str]:
    """
    Assigns clients to modality conditions with guarantees and constraints.

    Guarantees:
        - Each condition appears at least `guarantee_at_least_n_per_condition` times.
        - Full modality appears at least `guarantee_at_least_n_full_instances` times.

    Constraints:
        - No condition exceeds `max_instances_per_condition` if provided.

    Returns:
        Assigned modality conditions for each client.
    """
    full_modality = max(modality_conditions, key=len)
    full_modality_set = set(full_modality)
    max_modalities = len(full_modality_set)

    if guarantee_at_least_n_per_condition * len(modality_conditions) > num_clients:
        raise ValueError("Not enough clients to guarantee minimum instances for each condition.")

    if max_instances_per_condition:
        for cond in modality_conditions:
            min_required = max(
                guarantee_at_least_n_per_condition,
                guarantee_at_least_n_full_instances if cond == full_modality else 0
            )
            max_allowed = max_instances_per_condition.get(cond, num_clients)
            if min_required > max_allowed:
                raise ValueError(f"Condition '{cond}' has min required {min_required} but max allowed {max_allowed}.")

    # Step 1: Start with guaranteed counts
    assignments = []
    condition_counts = Counter()

    for cond in modality_conditions:
        count = guarantee_at_least_n_per_condition
        assignments.extend([cond] * count)
        condition_counts[cond] += count

    # Step 2: Add extra full modality if needed
    full_modality_count = condition_counts[full_modality]
    if full_modality_count < guarantee_at_least_n_full_instances:
        extra_needed = guarantee_at_least_n_full_instances - full_modality_count
        assignments.extend([full_modality] * extra_needed)
        condition_counts[full_modality] += extra_needed

    # Step 3: Fill remaining clients
    remaining = num_clients - len(assignments)
    if remaining < 0:
        raise ValueError("Too many guaranteed instances for available clients.")

    # Set up weighted probabilities
    weights = []
    for cond in modality_conditions:
        missing = max_modalities - len(cond)
        weights.append((missing + 1) * 2)
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    # Sampling under constraints
    while remaining > 0:
        cond = random.choices(modality_conditions, weights=probabilities, k=1)[0]
        current_count = condition_counts[cond]
        max_allowed = max_instances_per_condition.get(cond, num_clients) if max_instances_per_condition else num_clients
        if current_count < max_allowed:
            assignments.append(cond)
            condition_counts[cond] += 1
            remaining -= 1
        # else skip and retry

    # Shuffle and swap for debug
    random.shuffle(assignments)
    if full_modality in assignments:
        first_full_modality_index = assignments.index(full_modality)
        assignments[0], assignments[first_full_modality_index] = assignments[first_full_modality_index], assignments[0]
        logger.info(f"Swapped full modality '{full_modality}' with client 1 for debugging purposes.")
        print_info(console, f"Swapped full modality '{full_modality}' with client 1 for debugging purposes.")

    # Final report
    logger.info(f"Assignment counts: {dict(Counter(assignments))}")
    logger.info(f"Probabilities: {probabilities}")
    for modality, prob in zip(modality_conditions, probabilities):
        logger.info(f"  Modality '{modality}': {prob:.2%} probability")
        print_info(console, f"  Modality '{modality}': {prob:.2%} probability")

    return assignments


def generate_modality_assignments(
    num_clients: int,
    modalities: list[str] | str,
    method: Literal["congruent", "incongruent"] = "congruent",
    save_to_disk: bool = False,
) -> tuple[list[str], str]:
    if isinstance(modalities, str):
        full_modality = modalities
        modalities = [modalities]  # Convert to list for consistency
    elif isinstance(modalities, list):
        if len(modalities) == 0:
            raise ValueError("Modalities list cannot be empty.")
        full_modality = max(modalities, key=len)  # Use the longest modality name as the full modality

    # in the congruent setting, each client has the same full modality set
    if method == "congruent":
        print_info(console, f"Using congruent modality distribution with full modality: {full_modality}")
        assignments = [full_modality] * num_clients
        print_info(console, f"Len assignments: {len(assignments)}")
        print_info(console, f"Full modality: {full_modality}")
        assignment_counts = {full_modality: num_clients}

        if save_to_disk:
            # Save the full modality to a file for reproducibility
            ensure_dir(config.logging.log_path / "modality_assignments")
            with open(config.logging.log_path / "modality_assignments" / "full_modality.txt", "w") as f:
                f.write(full_modality)

            with open(config.logging.log_path / "modality_assignments" / "congruent_assignments.txt", "w") as f:
                for i in range(num_clients):
                    f.write(f"Client {i + 1}: {assignments[i]}\n")
                for pattern, count in assignment_counts.items():
                    print_info(console, f"  Pattern '{pattern}': {count} clients")
                f.write(f"Full modality: {full_modality}\n")
                f.write(f"Total clients: {num_clients}\n")
                f.write(f"Assignment counts: {pformat(assignment_counts)}\n")
                f.write(f"Method: {method}\n")

        # Log the distribution for debugging
        print_info(console, "Congruent assignment distribution:")
        for pattern, count in assignment_counts.items():
            print_info(console, f"  Pattern '{pattern}': {count} clients")
        return assignments, full_modality

    elif method == "incongruent":
        print_info(console, f"Using incongruent modality distribution with modalities: {modalities}")
        print_info(console, f"Full modality (complete set): {full_modality}")

        # For incongruent training, use the original assign_modalities algorithm
        # This gives higher weight to clients with more missing modalities
        assignments = assign_modalities(num_clients, modalities, guarantee_at_least_n_full_instances=1, guarantee_at_least_n_per_condition=2, max_instances_per_condition={"at": 3, "av": 3, "tv": 3, "a": 4, "v": 4, "t": 4})

        # Log the distribution for debugging
        assignment_counts = {}
        for assignment in assignments:
            assignment_counts[assignment] = assignment_counts.get(assignment, 0) + 1

        print_info(console, "Incongruent assignment distribution:")
        for pattern, count in assignment_counts.items():
            print_info(console, f"  Pattern '{pattern}': {count} clients")

        if save_to_disk:
            # Save the full modality to a file for reproducibility
            ensure_dir(config.logging.log_path / "modality_assignments")
            with open(config.logging.log_path / "modality_assignments" / "full_modality.txt", "w") as f:
                f.write(full_modality)

            with open(config.logging.log_path / "modality_assignments" / "incongruent_assignments.txt", "w") as f:
                for i in range(num_clients):
                    f.write(f"Client {i + 1}: {assignments[i]}\n")
                for pattern, count in assignment_counts.items():
                    print_info(console, f"  Pattern '{pattern}': {count} clients")
                f.write(f"Full modality: {full_modality}\n")
                f.write(f"Total clients: {num_clients}\n")
                f.write(f"Assignment counts: {pformat(assignment_counts)}\n")
                f.write(f"Method: {method}\n")

        return assignments, full_modality
    else:
        raise ValueError(
            f"Unknown modality distribution method: {method}. Supported methods are 'congruent' and 'incongruent'."
        )


def create_checkpoint_manager_for_client(
    config,
    client_id: int,
    save_metric: str,
    available_modalities: str | list[str],
    device: str = "cpu",
    cmams=True,
) -> CheckpointManager:
    client_model_output_path = config.logging.model_output_path / f"client_{client_id}"
    if "round_0" in str(client_model_output_path):
        raise ValueError(
            "Client model output path should not contain 'round_0'. This is likely due to a misconfiguration in the logging path."
        )
    if cmams:
        client_model_output_path = Path(os.path.join(client_model_output_path, "cmams", "-".join(available_modalities) if isinstance(available_modalities, list) else available_modalities))
        print_info(
            console,
            f"Creating checkpoint manager for client {client_id} with C-MAMs at {client_model_output_path}",
        )
        logger.info(
            f"Creating checkpoint manager for client {client_id} with C-MAMs at {client_model_output_path}"
        )

    ensure_dir(client_model_output_path)
    # logger.info(f"Creating checkpoint manager for client {client_id} at {client_model_output_path}")
    # print_info(console, f"Creating checkpoint manager for client {client_id} at {client_model_output_path}")

    checkpoint_manager = CheckpointManager(
        model_dir=client_model_output_path,
        save_metric=save_metric,
        mode="minimize" if save_metric == "loss" else "maximize",
        device=device,
    )

    logger.info(f"Checkpoint manager created for client {client_id} at {client_model_output_path}")
    print_success(console, f"Checkpoint manager created for client {client_id} at {client_model_output_path}")
    return checkpoint_manager


def create_client_cmam_data(
    cmam_configs: dict[str, CMAMConfig],
    cmam_training_configs: TrainingConfig | dict[str, TrainingConfig],
    modality_assignment: str,
    all_modalities: list[str],
    full_modality: str,
    training_method: str = "congruent",
) -> list[ClientCMAMData]:
    """
    Create C-MAM models, losses, and optimizers for a client based on their modality assignment.

    Args:
        cmam_configs (dict[str, CMAMConfig]): The available C-MAM configurations.
        cmam_training_configs (TrainingConfig | dict[str, TrainingConfig]): Training config(s) for C-MAMs.
        modality_assignment (str): The assigned modality condition.
        all_modalities (list[str]): All available modality patterns.
        full_modality (str): The complete modality pattern.
        training_method (str): Either "congruent" or "incongruent".

    Returns:
        list[ClientCMAMData]: List of C-MAM data objects for the client.
    """
    logger.info(f"Creating C-MAM(s) for modality assignment: {modality_assignment} (method: {training_method})")
    print_info(console, f"Creating C-MAM(s) for modality assignment: {modality_assignment} (method: {training_method})")

    cmam_assignments = []

    if modality_assignment == full_modality:
        # Client has all modalities - can train all available C-MAMs
        logger.info("Client has full modality set -- can train all C-MAMs")
        cmam_assignments.extend([m for m in all_modalities if m != full_modality])
    elif training_method == "congruent":
        # Congruent training: each client gets the same C-MAM configuration
        if modality_assignment not in cmam_configs:
            msg = f"No C-MAM config found for modality assignment: {modality_assignment}"
            logger.error(msg)
            print_error(console, msg)
            raise ValueError(msg)
        cmam_assignments.append(modality_assignment)
    elif training_method == "incongruent":
        # Incongruent training: use the same logic as congruent but with different modality assignments
        # The key difference is that clients have different modality_assignment values
        if modality_assignment not in cmam_configs:
            # If exact assignment not found, try to find compatible C-MAMs for this client
            logger.warning(f"No exact C-MAM config found for modality assignment: {modality_assignment}")
            print_warning(console, f"No exact C-MAM config found for modality assignment: {modality_assignment}")
            # For incongruent training, clients without exact matches may not get C-MAMs
            # This is expected behavior - not all client types need the same C-MAMs
        else:
            cmam_assignments.append(modality_assignment)
    else:
        raise ValueError(f"Unknown training method: {training_method}")

    # Remove duplicates while preserving order
    cmam_assignments = list(dict.fromkeys(cmam_assignments))

    print_info(console, f"Creating {len(cmam_assignments)} C-MAM(s) for assignments: {cmam_assignments}")

    if len(cmam_assignments) == 0:
        logger.warning(f"No valid C-MAM configurations found for modality assignment: {modality_assignment}")
        print_warning(console, f"No C-MAMs assigned to client with modality assignment: {modality_assignment}")
        return []

    cmams = []
    for i, assignment in enumerate(cmam_assignments, start=1):
        if assignment not in cmam_configs:
            msg = f"No C-MAM config found for assignment: {assignment}"
            logger.error(msg)
            print_error(console, msg)
            raise ValueError(msg)

        cmam_config = cmam_configs[assignment]
        cmam_model = SimpleCMAM(**cmam_config.kwargs)

        # Determine which training config to use
        if isinstance(cmam_training_configs, TrainingConfig):
            training_config = cmam_training_configs
        elif isinstance(cmam_training_configs, dict):
            training_config = TrainingConfig.from_dict(cmam_training_configs.get(assignment, cmam_training_configs))
        else:
            raise TypeError("C-MAM training config must be a TrainingConfig or a dict[str, TrainingConfig]")

        logger.debug(f"Using training config for C-MAM: {training_config}")

        print_info(
            console,
            f"Created C-MAM to reconstruct modality {cmam_model.target_modality} using {cmam_model.input_modalities}",
        )

        try:
            cmam_optimizer = training_config.get_optimizer(cmam_model)
            logger.debug(f"Created optimizer for C-MAM: {cmam_optimizer}")
            # print_success(console, "✓ C-MAM optimizer created")
        except Exception as e:
            msg = f"Failed to create optimizer for C-MAM: {e}"
            logger.exception(msg)
            print_error(console, msg)
            raise

        try:
            cmam_loss_function = training_config.get_loss_function()
            logger.debug(f"Created loss function for C-MAM: {cmam_loss_function}")
            # print_success(console, "✓ C-MAM loss function created")
        except Exception as e:
            msg = f"Failed to create loss function for C-MAM: {e}"
            logger.exception(msg)
            print_error(console, msg)
            raise

        save_metric = "loss"

        checkpoint_manager = create_checkpoint_manager_for_client(
            config=config,
            client_id=i,
            save_metric=save_metric,
            device="gpu" if torch.cuda.is_available() else "cpu",
            available_modalities=[str(m) for m in cmam_model.input_modalities],
            cmams=True,
        )

        cmam_data = ClientCMAMData(
            cmam=cmam_model,
            cmam_optimizer=cmam_optimizer,
            cmam_loss_function=cmam_loss_function,
            checkpoint_manager=checkpoint_manager,
        )
        # print_info(console, f"{cmam_model}")
        cmams.append(cmam_data)

    return cmams


def setup_clients(
    config: FederatedExperimentConfig,
    model_config: ModelConfig,
    datasets: dict[DataSplitType, FederatedDataset],
    metric_recorder: MetricRecorder,
    num_clients: int,
    client_modality_assignments: list[str],
    full_modality: str,
    device: str = "cpu",
    modality_distribution_method: Literal["congruent", "incongruent"] = "congruent",
    cmam_configs: Optional[dict[str, CMAMConfig]] = None,
    cmam_training_configs: Optional[TrainingConfig | dict[str, TrainingConfig]] = None,
) -> list[Client]:
    """
    Setup the clients for the federated experiment.
    Args:
        global_model (MultimodalModelProtocol): The global model to be used by clients.
        dataloaders (dict[DataSplitType, FedDataLoader]): The dataloaders for the experiment.
        metric_recorder (MetricRecorder): The metric recorder for logging metrics.
        num_clients (int): The number of clients to setup.
        cmam_config (Optional[CMAMConfig]): Configuration for C-MAMs if applicable.

    Returns:
        list[Client]: A list of configured clients for the federated experiment.
    """
    logger.debug("Setting up clients...")
    print_info(console, "Setting up clients...")
    clients = []
    datasets = {DataSplitType(split): dataset for split, dataset in datasets.items()}
    if DataSplitType.TRAIN in datasets:
        modalities = datasets[DataSplitType.TRAIN].all_possible_patterns
    elif DataSplitType.VALIDATION in datasets:
        modalities = datasets[DataSplitType.VALIDATION].all_possible_patterns
    elif DataSplitType.TEST in datasets:
        modalities = datasets[DataSplitType.TEST].all_possible_patterns
    else:
        raise ValueError(
            f"No valid data split found in dataloaders. Ensure at least one split is available. Available splits: {datasets.keys()}"
        )
    logger.info(f"Available modalities: {modalities}")
    print_success(console, f"Available modalities: {modalities}")
    print_info(console, f"Using modality distribution method: {modality_distribution_method}")

    for i in range(1, 1 + num_clients):
        local_metric_recorder = metric_recorder.clone()
        client_config = config.fed_config.client_training
        client_dataloaders = {}
        for split, dataset in datasets.items():
            print_info(console, f"Dataset patterns: {dataset.available_patterns}")
            split_batch_size = config.fed_data.datasets[split.value].client_batch_size
            if split_batch_size is None:
                logger.error(
                    f"Batch size not set for split '{split}'. Please set 'client_batch_size' in the dataset config."
                )
                print_error(
                    console,
                    f"Batch size not set for split '{split}'. Please set 'client_batch_size' in the dataset config.",
                )
                raise ValueError(f"Batch size not set for split '{split}'.")

            client_dataloaders[split] = dataset.get_client_dataloader(i - 1, batch_size=split_batch_size)

        model_cls: MultimodalModelProtocol = resolve_model_name(model_config.name)
        model: MultimodalModelProtocol = model_cls(**model_config.kwargs)
        print_info(console, f"{pformat(client_modality_assignments)}")
        print_info(console, f"CMAM configs: {cmam_configs}")
        client_cmam_data = None
        if cmam_configs is not None and cmam_training_configs is not None:
            print_info(
                console,
                f"Creating C-MAMs for client {i} with modality assignment: {client_modality_assignments[i - 1]}",
            )
            client_cmam_data = create_client_cmam_data(
                cmam_configs,
                cmam_training_configs,
                client_modality_assignments[i - 1],
                all_modalities=modalities,
                full_modality=full_modality,
                training_method=modality_distribution_method,
            )
            print_info(console, f"Created {len(client_cmam_data)} C-MAMs for client {i}")

        model_checkpoint_manager = create_checkpoint_manager_for_client(
            config=config, client_id=i, save_metric=config.save_metric, cmams=False, available_modalities= client_modality_assignments[i - 1]
        )

        if client_cmam_data and client_modality_assignments[i - 1] != full_modality:
            ## i.e. we are doing incongruent training using C-MAMs
            client_optimizer = client_config.get_optimizer([model, client_cmam_data[0].cmam])
            print_info(console, f"Created optimizer for client {i} with C-MAMs")
            client_loss_function = client_config.get_loss_function()
            client_scheduler = client_config.get_scheduler(client_optimizer)
            client_model_data = ClientModelData(
                model=model,
                optimizer=client_optimizer,
                loss_function=client_loss_function,
                scheduler=client_scheduler,
                checkpoint_manager=model_checkpoint_manager,
            )
        else:
            client_optimizer = client_config.get_optimizer(model)
            client_loss_function = client_config.get_loss_function()
            client_scheduler = client_config.get_scheduler(client_optimizer)
            client_model_data = ClientModelData(
                model=model,
                optimizer=client_optimizer,
                loss_function=client_loss_function,
                scheduler=client_scheduler,
                checkpoint_manager=model_checkpoint_manager,
            )

        experiment_data = {
            "metrics_history": {"train": [], "validation": [], "test": []},
            "timing_history": {"train": [], "validation": []},
        }

        client_exp_data = ClientExpData(
            config=config,
            experiment_data=experiment_data,
            metric_recorder=local_metric_recorder,
            device=device,
        )
        print_info(console, f"Client {i} has {dataset.available_patterns}")
        # Determine what modalities this client should track for embedding output
        tracked_modalities = []
        if client_cmam_data and not args.baseline:
            # If client has C-MAMs AND we're training C-MAMs, track the target modalities they reconstruct
            for cmam_data in client_cmam_data:
                target_mod = str(cmam_data.cmam.target_modality).lower()
                if target_mod not in tracked_modalities:
                    tracked_modalities.append(target_mod)
            print_info(console, f"Client {i} with modality assignment '{client_modality_assignments[i - 1]}' will track reconstructed modalities: {tracked_modalities}")
        else:
            # If no C-MAMs or baseline mode, track the available modalities
            tracked_modalities = [client_modality_assignments[i - 1]]
            print_info(console, f"Client {i} (baseline mode: {args.baseline}) tracking available modalities: {tracked_modalities}")
        
        client = Client(
            id=i,
            available_modalities=client_modality_assignments[i - 1],
            model=client_model_data,
            exp_data=client_exp_data,
            dataloaders=client_dataloaders,
            cmam_data=client_cmam_data,
            metric_recorder=local_metric_recorder,
            tracked_modalities=tracked_modalities,
            baseline=args.baseline,
            full_modality_pattern= full_modality,
        )
        clients.append(client)

    return clients


def setup_data(
    logging_config: LoggingConfig,
    config: FedDataConfig,
    num_clients:int,
    training: bool = True,
    testing: bool = True,
    method: Literal["congruent", "incongruent"] = "congruent",
) -> dict[DataSplitType:FederatedDataset]:
    """
    Setup the dataloaders for the federated experiment.

    Args:
        config (FedDataConfig): The configuration for the federated data.

    Returns:
        dict[DataSplitType, FedDataLoader]: A dictionary mapping data split types to their respective dataloaders.
    """
    logger.debug("Setting up dataloaders...")
    print_info(console, "Setting up dataloaders...")

    datasets: dict[str, MultimodalBaseDataset] = config.build_all_datasets()
    datasets = {DataSplitType(split): dataset for split, dataset in datasets.items()}

    distribution_strategy = config.distribution_strategy
    task_type = config.task_type
    global_fraction = config.global_fraction

    logger.info(f"Using distribution strategy: {distribution_strategy}")
    print_info(console, f"Using distribution strategy: {distribution_strategy}")

    logger.debug("Creating federated datasets...")
    print_info(console, "Creating federated datasets...")

    global_datasets = {}
    client_datasets = {}
    if DataSplitType.TRAIN in datasets:
        modalities = datasets[DataSplitType.TRAIN].get_all_possible_patterns()
    elif DataSplitType.VALIDATION in datasets:
        modalities = datasets[DataSplitType.VALIDATION].get_all_possible_patterns()
    elif DataSplitType.TEST in datasets:
        modalities = datasets[DataSplitType.TEST].get_all_possible_patterns()

    client_modality_assignments, full_modality = generate_modality_assignments(
        num_clients, modalities, method=method, save_to_disk=True
    )
    print_info(console, f"Client modality assignments: {client_modality_assignments}")

    for split, dataset in datasets.items():
        if not training and split == DataSplitType.TRAIN:
            logger.info(f"Skipping training split '{split}' as training is disabled.")
            print_warning(console, f"Skipping training split '{split}' as training is disabled.")
            continue

        if not testing and split == DataSplitType.TEST:
            logger.info(f"Skipping testing split '{split}' as testing is disabled.")
            print_warning(console, f"Skipping testing split '{split}' as testing is disabled.")
            continue
        global_split, clients_splits = split_global_client(
            base_dataset=dataset,
            num_clients=num_clients,
            distribution_strategy=distribution_strategy,
            task_type=task_type,
            global_fraction=global_fraction,
            alpha=config.alpha,
            global_split_strategy="stratified",
            client_patterns=client_modality_assignments,
            modality_distribution_method=method,
        )
        logger.info(f"Creating federated dataset for split '{split}' with {num_clients} clients.")
        print_info(console, f"Creating federated dataset for split '{split}' with {num_clients} clients.")
        print_info(console, f"Global split number of samples: {len(global_split)}")
        global_datasets[split] = global_split
        client_datasets[split] = clients_splits

        try:
            from pathlib import Path

            # Use the dataset name and split info for the experiment name
            experiment_name = f"federated_{split.value}_{distribution_strategy}_{method}"
            if hasattr(config, "alpha") and config.alpha is not None:
                experiment_name += f"_alpha{config.alpha}"

            metrics_output_dir = Path(logging_config.model_output_path) / "fed_metrics"
            ensure_dir(metrics_output_dir)
            metrics_file = clients_splits.save_split_metrics(
                output_dir=metrics_output_dir, experiment_name=experiment_name
            )
            logger.info(f"Dataset split metrics saved to: {metrics_file}")
            print_success(console, f"Dataset split metrics saved to: {metrics_file}")
        except Exception as e:
            logger.warning(f"Failed to compute dataset split metrics: {e}")
            print_error(console, f"Failed to compute dataset split metrics: {e}")
            raise Exception(
                f"Failed to compute dataset split metrics: {e}. Ensure the dataset supports metric computation."
            )

    logger.info("Federated datasets created successfully.")
    print_success(console, "Federated datasets created successfully.")
    return global_datasets, client_datasets, full_modality, client_modality_assignments


def setup_global_model_data(config: FederatedExperimentConfig) -> GlobalModelData:
    logger.info("Setting up global model data...")
    print_info(console, "Setting up global model data...")
    model_cls: MultimodalModelProtocol = resolve_model_name(config.model.name)
    model: MultimodalModelProtocol = model_cls(**config.model.kwargs)
    if config.model.init_fn is not None:
        init_fn = resolve_init_fn(config.model.init_fn)
        init_fn(model)
        logger.info(f"Initialized model with {config.model.init_fn}")
        print_success(console, f"Initialized model with {config.model.init_fn}")

    print_success(console, "Global model initialized successfully.")
    console.print(
        Panel(str(model), box=box.SQUARE, highlight=True, expand=True, title="[heading]Model Architecture[/]")
    )

    logger.info(f"Model: {model}")
    optimizer = config.fed_config.global_training.get_optimizer(model)
    criterion = config.fed_config.global_training.loss_functions

    print_success(console, "Optimizer and criterion created")
    logger.info(f"Optimizer and criterion created\n{optimizer}\n{criterion}")

    scheduler = None

    if config.fed_config.global_training.scheduler:
        scheduler = config.fed_config.global_training.get_scheduler(optimizer=optimizer)
        print_success(console, "Scheduler created")
        logger.info(f"Scheduler created\n{scheduler}")
    else:
        print_warning(console, "No scheduler")

    global_model_data = GlobalModelData(
        config=config,
        model=model,
        optimizer=optimizer,
        loss_function=criterion,
        scheduler=scheduler,
    )
    logger.info("Global model data setup complete.")
    print_success(console, "Global model data setup complete.")
    return global_model_data


def setup_global_cmams(config: FederatedExperimentConfig, device: str = "cpu") -> dict[str, GlobalCMAMData]:
    """
    Setup the global C-MAMs for the federated experiment.

    Args:
        config (FederatedExperimentConfig): The configuration for the federated experiment.
        device (str): The device to use for the C-MAMs.

    Returns:
        dict[Modality, GlobalCMAMData]: A dictionary mapping modalities to their respective C-MAM data.
    """
    logger.debug("Setting up global C-MAMs...")
    print_info(console, "Setting up global C-MAMs...")

    cmam_configs = config.cmams
    if cmam_configs is None or len(cmam_configs) == 0:
        logger.info("No C-MAM configurations provided. Skipping C-MAM setup.")
        print_warning(console, "No C-MAM configurations provided. Skipping C-MAM setup.")
        return {}

    global_cmam_training_configs = config.fed_config.global_cmam_training

    global_cmams = {}
    for modality, cmam_config in cmam_configs.items():
        logger.info(f"Setting up C-MAM for modality: {modality}")
        print_info(console, f"Setting up C-MAM for modality: {modality}")

        cmam_model = SimpleCMAM(**cmam_config.kwargs)

        if isinstance(global_cmam_training_configs, TrainingConfig):
            cmam_optimizer = global_cmam_training_configs.get_optimizer(cmam_model)
            cmam_loss_function = global_cmam_training_configs.get_loss_function()
        elif isinstance(global_cmam_training_configs, dict):
            cmam_training_config = global_cmam_training_configs.get(modality, global_cmam_training_configs)
            cmam_training_config = (
                TrainingConfig.from_dict(cmam_training_config)
                if isinstance(cmam_training_config, dict)
                else cmam_training_config
            )
            cmam_optimizer = cmam_training_config.get_optimizer(cmam_model)
            cmam_loss_function = cmam_training_config.get_loss_function()
        else:
            logger.error("Invalid C-MAM training configuration. Must be a TrainingConfig or a dict of TrainingConfig.")
            print_error(
                console, "Invalid C-MAM training configuration. Must be a TrainingConfig or a dict of TrainingConfig."
            )
            raise ValueError(
                "Invalid C-MAM training configuration. Must be a TrainingConfig or a dict of TrainingConfig."
            )

        inputs = "-".join([str(m) for m in cmam_model.input_modalities])
        modality = f"{inputs}_{str(cmam_model.target_modality).lower()}"

        logger.info(f"Creating C-MAM model for modality: {modality}")
        print_info(console, f"Creating C-MAM model for modality: {modality}")

        cmam_checkpoint_manager = CheckpointManager(
            model_dir=config.logging.model_output_path / "global_model" / "cmams" / modality,
            save_metric=config.fed_config.global_cmam_training.get(
                modality, config.fed_config.global_cmam_training
            ).get("save_metric", config.save_metric),
            mode="minimize"
            if config.fed_config.global_cmam_training.get(modality, config.fed_config.global_cmam_training).get(
                "save_metric", config.save_metric
            )
            == "loss"
            else "maximize",
            device=device,
        )

        modality_key = "".join(sorted([str(m)[0] for m in cmam_model.input_modalities])).lower()

        global_cmams[modality_key] = GlobalCMAMData(
            config=cmam_config,
            cmam=cmam_model,
            optimizer=cmam_optimizer,
            loss_function=cmam_loss_function,
            save_metric=config.fed_config.global_cmam_training.get(
                modality_key, config.fed_config.global_cmam_training
            ).get("save_metric", config.save_metric),
            checkpoint_manager=cmam_checkpoint_manager,
        )

        logger.info(f"Created C-MAM for modality: {modality_key}")
        print_success(console, f"Created C-MAM for modality: {modality_key}")
    if len(global_cmams) == 0:
        logger.warning("No C-MAMs were created. Ensure C-MAM configurations are provided.")
        print_warning(console, "No C-MAMs were created. Ensure C-MAM configurations are provided.")
    else:
        logger.info(f"Global C-MAMs setup complete with {len(global_cmams)} modalities.")
        print_success(console, f"Global C-MAMs setup complete with {len(global_cmams)} modalities.")
    return global_cmams


def setup_metric_recorder(metric_config: MetricConfig) -> MetricRecorder:
    """
    Setup the metric recorder for the federated experiment.

    Args:
        metric_config (MetricConfig): The configuration for metrics.

    Returns:
        MetricRecorder: The configured metric recorder.
    """
    logger.debug("Setting up metric recorder...")
    print_info(console, "Setting up metric recorder...")

    metric_recorder = MetricRecorder(metric_config)

    logger.info("Metric recorder setup complete.")
    print_success(console, "Metric recorder setup complete.")

    return metric_recorder


def main(args, config: FederatedExperimentConfig):
    set_current_run_id(config.experiment.run_id)
    # Clean old checkpoints
    logger.debug("Cleaning up old checkpoints...")
    print_info(console, f"Cleaning up old checkpoints for run ID {config.experiment.run_id}...")
    # Todo !: clean checkpoints per client and global model

    device = config.experiment.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA is not available on this system. Please check your CUDA installation.")
        print_error(console, "CUDA is not available on this system. Please check your CUDA installation.")
        raise RuntimeError("CUDA is not available on this system.")

    if device == "cpu" and torch.cuda.is_available():
        logger.warning("CUDA is available but 'cpu' device is selected. Consider using 'cuda' for better performance.")
        print_warning(
            console, "CUDA is available but 'cpu' device is selected. Consider using 'cuda' for better performance."
        )

    logger.info(f"Using device: {device}")
    print_info(console, f"Using device: {device}")

    # Setup components
    num_clients = config.num_clients
    training = config.experiment.is_train
    testing = config.experiment.is_test

    method = config.fed_config.modality_distribution_method

    global_datasets, client_datasets, full_modality, client_modality_assignments = setup_data(
        config.logging, config.fed_data, num_clients, training=training, testing=testing, method=method
    )

    try:
        if DataSplitType.VALIDATION in global_datasets:
            tracked_modalities = global_datasets[DataSplitType.VALIDATION].patterns
        elif DataSplitType.TEST in global_datasets:
            tracked_modalities = global_datasets[DataSplitType.TEST].patterns

    except KeyError:
        logger.warning("Validation split not found in global datasets. Using training split for validation.")
        print_warning(console, "Validation split not found in global datasets. Using training split for validation.")

        raise

    global_dataloaders = {}
    for split, dataset in global_datasets.items():
        dataset_config = config.fed_data.datasets[split.value]
        dataloader_args = dataset_config.get_dataloader_args()
        try:
            batch_size = dataloader_args["batch_size"]
        except KeyError as key_err:
            raise KeyError(
                f"Batch size not found in dataloader arguments for split '{split}'. Ensure 'batch_size' is set in the dataset configuration."
            ) from key_err

        if config.fed_data.use_collate_fn and hasattr(dataset, "collate_fn"):
            logger.info(f"Using custom collate_fn for {split} split")
            print_info(console, f"[yellow]Using custom collate_fn for {split} split[/]")
            dataloader_args["collate_fn"] = dataset.collate_fn
        try:
            del dataloader_args["use_collate_fn"]
        except KeyError:
            pass

        logger.info(f"Creating dataloader for split '{split}' with batch size {batch_size}")
        print_info(console, f"Creating dataloader for split '{split}' with batch size {batch_size}")
        dataloader = DataLoader(dataset, **dataloader_args)
        # Log success
        logger.info(f"Created global DataLoader for {split} split " f"(batch_size={dataloader_args['batch_size']})")
        print_info(console, f"[green]✓[/] Created DataLoader for {split} split")
        global_dataloaders[split] = dataloader

    global_model_data = setup_global_model_data(config)

    metric_recorder = setup_metric_recorder(config.metrics)

    global CONDITION
    CONDITION = method
    logger.info(f"Using modality distribution condition: {CONDITION}")
    print_info(console, f"Using modality distribution condition: {CONDITION}")

    clients = setup_clients(
        config=config,
        model_config=config.model,
        datasets=client_datasets,
        metric_recorder=metric_recorder,
        num_clients=config.num_clients,
        full_modality=full_modality,
        client_modality_assignments=client_modality_assignments,
        device=device,
        modality_distribution_method=method,
        cmam_configs=config.cmams,
        cmam_training_configs=config.fed_config.client_cmam_training,
    )

    print_success(console, f"Created {sum([c.num_cmams for c in clients])} C-MAMs on {num_clients} clients")

    logger.info("Setting up global checkpoint manager...")
    print_info(console, "Setting up global checkpoint manager...")
    global_path = config.logging.model_output_path / "global_model"
    global_checkpoint_manager = CheckpointManager(
        model_dir=global_path,
        save_metric=config.save_metric,
        mode="minimize" if config.save_metric == "loss" else "maximize",
        device=device,
    )

    global_cmams = setup_global_cmams(config, device)

    # Determine if C-MAMs should be trained based on configuration epochs
    global_cmam_epochs = getattr(config.fed_config.global_cmam_training, 'epochs', 0) if hasattr(config.fed_config, 'global_cmam_training') else 0
    client_cmam_epochs = getattr(config.fed_config.client_cmam_training, 'epochs', 0) if hasattr(config.fed_config, 'client_cmam_training') else 0
    should_train_cmams = (global_cmams is not None and (global_cmam_epochs > 0 or client_cmam_epochs > 0)) and not args.baseline

    print_info(console, f"C-MAM training configuration: global_epochs={global_cmam_epochs}, client_epochs={client_cmam_epochs}, baseline_flag={args.baseline}")
    print_info(console, f"Will train C-MAMs: {should_train_cmams}")

    trainer = FederatedTrainer(
        base=global_model_data,
        dataloaders=global_dataloaders,
        metric_recorder=metric_recorder,
        checkpoint_manager=global_checkpoint_manager,
        clients=clients,
        cmams=global_cmams,
        train_cmams=should_train_cmams,
        device=device,
        epochs=config.epochs,
        tracked_modalities=tracked_modalities,
    )

    if args.dry_run:
        logger.info("Dry run mode enabled. Skipping training and testing.")
        print_info(console, "Dry run mode enabled. Skipping training and testing.")
        return

    if args.eval_only:
        logger.info("Evaluation-only mode enabled. Skipping training, loading checkpoints, and extracting embeddings.")
        print_info(console, "Evaluation-only mode enabled. Skipping training, loading checkpoints, and extracting embeddings.")
        
        # Load checkpoints and run comprehensive testing
        round_to_load = args.checkpoint_round if args.checkpoint_round is not None else config.num_rounds
        logger.info(f"Loading checkpoints from round {round_to_load}")
        print_info(console, f"Loading checkpoints from round {round_to_load}")
        
        # Load global model checkpoint
        try:
            trainer.checkpoint_manager.load_checkpoint(
                model=trainer.global_model, 
                load_best=True, 
                round=round_to_load,
                ignore="cmams"
            )
            logger.info("Global model checkpoint loaded successfully")
            print_success(console, "Global model checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load global model checkpoint: {e}")
            print_error(console, f"Failed to load global model checkpoint: {e}")
            print_warning(console, "Consider checking checkpoint paths and ensuring training completed successfully")
            raise
        
        # Load client checkpoints
        for client in trainer.clients:
            try:
                # Load base model checkpoint
                client.model.checkpoint_manager.load_checkpoint(
                    model=client.model.model, 
                    load_best=True, 
                    round=round_to_load,
                    ignore="cmams"
                )
                logger.info(f"Client {client.id} base model checkpoint loaded")
                
                # Load C-MAM checkpoints if available
                if client.cmam_data:
                    for cmam_data in client.cmam_data:
                        try:
                            cmam_data.checkpoint_manager.load_checkpoint(
                                model=cmam_data.cmam, 
                                load_best=True, 
                                round=round_to_load
                            )
                            logger.info(f"Client {client.id} C-MAM checkpoint loaded")
                        except Exception as e:
                            logger.warning(f"Failed to load C-MAM checkpoint for client {client.id}: {e}")
                            
            except Exception as e:
                logger.error(f"Failed to load checkpoint for client {client.id}: {e}")
                print_error(console, f"Failed to load checkpoint for client {client.id}: {e}")
                raise
        
        # Load global C-MAM checkpoints if available
        if trainer.cmams:
            for modality_key, cmam_data in trainer.cmams.items():
                try:
                    cmam_data.checkpoint_manager.load_checkpoint(
                        model=cmam_data.cmam, 
                        load_best=True, 
                        round=round_to_load
                    )
                    logger.info(f"Global C-MAM {modality_key} checkpoint loaded")
                except Exception as e:
                    logger.warning(f"Failed to load global C-MAM {modality_key} checkpoint: {e}")
        
        print_success(console, "All checkpoints loaded successfully")
        
        # Run comprehensive testing to extract embeddings
        logger.info("Starting comprehensive testing to extract embeddings...")
        print_info(console, "Starting comprehensive testing to extract embeddings...")
        
        client_results = trainer.run_final_comprehensive_testing()
        
        logger.info("Evaluation-only mode completed successfully")
        print_success(console, "Evaluation-only mode completed successfully")
        
        # Save evaluation results summary
        eval_results_path = Path(config.logging.log_path) / "eval_results.json"
        try:
            import json
            with open(eval_results_path, 'w') as f:
                json.dump(client_results, f, indent=2, default=str)
            logger.info(f"Evaluation results saved to {eval_results_path}")
            print_success(console, f"Evaluation results saved to {eval_results_path}")
        except Exception as e:
            logger.warning(f"Failed to save evaluation results: {e}")
            print_warning(console, f"Failed to save evaluation results: {e}")
        
        return

    if config.training:
        total_rounds = config.num_rounds

        # Check if this is congruent training (sequential phases) or incongruent (intertwined)
        is_congruent = config.fed_config.modality_distribution_method == "congruent"

        if is_congruent:
            print_info(
                console,
                f"Starting CONGRUENT federated training: {total_rounds} rounds base model + {total_rounds} rounds C-MAMs",
            )

            # Phase 1: Complete base model training
            console.rule("[bold yellow]PHASE 1: BASE MODEL FEDERATED TRAINING")
            console.start_task("Base Model Rounds", total=total_rounds)
            base_model_timings = []

            if not args.skip_base_model:
                for round in range(1, total_rounds + 1):
                    logger.info(f"Starting base model round {round}/{total_rounds}")
                    print_info(console, f"Base model round {round}/{total_rounds}")
                    start_time = time.time()
                    trainer.run(round=round, training_phase="base_model")
                    time_taken = time.time() - start_time
                    base_model_timings.append(time_taken)
                    console.update_task("Base Model Rounds", advance=1)
                    logger.info(f"Base model round {round} completed in {time_taken:.2f}s")

                console.complete_task("Base Model Rounds")
                print_success(
                    console, f"Base model training completed in {np.mean(base_model_timings):.2f}s avg per round"
                )

            # Phase 2: Complete C-MAM training (if enabled)
            if trainer.train_cmams:
                console.rule("[bold magenta]PHASE 2: C-MAM FEDERATED TRAINING")
                console.start_task("C-MAM Rounds", total=total_rounds)
                cmam_timings = []

                for round in range(1, total_rounds + 1):
                    logger.info(f"Starting C-MAM round {round}/{total_rounds}")
                    print_info(console, f"C-MAM round {round}/{total_rounds}")
                    start_time = time.time()
                    trainer.run(round=round, training_phase="cmam")
                    time_taken = time.time() - start_time
                    cmam_timings.append(time_taken)
                    console.update_task("C-MAM Rounds", advance=1)
                    logger.info(f"C-MAM round {round} completed in {time_taken:.2f}s")

                console.complete_task("C-MAM Rounds")
                print_success(console, f"C-MAM training completed in {np.mean(cmam_timings):.2f}s avg per round")
            else:
                print_info(console, "C-MAM training disabled, skipping Phase 2")

            total_time = sum(base_model_timings) + (sum(cmam_timings) if trainer.train_cmams else 0)
            print_success(console, f"Congruent federated training completed in {total_time:.2f}s total")

        else:
            # Incongruent training: intertwined base model and C-MAM training
            print_info(console, f"Starting INCONGRUENT federated training: {total_rounds} rounds (intertwined)")
            console.start_task("Incongruent Rounds", total=total_rounds)
            round_timings = []

            for round in range(1, total_rounds + 1):
                logger.info(f"Starting incongruent round {round}/{total_rounds}")
                print_info(console, f"Incongruent round {round}/{total_rounds}")
                start_time = time.time()
                trainer.run(round=round, training_phase="incongruent")
                time_taken = time.time() - start_time
                round_timings.append(time_taken)
                console.update_task("Incongruent Rounds", advance=1)
                logger.info(f"Incongruent round {round} completed in {time_taken:.2f}s")

            console.complete_task("Incongruent Rounds")
            print_success(console, f"Incongruent training completed in {np.mean(round_timings):.2f}s avg per round")

    if testing:
        logger.info("Starting comprehensive final testing phase...")
        print_info(console, "Starting comprehensive final testing phase...")

        client_results = trainer.run_final_comprehensive_testing()

        # logger.info("=== FINAL TEST RESULTS SUMMARY ===")
        # if "global_model" in final_test_results:
        #     global_metrics = final_test_results["global_model"]
        #     logger.info(f"Global Model - Loss: {global_metrics.get('loss', 'N/A'):.4f}")

        #     for metric_name, value in global_metrics.items():
        #         if "accuracy" in metric_name.lower() and isinstance(value, (int, float)):
        #             logger.info(f"Global Model - {metric_name}: {value:.4f}")

        # if "global_cmams" in final_test_results:
        #     cmam_metrics = final_test_results["global_cmams"]
        #     logger.info(f"Global C-MAMs tested: {len([k for k in cmam_metrics.keys() if k.startswith('cmam_')])}")

        #     for metric_name, value in cmam_metrics.items():
        #         if "accuracy" in metric_name.lower() and isinstance(value, (int, float)):
        #             logger.info(f"C-MAMs - {metric_name}: {value:.4f}")

        # logger.info("=== END FINAL TEST RESULTS ===")
        print_success(console, "Comprehensive final testing phase completed successfully")


if __name__ == "__main__":
    parser = ArgumentParser(description="Train a federated multimodal model (and C-MAMs) in a congruent environment.")

    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--run_id", type=int, default=-1, help="The run ID for this experiment.")

    optional_args = parser.add_argument_group("Optional arguments")
    optional_args.add_argument("--dry-run", action="store_true", help="Run a dry run of the experiment.")
    optional_args.add_argument("--skip-train", action="store_true", help="Skip training phase.")
    optional_args.add_argument("--skip-test", action="store_true", help="Skip testing phase.")
    optional_args.add_argument(
        "--skip-base-model", action="store_true", help="Skip training the base model in congruent training."
    )
    optional_args.add_argument(
        "--baseline", action="store_true", help="Run the experiment in baseline mode (no C-MAMs)."
    )
    optional_args.add_argument(
        "--disable_monitoring", action="store_true", help="Disable monitoring of model weights and gradients."
    )
    optional_args.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility (default: 42)."
    )
    optional_args.add_argument(
        "--eval-only", action="store_true", help="Run in evaluation-only mode (skip training, load checkpoints, extract embeddings)."
    )
    optional_args.add_argument(
        "--checkpoint-round", type=int, default=None, help="Specific round to load checkpoints from (default: latest/best)."
    )

    args = parser.parse_args()

    # Setup experiment
    config = setup_experiment(args.config, args.run_id, args.seed)
    config.experiment.dry_run = args.dry_run
    config.experiment.is_train = not args.skip_train and not args.eval_only
    config.experiment.is_test = not args.skip_test or args.eval_only  # Always test in eval-only mode

    ## create all the folders
    os.makedirs(config.logging.log_path, exist_ok=True)
    main(args, config)
