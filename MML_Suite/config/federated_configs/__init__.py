from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from config.base_config import BaseConfig
from config.logging_config import LoggingConfig
from config.metric_config import MetricConfig
from experiment_utils.global_state import set_current_exp_name, set_current_run_id
from experiment_utils.utils import format_path_with_env
from config.model_config import ModelConfig
from experiment_utils.logging import get_logger
from experiment_utils.printing import get_console, print_debug, print_error, print_info, print_success
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from config.experiment_config import ExperimentConfig
from config.federated_configs.federated_data_config import FedDataConfig
from config.multimodal_training_config import TrainingConfig, create_lambda_scheduler
from config.optimizer_config import ParameterGroupsOptimizer
from config.resolvers import resolve_scheduler
from fed import Aggregation, ClientSelectionStrategy

console = get_console()
logger = get_logger()


@dataclass
class FederatedExperimentConfig:
    """
    Configuration for federated learning experiments.
    This configuration is used to set up the federated learning environment,
    including data, clients, and model configurations.
    """

    experiment: ExperimentConfig
    model: ModelConfig
    fed_data: FedDataConfig
    fed_config: FederatedConfig
    logging: LoggingConfig
    metrics: MetricConfig
    cmams: Optional[ModelConfig] = None

    def __getitem__(self, key: str) -> Any:
        """Access configuration components."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration components."""
        setattr(self, key, value)

    def __str__(self) -> str:
        """Converts all the components into appropriate tabular format"""
        lines = [
            "--Experiment Config--",
            str(self.experiment),
            "",
            "--Data Config--",
            str(self.fed_data),
            "",
            "--Model Config--",
            str(self.model),
            "",
            "--Federated Config--",
            str(self.fed_config),
        ]
        if self.cmams is not None:
            lines.extend([
            "",
            "--CMAMS Model Config--",
            str(self.cmams),
            ])
        return "\n".join(lines)

    def get(self, key: str, default=None) -> Any:
        """Get configuration component by key."""
        return getattr(self, key, default)

    def setup(self) -> None:
        """Setup experiment-specific configuration."""
        pass

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)
            logger.info(f"Configuration saved to {path}")

    def get_optimizer(self, model: Any) -> Optimizer:
        """Create optimizer instance."""
        try:
            parameter_group_optimizer = ParameterGroupsOptimizer(self.training.optimizer)
            optimizer = parameter_group_optimizer.get_optimizer(model)
            logger.info(f"Created optimizer: {optimizer.__class__.__name__}")
            return optimizer
        except Exception as e:
            error_msg = f"Error creating optimizer: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise

    def get_scheduler(self, optimizer: Any) -> Optional[LRScheduler]:
        """Create scheduler instance."""
        if not self.training.scheduler:
            return None

        try:
            scheduler_class = resolve_scheduler(self.training.scheduler)

            if self.training.scheduler == "lambda":
                scheduler_args = self.training.scheduler_args.copy()
                console.print(f"Scheduler args: {scheduler_args}")
                scheduler = create_lambda_scheduler(optimizer, scheduler_args)
                console.print(f"Created LambdaLR scheduler with args: {scheduler_args}")
            else:
                scheduler = scheduler_class(optimizer, **self.training.scheduler_args)

            logger.info(f"Created scheduler: {scheduler.__class__.__name__}")
            return scheduler

        except Exception as e:
            error_msg = f"Error creating scheduler: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise

    @classmethod
    def load(cls, path: str | Path, run_id: int, seed: Optional[int] = None) -> FederatedExperimentConfig:
        """Load and create configuration from YAML file."""
        set_current_run_id(run_id)

        print_info(console, f"\nLoading configuration from: {path}")

        try:
            console.print(os.path.exists(path))
            console.print(os.path.exists(Path(path).parent))
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            # Create component configs
            experiment_config = data["experiment"]
            experiment_config["run_id"] = run_id

            if seed is not None:
                experiment_config["seed"] = seed
                print_debug(console, f"Seed set to: {seed}")

            console.print(f"Experiment Name: {experiment_config['name']}")
    
            set_current_exp_name(experiment_config["name"])

            model_config: ModelConfig = data["model"]

            logging_config: LoggingConfig = LoggingConfig.from_dict(
                data["logging"],
                experiment_name=experiment_config["name"],
                run_id=run_id,
            )

            if model_config.pretrained_path is not None:
                model_config.pretrained_path = logging_config.format_path(
                    format_path_with_env(model_config.pretrained_path)
                )
                print_debug(f"Pretrained Path: {model_config.pretrained_path}")
            model_config.validate_config(run_id=run_id, is_cv=experiment_config.cross_validation is not None)
            
            fed_data_config = data["fed_data"]
            fed_config: FederatedConfig = data["fed_config"]
            
            cmams_config: Optional[ModelConfig] = data.get("cmams", None)


            metrics_config = MetricConfig.from_dict(data["metrics"])

            # Create complete config
            config = FederatedExperimentConfig(
                experiment=experiment_config,
                model=model_config,
                fed_data=fed_data_config,
                fed_config=fed_config,
                logging=logging_config,
                metrics=metrics_config,
                cmams=cmams_config
            )
            # Setup and validate
            config.setup()

            logger.info(f"Successfully loaded configuration from {path}")
            print_success(console, "Configuration loaded successfully")

            return config

        except Exception as e:
            console.print(os.listdir(Path(path)))
            console.print(os.getcwd())

            error_msg = f"Error loading configuration: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            print_error(console, f"{error_msg}")
            raise Exception(error_msg)

    @property
    def num_clients(self) -> int:
        """Get number of clients from federated configuration."""
        return self.fed_config.num_clients
    
    @property
    def num_rounds(self) -> int:
        """Get number of rounds from federated configuration."""
        return self.fed_config.num_rounds
    
    @property
    def save_metric(self) -> str:
        """Get the metric to save from the metrics configuration."""
        return self.logging.save_metric

    @property
    def epochs(self) -> int:
        """Get number of epochs from global training configuration."""
        return self.fed_config.global_training.epochs
    
    @property
    def training(self) -> bool:
        """Get the training configuration for global training."""
        return self.experiment.is_train
    
    @property
    def testing(self) -> bool:
        """Get the training configuration for testing."""
        return self.experiment.is_test
    

@dataclass(kw_only=True)
class FederatedConfig:
    """
    Configuration for federated congruent training.
    This configuration is used to set up the federated learning environment
    with congruent modalities.
    """

    num_clients: int
    num_rounds: int
    client_selection: ClientSelectionStrategy
    aggregation: Aggregation
    global_training: TrainingConfig
    client_training: TrainingConfig
    global_cmam_training: dict[str, TrainingConfig] | TrainingConfig
    client_cmam_training: dict[str, TrainingConfig] | TrainingConfig
    modality_distribution_method: Literal["congruent", "incongruent"] = "congruent"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FederatedConfig:
        """Create configuration from a dictionary."""
        try: 
            globaL_training = data["global_training"]
            global_training = TrainingConfig.from_dict(globaL_training)
            client_training = data["client_training"]
            client_training = TrainingConfig.from_dict(client_training)

            return cls(
                num_clients=data.get("num_clients", 10),
                num_rounds=data.get("num_rounds", 100),
                client_selection=ClientSelectionStrategy.from_dict(data["client_selection"]),
                aggregation=Aggregation.from_dict(data["aggregation"]),
                global_training=global_training,
                client_training= client_training,
                global_cmam_training=data.get("global_cmam_training"),
                client_cmam_training=data.get("client_cmam_training"),
                modality_distribution_method=data.get("modality_distribution_method", "congruent")
            )
        except KeyError as e:
            error_msg = f"Missing key in federated config data: {str(e)}, available keys: {list(data.keys())}"
            logger.error(error_msg)
            print_error(console, error_msg)
            raise KeyError(error_msg)
    