import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from config.base_config import BaseConfig
from config.data_config import DataConfig
from config.experiment_config import ExperimentConfig
from config.logging_config import LoggingConfig
from config.metric_config import MetricConfig
from config.model_config import ModelConfig
from config.monitor_config import MonitorConfig
from config.optimizer_config import OptimizerConfig, ParameterGroupsOptimizer
from config.resolvers import resolve_scheduler
from experiment_utils.global_state import set_current_exp_name, set_current_run_id
from experiment_utils.logging import get_logger
from experiment_utils.loss import LossFunctionGroup
from experiment_utils.printing import get_console
from experiment_utils.utils import format_path_with_env
from rich.panel import Panel
from rich.table import Table
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

logger = get_logger()
console = get_console()


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training parameters and optimization."""

    epochs: int
    num_modalities: int
    optimizer: OptimizerConfig
    loss_functions: LossFunctionGroup
    scheduler: Optional[str] = None
    scheduler_args: Dict[str, Any] = field(default_factory=dict)
    validation_interval: int = 1
    missing_rates: Optional[List[float]] = None
    do_validation_visualization: bool = False
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_metric: str = "loss"
    save_metric: str = "loss"

    def __post_init__(self):
        """Validate training configuration."""
        self._validate_config()
        self._display_config()

    def _validate_config(self) -> None:
        """Validate training parameters."""
        try:
            if self.num_modalities < 1:
                raise ValueError("Number of modalities must be at least 1")

            if self.missing_rates is not None:
                if len(self.missing_rates) != self.num_modalities:
                    raise ValueError(
                        f"Number of missing rates ({len(self.missing_rates)}) "
                        f"must match number of modalities ({self.num_modalities})"
                    )
                if not all(0.0 <= rate <= 1.0 for rate in self.missing_rates):
                    raise ValueError("Missing rates must be between 0 and 1")
            else:
                self.missing_rates = [0.0] * self.num_modalities

            logger.info("Training configuration validated successfully")

        except Exception as e:
            error_msg = f"Training configuration validation failed: {str(e)}"
            logger.error(error_msg)
            console.print(f"[red]✗[/] {error_msg}")
            raise

    def _display_config(self) -> None:
        """Display training configuration."""
        console.rule("[heading]Initialising Training Configuration[/]")
        table = Table(title="Training Configuration", title_style="bold white", expand=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        # Add basic parameters
        table.add_row("Epochs", str(self.epochs))

        for key, value in self.loss_functions.items():
            table.add_row(f"{key} Criterion", f"{value}")

        if self.scheduler:
            table.add_row("Scheduler", f"{self.scheduler} {self.scheduler_args}")

        # Add modality information
        table.add_row("Number of Modalities", str(self.num_modalities))
        if self.missing_rates:
            table.add_row("Missing Rates", str(self.missing_rates))

        # Add early stopping information if enabled
        if self.early_stopping:
            table.add_row("Early Stopping", "Enabled")
            table.add_row("Patience", str(self.early_stopping_patience))
            table.add_row("Min Delta", str(self.early_stopping_min_delta))

        console.print(table)


    def get_optimizer(self, model: torch.nn.Module | list[torch.nn.Module]) -> Optimizer:
        """Create optimizer instance."""
        try:
            parameter_group_optimizer = ParameterGroupsOptimizer(self.optimizer)

            all_params = {}
            if isinstance(model, list):
                for m in model:
                    all_params.update(dict(m.named_parameters()))
            else:
                all_params.update(dict(model.named_parameters()))

            optimizer = parameter_group_optimizer.get_optimizer(all_params)
            logger.info(f"Created optimizer: {optimizer.__class__.__name__}")
            return optimizer
        except Exception as e:
            error_msg = f"Error creating optimizer: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise
    
    def get_scheduler(self, optimizer: Any) -> Optional[LRScheduler]:
        """Create scheduler instance."""
        if not self.scheduler:
            return None

        try:
            scheduler_class = resolve_scheduler(self.scheduler)

            if self.scheduler == "lambda":
                scheduler_args = self.scheduler_args.copy()
                console.print(f"Scheduler args: {scheduler_args}")
                scheduler = create_lambda_scheduler(optimizer, scheduler_args)
                console.print(f"Created LambdaLR scheduler with args: {scheduler_args}")
            else:
                scheduler = scheduler_class(optimizer, **self.scheduler_args)

            logger.info(f"Created scheduler: {scheduler.__class__.__name__}")
            return scheduler

        except Exception as e:
            error_msg = f"Error creating scheduler: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise

    def get_loss_function(self) -> LossFunctionGroup:
        """Get loss functions."""
        return self.loss_functions



@dataclass
class BaseExperimentConfig(ABC):
    """Abstract base class for experiment configurations."""

    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    logging: LoggingConfig
    metrics: MetricConfig
    training: TrainingConfig
    monitoring: MonitorConfig
    _config_path: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        """Access configuration components."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration components."""
        setattr(self, key, value)

    def __str__(self) -> str:
        """Converts all the components into appropriate tabular format"""
        return f"--Experiment Config--\n{self.experiment}\n\n--Data Config--\n{self.data}\n\n--Model Config--\n{self.model}\n\n-Logging Config--\n{self.logging}\n\n--Metric Config--\n{self.metrics}\n\n--Training Config--\n{self.training}\n--Monitoring Config--\n{self.monitoring}"

    @abstractmethod
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
            all_params = {}
            if isinstance(model, list):
                for m in model:
                    all_params.update(dict(m.named_parameters()))
            else:
                all_params.update(dict(model.named_parameters()))
            optimizer = parameter_group_optimizer.get_optimizer(all_params)
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


def create_lambda_scheduler(optimizer, scheduler_args):
    """
    Creates a PyTorch LambdaLR scheduler with proper variable scoping.

    Args:
        optimizer: PyTorch optimizer
        scheduler_args: Dictionary containing scheduler arguments
    """
    # Extract all the arguments
    niter = scheduler_args.pop("niter", None)
    niter_decay = scheduler_args.pop("niter_decay", None)
    epoch_count = scheduler_args.pop("epoch_count", 0)

    def lmbda(epoch):
        return 1.0 - max(0, epoch + epoch_count - niter) / float(niter_decay + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    console.print(f"Created LambdaLR scheduler with lambda function: {lmbda}")
    return scheduler

@dataclass
class StandardMultimodalConfig(BaseExperimentConfig):
    """Standard implementation of experiment configuration."""

    def setup(self) -> None:
        """Setup standard configuration."""
        self._validate_components()

        # self._display_summary()

    def _validate_components(self) -> None:
        """Validate all configuration components."""
        try:
            # Add more cross-component validations as needed
            logger.info("All configuration components validated successfully")

        except Exception as e:
            error_msg = f"Configuration validation failed: {str(e)}"
            logger.error(error_msg)
            console.print(f"[red]✗[/] {error_msg}")
            raise

    def _display_summary(self) -> None:
        """Display complete configuration summary."""
        console.print(Panel("[bold blue]Experiment Configuration Summary[/]"))

        # Display experiment info
        console.print(f"\n[cyan]Experiment:[/] {self.experiment.name}")
        console.print(f"Run ID: {self.experiment.run_id}")
        console.print(f"Device: {self.experiment.device}")

        # Display configuration status
        table = Table(title="Configuration Components")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        components = {
            "Model": self.model.name,
            "Dataset": len(self.data.datasets),
            "Metrics": len(self.metrics.metrics) if hasattr(self.metrics, "metrics") else "N/A",
            "Training": f"{self.training.epochs} epochs",
        }

        for component, status in components.items():
            table.add_row(component, str(status))

        console.print(table)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "experiment": self.experiment.to_dict(),
            "data": self.data.to_dict(),
            "model": self.model.to_dict(),
            "logging": self.logging.to_dict(),
            "metrics": self.metrics.to_dict(),
            "training": self.training.to_dict(),
        }

    def __str__(self) -> str:
        return super().__str__()

    @classmethod
    def load(cls, path: Union[str, Path, PathLike], run_id: int) -> "StandardMultimodalConfig":
        """Load and create configuration from YAML file."""
        set_current_run_id(run_id)

        console.print(f"\nLoading configuration from: {path}")
        import os

        try:
            console.print(os.path.exists(path))
            console.print(os.path.exists(Path(path).parent))
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            # Create component configs
            experiment_config = data["experiment"]
            experiment_config["run_id"] = run_id
    
            set_current_exp_name(experiment_config["name"])

            data_config = data["data"]

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
                console.print(f"Pretrained Path: {model_config.pretrained_path}")
                model_config.validate_config(run_id=run_id, is_cv=experiment_config.cross_validation is not None)
            training_config = TrainingConfig.from_dict(data["training"])

            metrics_config = MetricConfig.from_dict(data["metrics"])

            monitoring_config = MonitorConfig.from_dict(data["monitoring"])

            # Create complete config
            config = cls(
                experiment=experiment_config,
                data=data_config,
                model=model_config,
                logging=logging_config,
                training=training_config,
                metrics=metrics_config,
                monitoring=monitoring_config,
                _config_path=str(path),
            )

            # Setup and validate
            config.setup()

            logger.info(f"Successfully loaded configuration from {path}")
            console.print("[green]✓[/] Configuration loaded successfully")

            return config

        except Exception as e:
            import os

            console.print(os.listdir(Path(path).parent))
            console.print(os.getcwd())

            error_msg = f"Error loading configuration: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            console.print(f"[red]✗[/] {error_msg}")
            raise
