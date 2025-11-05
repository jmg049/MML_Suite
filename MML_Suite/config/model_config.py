from __future__ import annotations
from datetime import datetime
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from experiment_utils.utils import SafeDict, format_path_with_env
from experiment_utils.printing import get_console
from experiment_utils.logging import get_logger
from rich.table import Table

from .base_config import BaseConfig

logger = get_logger()
console = get_console()


@dataclass
class ModelConfig(BaseConfig):
    """
    Generic model configuration that can handle any model type.
    Provides validation, path handling, and rich console output.
    """

    name: str
    model_type: str
    pretrained_path: Optional[str] = None
    init_fn: Optional[Literal["xavier", "kaiming", "orthogonal"]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    version: str = field(default="1.0.0")

    def __post_init__(self):
        """Initialize and validate the model configuration."""
        console.rule(f"[heading]Initializing Model Configuration: {self.name}[/]")
        self._display_config()

    def format_path(self, path: str, run_id: int) -> Path:
        """Format a path string with config values and return a Path object."""
        if not path:
            return None

        format_vars = SafeDict(
            {
                "run_id": run_id,
            }
        )
        formatted_path = path.format_map(format_vars)
        # formatted_path = self._sanitize_string()
        console.print(f"[bold]Formatted Path:[/] {formatted_path}")
        return Path(formatted_path)

    def validate_config(self, run_id: int, is_cv: bool = False) -> None:
        """Validate the model configuration."""
        # Validate pretrained path
        if self.pretrained_path is not None:
            try:
                path = self.format_path(str(self.pretrained_path), run_id)
                path = format_path_with_env(str(path))
                path = Path(path)
                if not is_cv and not path.exists():
                    raise FileNotFoundError(f"Pretrained path not found: {path}")
                console.print(f"[bold green]✓[/] Pretrained path verified: {path}")
                self.pretrained_path = str(path.resolve())
                logger.info(f"Validated pretrained path: {path}")
                console.print(f"[bold green]✓[/] Pretrained path verified: {self.pretrained_path}")
            except Exception as e:
                error_msg = f"Error validating pretrained path: {str(e)}"
                logger.error(error_msg)
                console.print(f"[bold red]✗[/] {error_msg}")
                raise

        # Try to import model type if it's a path
        if "." in self.model_type:
            try:
                module_path, class_name = self.model_type.rsplit(".", 1)
                module = importlib.import_module(module_path)
                getattr(module, class_name)
                logger.info(f"Validated model type import: {self.model_type}")
                console.print(f"[bold green]✓[/] Model type verified: {self.model_type}")
            except Exception as e:
                logger.warning(f"Could not import model type {self.model_type}: {str(e)}")
                console.print(f"[bold yellow]![/] Model type import warning: {self.model_type}")

        # Log kwargs validation
        logger.info(f"Model configuration includes {len(self.kwargs)} additional parameters")

    def _display_config(self) -> None:
        """Display the model configuration in a formatted table."""
        # Basic config table
        config_table = Table(title=f"Model Configuration: {self.name}", expand=True)
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Name", self.name)
        config_table.add_row("Model Type", self.model_type)
        config_table.add_row("Version", self.version)
        config_table.add_row(
            "Pretrained Path",
            str(self.pretrained_path) if self.pretrained_path else "None",
        )

        console.print(config_table)

        # Additional parameters table
        if self.kwargs:
            kwargs_table = Table(title="Additional Parameters", expand=True, title_style="bold white")
            kwargs_table.add_column("Parameter", style="cyan")
            kwargs_table.add_column("Value")

            for key, value in self.kwargs.items():
                kwargs_table.add_row(str(key), str(value))

            console.print(kwargs_table)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        """Create a ModelConfig instance from a dictionary."""
        try:
            # Extract base fields
            name = data.pop("name")
            model_type = data.pop("model_type")
            pretrained_path = data.pop("pretrained_path", None)
            init_fn = data.pop("init_fn", None)
            version = data.pop("version", "1.0.0")

            # All remaining fields go to kwargs
            kwargs = data

            return cls(
                name=name,
                model_type=model_type,
                pretrained_path=pretrained_path,
                init_fn=init_fn,
                kwargs=kwargs,
                version=version,
            )
        except KeyError as e:
            error_msg = f"Missing required field in model configuration: {str(e)}"
            logger.error(error_msg)
            console.print(f"[bold red]Error:[/] {error_msg}")
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error creating model configuration: {str(e)}"
            logger.error(error_msg)
            console.print(f"[bold red]Error:[/] {error_msg}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = {
            "name": self.name,
            "model_type": self.model_type,
            "init_fn": self.init_fn,
            "version": self.version,
        }

        if self.pretrained_path:
            base_dict["pretrained_path"] = self.pretrained_path

        # Add all kwargs to the base dictionary
        base_dict.update(self.kwargs)

        return base_dict

    def update_kwargs(self, **kwargs) -> None:
        """Update model configuration parameters."""
        self.kwargs.update(kwargs)
        logger.info(f"Updated model parameters: {kwargs}")
        self._display_config()

    def get_param(self, param: str, default: Any = None) -> Any:
        """
        Get a parameter from kwargs or base configuration.

        Args:
            param: Parameter name
            default: Default value if parameter not found
        """
        if param in {"name", "model_type", "pretrained_path", "version"}:
            return getattr(self, param)
        return self.kwargs.get(param, default)

    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        header = f"ModelConfig(name={self.name}, model_type={self.model_type})"
        model_params = "\n".join([f"{k}={v}" for k, v in self.kwargs.items()])
        return f"{header}\n{model_params}"
