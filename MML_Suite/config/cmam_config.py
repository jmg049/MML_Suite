from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
import traceback
from typing import Any, Dict, Union

import yaml
from config.metric_config import MetricConfig
from config.monitor_config import MonitorConfig
from config.logging_config import LoggingConfig
from experiment_utils.global_state import set_current_exp_name, set_current_run_id

from config.base_config import BaseConfig
from config.model_config import ModelConfig
from config.multimodal_training_config import BaseExperimentConfig, StandardMultimodalConfig, TrainingConfig
from experiment_utils.printing import get_console
from experiment_utils.utils import format_path_with_env
from modalities import Modality
from experiment_utils.logging import get_logger


logger = get_logger()
console = get_console()


@dataclass(kw_only=True)
class AssociationNetworkConfig(BaseConfig):
    input_size: int
    hidden_size: int
    output_size: int
    batch_norm: bool = False
    dropout: float = 0.0

    def __str__(self) -> str:
        return f"AssociationNetworkConfig(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size})"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AssociationNetworkConfig:
        return cls(
            input_size=data["input_size"],
            hidden_size=data["hidden_size"],
            output_size=data["output_size"],
            batch_norm=data.get("batch_norm", False),
            dropout=data.get("dropout", 0.0),
        )


@dataclass(kw_only=True)
class CMAMConfig(BaseExperimentConfig):
    cmam: ModelConfig

    def setup(self) -> None:
        pass

    def __str__(self) -> str:
        return super().__str__()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cmam": self.cmam.to_dict(),
            "target_modality": self.target_modality,
            **super().to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CMAMConfig:
        return CMAMConfig(
            cmam=ModelConfig.from_dict(data["cmam"]),
            target_modality=data["target_modality"],
            experiment=BaseExperimentConfig.from_dict(data),
        )

    @classmethod
    def load(cls, path: Union[str, Path, PathLike], run_id: int) -> CMAMConfig:
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            # Create component configs
            experiment_config = data["experiment"]
            experiment_config["run_id"] = run_id

            set_current_run_id(run_id)
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

            training_config._display_config()

            metrics_config = MetricConfig.from_dict(data["metrics"])

            monitoring_config = MonitorConfig.from_dict(data["monitoring"])

            logger.info(f"Successfully loaded configuration from {path}")
            console.print("[green]✓[/] Configuration loaded successfully")

            cmam_model_config: ModelConfig = data["cmam"]
            cmam_model_config.validate_config(run_id=run_id)
            try:
                cmam_config = cls(
                    cmam=cmam_model_config,
                    experiment=experiment_config,
                    data=data_config,
                    model=model_config,
                    logging=logging_config,
                    training=training_config,
                    metrics=metrics_config,
                    monitoring=monitoring_config,
                    _config_path=str(path),
                )
            except KeyError as ke:
                raise KeyError(f"Missing key in CMAM configuration: {str(ke)}")

            return cmam_config

        except Exception as e:
            error_msg = f"Error loading configuration: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            console.print(f"[red]✗[/] {error_msg}")
            raise
