import re
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Dict, List, Optional

import torch.optim as optim
from experiment_utils.printing import get_console
from experiment_utils.logging import get_logger
from rich.panel import Panel
from torch.nn import Module

from .base_config import BaseConfig
from .resolvers import resolve_optimizer

logger = get_logger()

console = get_console()


@dataclass
class ParameterGroupConfig(BaseConfig):
    """Configuration for a parameter group in the optimizer."""

    name: str
    pattern: List[str]
    lr: float
    weight_decay: float = 0.0
    momentum: Optional[float] = None
    dampening: Optional[float] = None
    betas: Optional[tuple] = None
    eps: Optional[float] = None
    maximize: bool = False

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_mapping(
            "!ParameterGroupConfig",
            {
                "name": node.name,
                "pattern": node.pattern,
                "lr": node.lr,
                "weight_decay": node.weight_decay,
                "momentum": node.momentum,
                "dampening": node.dampening,
                "betas": node.betas,
                "eps": node.eps,
                "maximize": node.maximize,
            },
        )

    @classmethod
    def from_yaml(cls, constructor, node):
        data = constructor.construct_mapping(node, deep=True)
        return cls(**data)

    def __str__(self):
        return str(self.to_dict())


@dataclass
class OptimizerConfig(BaseConfig):
    """Configuration for the optimizer."""

    name: str
    param_groups: Optional[List[ParameterGroupConfig]] = field(default_factory=list)
    default_kwargs: Optional[Dict[str, Any]] = field(default_factory=OrderedDict)

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_mapping(
            "!Optimizer", {"name": node.name, "param_groups": node.param_groups, "default_kwargs": node.default_kwargs}
        )

    @classmethod
    def from_yaml(cls, constructor, node):
        data = constructor.construct_mapping(node, deep=True)
        return cls(**data)

    def __repr__(self):
        return (
            f"OptimizerConfig(name={self.name}, param_groups={self.param_groups}, default_kwargs={self.default_kwargs})"
        )

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return {
            "name": self.name,
            "param_groups": self.param_groups,
            "default_kwargs": self.default_kwargs,
        }

    def __str__(self):
        return pformat(self.to_dict())


class ParameterGroupsOptimizer:
    def __init__(self, optimizer_config: OptimizerConfig):
        """
        Initialize optimizer with parameter groups configuration.
        """
        self.optimizer_config = optimizer_config

    def _validate_regex_patterns(self, all_params) -> Dict[str, List[str]]:
        """
        Validate regex patterns and detect potential overlaps.
        Returns a dictionary mapping group names to lists of overlapping parameters.
        """
        # all_params = dict(model.named_parameters())
        overlap_map = {}
        used_params = set()

        for group in self.optimizer_config.param_groups:
            matching_params = set()
            for pattern in group["pattern"]:
                try:
                    regex = re.compile(pattern)
                except re.error as e:
                    g_name = group["name"]
                    raise ValueError(f"Invalid regex pattern '{pattern}' in group '{g_name}': {str(e)}")

                current_matches = {name for name in all_params.keys() if regex.match(name)}
                matching_params.update(current_matches)

            # Check for overlaps with previously assigned parameters
            overlaps = matching_params & used_params
            if overlaps:
                overlap_map[group["name"]] = list(overlaps)

            used_params.update(matching_params)

        return overlap_map

    def _create_parameter_groups(self, all_params) -> List[Dict]:
        """Create parameter groups based on configuration."""
        # First validate patterns and check for overlaps
        overlap_map = self._validate_regex_patterns(all_params)
        if overlap_map:
            overlap_details = "\n".join(
                f"  - Group '{group}' overlaps on parameters: {', '.join(params)}"
                for group, params in overlap_map.items()
            )
            logger.warning(
                "Detected overlapping parameter patterns. Parameters will be assigned to the first "
                f"matching group only:\n{overlap_details}"
            )

        used_params = set()
        parameter_groups = []

        # Process configured groups
        for group in self.optimizer_config.param_groups:
            group_params = []
            matched_params = set()

            # Find parameters matching the patterns
            for pattern in group["pattern"]:
                regex = re.compile(pattern)
                matching_params = [
                    (name, param) for name, param in all_params.items() if regex.match(name) and name not in used_params
                ]

                for name, param in matching_params:
                    matched_params.add(name)
                    if group["lr"] != 0.0:  # Skip if lr = 0
                        group_params.append(param)

            if matched_params:
                g_name = group["name"]
                logger.debug(f"Group '{g_name}' matched parameters: {', '.join(sorted(matched_params))}")

            used_params.update(matched_params)

            if group_params:
                group_dict = {
                    "params": group_params,
                    "lr": float(group["lr"]),
                    "weight_decay": float(group["weight_decay"]),
                }

                # Add optional parameters if they are set
                if "momentum" in group:
                    group_dict["momentum"] = group["momentum"]
                if "dampening" in group:
                    group_dict["dampening"] = group["dampening"]
                if "betas" in group:
                    group_dict["betas"] = group["betas"]
                if "eps" in group:
                    group_dict["eps"] = group["eps"]
                if "maximize" in group:
                    group_dict["maximize"] = group["maximize"]

                parameter_groups.append(group_dict)

        # Handle remaining parameters with default settings
        remaining_params = [param for name, param in all_params.items() if name not in used_params]

        if remaining_params:
            self.optimizer_config.default_kwargs["lr"] = float(self.optimizer_config.default_kwargs.get("lr", 0.0))
            remaining_group = {"params": remaining_params, **self.optimizer_config.default_kwargs}
            parameter_groups.append(remaining_group)

            remaining_names = [name for name in all_params.keys() if name not in used_params]
            logger.debug(f"Default group contains parameters: {', '.join(sorted(remaining_names))}")

        return parameter_groups

    def get_optimizer(self, all_params) -> optim.Optimizer:
        """Create optimizer instance with parameter groups."""
        try:
            optimizer_class = resolve_optimizer(self.optimizer_config.name)
            parameter_groups = self._create_parameter_groups(all_params)

            # Log parameter group information
            for idx, group in enumerate(parameter_groups):
                num_params = sum(p.numel() for p in group["params"])
                logger.info(
                    f"Parameter group {idx}: {num_params:,} parameters, "
                    f"lr={group.get('lr')}, weight_decay={group.get('weight_decay')}"
                )

            optimizer = optimizer_class(parameter_groups)
            logger.info(f"Created {optimizer.__class__.__name__} with {len(parameter_groups)} parameter groups")
            console.print(f"Created {optimizer.__class__.__name__} with {len(parameter_groups)} parameter groups")
            optimizer_panel = Panel(
                title="[heading]Optimizer Configuration[/]",
                highlight=True,
                expand=True,
                renderable=str(self.optimizer_config),
            )
            console.print(optimizer_panel)
            return optimizer

        except Exception as e:
            error_msg = f"Error creating optimizer: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise
