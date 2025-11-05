import logging
import os
from pathlib import Path
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypeVar, Union

import h5py
import numpy as np
import torch
from modalities import Modality
from numpy import ndarray
from torch import Tensor
from torch.nn import BatchNorm2d, Conv2d, Linear, Module, init
from typing_extensions import TypeGuard


from .printing import get_console, print_debug, print_info, print_warning

console = get_console()

torch.set_default_dtype(torch.float32)

PARAMETER_SIZE_BYTES: int = 4  # Size of a float parameter in bytes
CONDITION: str = "congruent"  # Default condition for modality assignment


T = TypeVar("T")  # Return type


class AccessErrorType(Enum):
    KEY_ERROR = "key_error"
    TYPE_ERROR = "type_error"
    MAX_DEPTH_EXCEEDED = "max_depth_exceeded"
    VALIDATION_ERROR = "validation_error"


@dataclass
class AccessError(BaseException):
    """Represents an error that occurred during nested dictionary access."""

    error_type: AccessErrorType
    path: str
    failed_key: str
    available_keys: Optional[List[str]]
    attempted_path: List[Union[str, int]]
    message: str
    original_error: Optional[Exception] = None

    def __str__(self) -> str:
        parts = [
            f"Failed to access nested data at path '{self.path}'",
            f"Error type: {self.error_type.value}",
            f"Failed key: '{self.failed_key}'",
        ]

        if self.available_keys is not None:
            parts.append(f"Available keys: {self.available_keys}")

        parts.append(f"Attempted path: {' -> '.join(str(k) for k in self.attempted_path)}")

        return " | ".join(parts)

    def format_multiline(self) -> str:
        """Returns a multiline formatted version of the error"""
        parts = [
            f"Failed to access nested data at path '{self.path}'",
            f"Error type: {self.error_type.value}",
            f"Failed key: '{self.failed_key}'",
        ]

        if self.available_keys is not None:
            parts.append(f"Available keys: {self.available_keys}")

        parts.append(f"Attempted path: {' -> '.join(str(k) for k in self.attempted_path)}")

        return "\n".join(parts)


def is_valid_dict(obj: Any) -> TypeGuard[Dict]:
    """Type guard to check if an object is a valid dictionary."""
    return isinstance(obj, dict)


class NestedDictAccess:
    """
    Provides safe access to nested dictionary structures with detailed error reporting.
    """

    def __init__(self, max_depth: int = 20, logger: Optional[logging.Logger] = None, raise_on_error: bool = True):
        """
        Initialize the nested dictionary accessor.

        Args:
            max_depth: Maximum allowed depth for nested access
            logger: Optional logger for error logging
            raise_on_error: If True, raises exceptions; if False, returns Result object
        """
        self.max_depth = max_depth
        self.logger = logger or logging.getLogger(__name__)
        self.raise_on_error = raise_on_error

    def _validate_depth(self, depth: int, keys: List[Any]) -> None:
        """Check if the access depth exceeds the maximum allowed depth."""
        if depth > self.max_depth:
            raise ValueError(f"Maximum depth of {self.max_depth} exceeded. " f"Attempted depth: {depth}")

    def _format_path(self, path: List[Any]) -> str:
        """Format the current path for error reporting."""
        return ".".join(str(k) for k in path)

    def get(
        self, data: Dict, keys: List[Union[str, int]], default: Optional[T] = None, validator: Optional[callable] = None
    ) -> Union[T, AccessError]:
        """
        Safely access nested dictionary values with detailed error reporting.
        Allows non-dictionary values for the final key in the path.
        """
        current_data = data
        path: List[Union[str, int]] = []

        try:
            self._validate_depth(len(keys), keys)

            for i, key in enumerate(keys):
                # Only check for dictionary type if this isn't the last key
                if i < len(keys) - 1 and not is_valid_dict(current_data):
                    error = AccessError(
                        error_type=AccessErrorType.TYPE_ERROR,
                        path=self._format_path(path),
                        failed_key=str(key),
                        available_keys=None,
                        attempted_path=keys,
                        message=f"Expected dictionary at intermediate path, got {type(current_data)}",
                    )
                    self.logger.error(str(error))
                    if self.raise_on_error:
                        raise TypeError(str(error))
                    raise error

                try:
                    current_data = current_data[key]
                    path.append(key)
                except (KeyError, IndexError, TypeError) as e:
                    # Handle different types of access errors
                    available_keys = list(current_data.keys()) if is_valid_dict(current_data) else None
                    error = AccessError(
                        error_type=AccessErrorType.KEY_ERROR,
                        path=self._format_path(path),
                        failed_key=str(key),
                        available_keys=available_keys,
                        attempted_path=keys,
                        message=f"Failed to access key/index '{key}': {str(e)}",
                    )
                    self.logger.error(str(error))
                    if self.raise_on_error:
                        raise KeyError(str(error))
                    raise error

            if validator is not None and not validator(current_data):
                error = AccessError(
                    error_type=AccessErrorType.VALIDATION_ERROR,
                    path=self._format_path(path),
                    failed_key=str(keys[-1]),
                    available_keys=None,
                    attempted_path=keys,
                    message="Value failed validation",
                )
                self.logger.error(str(error))
                if self.raise_on_error:
                    raise ValueError(str(error))
                raise error

            return current_data

        except Exception as e:
            if not isinstance(e, (TypeError, KeyError, ValueError, IndexError)):
                self.logger.exception("Unexpected error during nested access")
                error = AccessError(
                    error_type=AccessErrorType.TYPE_ERROR,
                    path=self._format_path(path),
                    failed_key=str(keys[len(path)]) if len(path) < len(keys) else "",
                    available_keys=None,
                    attempted_path=keys,
                    message=str(e),
                    original_error=e,
                )
                if self.raise_on_error:
                    raise
                return error

            raise


def flatten_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
    - data (Dict[str, Any]): Input dictionary.

    Returns:
    - Dict[str, Any]: Flattened dictionary.
    """

    out_dict = {}

    def _flatten_dict(data, parent_key=""):
        for k, v in data.items():
            new_key = k
            # new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                _flatten_dict(v, new_key)
            else:
                out_dict[new_key] = v

    _flatten_dict(data)

    return out_dict


def hdf5_to_dict(f: h5py.File) -> Dict[str, Any]:
    """
    Convert an HDF5 file to a dictionary.

    Args:
    - f (h5py.File): HDF5 file object.

    Returns:
    - Dict[str, Any]: Dictionary containing the contents of the HDF5 file.
    """
    return {k: f[k][()] for k in f.keys()}


def format_path_with_env(path_template, **kwargs):
    # Find environment variables in the path template
    env_vars = re.findall(r"\$(\w+)", path_template)

    # Replace each environment variable with its value or a default
    for var in env_vars:
        default_value = kwargs.get(var.lower(), ".")  # Fetch from kwargs or use empty string
        path_template = path_template.replace(f"${var}", os.getenv(var, default_value))

    # Format with additional variables like exp_name and run
    return path_template


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def gpu_memory() -> str:
    return (
        f"Allocated:\t{torch.cuda.memory_allocated()/1e9:.2f}GB\nCached:\t{torch.cuda.memory_reserved()/1e9:.2f}GB"
        if torch.cuda.is_available()
        else "!!!GPU not available!!!"
    )


def to_gpu_safe(x: Tensor | Dict[str | Modality, Tensor | Any]) -> Tensor | Dict[str | Modality, Tensor | Any]:
    """
    Safely move a tensor or dictionary containing tensors to the GPU. If the GPU is not available, fallsback to the CPU with a warning.

    Args:
        x (Tensor | Dict[str  |  Modality, Tensor  |  Any]): The input(s) to move to the GPU (if available)

    Returns:
        Tensor | Dict[str | Modality, Tensor | Any]: The input(s) moved to the GPU (if available)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("CUDA device not available, using CPU as a fallback")
        console.info("[bold yellow]\u26a0[/] [yellow]CUDA device not available, using CPU as a fallback[/]")

    if isinstance(x, Tensor):
        return x.to(device)

    return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in x.items()}


def kaiming_init(module: Module) -> None:
    if isinstance(module, (Conv2d, Linear)):
        init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)


def clean_checkpoints(
    checkpoints_dir: str,
    round: Optional[int] = None,
    keep_epochs: List[int] = None,
    keep_best: bool = True,
    keep_last: bool = True,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Clean model checkpoints in a directory.

    Args:
    - checkpoints_dir (str): Directory containing checkpoint files.
    - keep_epochs (List[int]): List of epoch numbers to keep.
    - keep_best (bool): Whether to keep the best checkpoint. Defaults to True.
    - keep_last (bool): Whether to keep the last checkpoint. Defaults to True.
    - dry_run (bool): If True, only print what would be done without actually deleting. Defaults to False.
    - print_fn (Callable): Function to use for printing. Defaults to built-in print.

    Returns:
    - Optional[str]: Path to the last kept checkpoint, or None if no checkpoints were kept.
    """
    if round is not None:
        checkpoints_dir = os.path.join(checkpoints_dir, f"round_{round}")

    if not os.path.exists(checkpoints_dir):
        console.print(f"Directory {checkpoints_dir} does not exist.")
        return None

    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
    kept_checkpoints = []
    last_checkpoint = None

    epoch_pattern = re.compile(r"epoch_(\d+)\.pth$")
    if len(checkpoints) == 0:
        return None
    console.print(f"Cleaning checkpoints in {checkpoints_dir}...")
    for checkpoint in sorted(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x))):
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint)

        if keep_best and checkpoint.endswith("best.pth"):
            kept_checkpoints.append(checkpoint)
            console.print(f"Keeping best checkpoint: {checkpoint}")
            continue

        match = epoch_pattern.search(checkpoint)
        if match:
            epoch = int(match.group(1))
            if keep_epochs is not None and epoch in keep_epochs:
                kept_checkpoints.append(checkpoint)
                console.print(f"Keeping checkpoint for epoch {epoch}: {checkpoint}")
            else:
                if not dry_run:
                    os.remove(checkpoint_path)
                console.print(f"{'Would remove' if dry_run else 'Removing'} checkpoint: {checkpoint}")
        else:
            if not dry_run:
                os.remove(checkpoint_path)
            console.print(f"{'Would remove' if dry_run else 'Removing'} non-conforming checkpoint: {checkpoint}")

        last_checkpoint = checkpoint

    if keep_last and last_checkpoint and last_checkpoint not in kept_checkpoints:
        kept_checkpoints.append(last_checkpoint)
        console.print(f"Keeping last checkpoint: {last_checkpoint}")
        if not dry_run and os.path.exists(os.path.join(checkpoints_dir, last_checkpoint)):
            os.rename(
                os.path.join(checkpoints_dir, last_checkpoint),
                os.path.join(checkpoints_dir, last_checkpoint.replace(".pth", "_last.pth")),
            )

    return os.path.join(checkpoints_dir, kept_checkpoints[-1]) if kept_checkpoints else None


def safe_detach(tensor: Tensor | ndarray, to_np: bool = True) -> Tensor | ndarray:
    """
    Safely detaches PyTorch tensors and optionally converts them to numpy arrays. If the input is already a numpy array then it returns the input.

    Args:
        tensor: Input tensor or numpy array
        to_np: If True, converts PyTorch tensor to numpy array

    Returns:
        Detached tensor or numpy array
    """
    assert (
        isinstance(tensor, Tensor) or isinstance(tensor, ndarray) or isinstance(tensor, list)
    ), f"Expected tensor, list or numpy array, got {type(tensor)}"

    match isinstance(tensor, Tensor):
        case True:
            if to_np:
                return tensor.detach().cpu().numpy()
            return tensor.detach()
        case False:
            return tensor


def prepare_metrics_for_json(metrics_list: List[Dict[str, Dict[str, Any]]]) -> List[Dict[str, Dict[str, Any]]]:
    """
    Recursively convert metrics to JSON-serializable format, handling numpy types.
    """

    def convert_value(v):
        if isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [convert_value(item) for item in v]
        elif isinstance(v, (str, bool)):
            return v
        elif isinstance(
            v,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(v)
        elif isinstance(v, (np.float64, np.float16, np.float32)):
            return float(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return str(v)

    return [
        {outer_k: convert_value(inner_dict) for outer_k, inner_dict in epoch_metrics.items()}
        for epoch_metrics in metrics_list
    ]


def safe_extend(output: dict[str, Any], lst: list[Any], key: str, warn: bool = False, error: bool = False) -> None:
    if key in output:
        if isinstance(output[key], Tensor):
            lst.extend(output[key].detach().cpu().numpy().tolist())
        elif isinstance(output[key], ndarray):
            lst.extend(output[key].tolist())
        elif isinstance(output[key], list):
            lst.extend(output[key])
        else:
            raise TypeError(f"Expected output[{key}] to be a Tensor, ndarray, or list, got {type(output[key])}")

    else:
        if error:
            raise KeyError(
                f"Key '{key}' not found in output. Cannot extend list. Available keys: {list(output.keys())}"
            )
        if warn:
            print_warning(console, f"Key '{key}' not found in output. Skipping extension.")
            return


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_correct_cmam_dataset_selected_patterns(condition: Literal["congruent", "incongruent"], cmam) -> str:
    assert condition in [
        "congruent",
        "incongruent",
    ], f"Invalid condition: {condition}. Must be 'congruent' or 'incongruent'."
    input_modalities = [str(m)[0] for m in cmam.input_modalities]
    target_modality = str(cmam.target_modality)[0]
    pattern_inputs = input_modalities + ([target_modality] if condition == "congruent" else [])
    pattern = "".join(sorted(pattern_inputs, key=lambda x: (len(x), x))).lower()
    cmam_msg = "-".join([str(m) for m in cmam.input_modalities]) + f" -> {cmam.target_modality}"
    # print_debug(console, f"[{condition.title()}] Pattern for C-MAM {cmam_msg}: {pattern}")
    print_info(console, f"[{condition.title()}] Pattern for C-MAM {cmam_msg}: {pattern}")
    return pattern
