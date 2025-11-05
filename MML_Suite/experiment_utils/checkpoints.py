import os
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from experiment_utils.utils import AccessError, NestedDictAccess
from experiment_utils.printing import print_info
from .logging import get_logger
from .printing import get_console

logger = get_logger()
console = get_console()


class CheckpointManager:
    """Manages model checkpointing and loading."""

    def __init__(
        self,
        model_dir: Path,
        save_metric: str = "loss",
        mode: str = "minimize",
        device: str = "cuda",
    ):
        self.model_dir = Path(model_dir)
        self.save_metric = save_metric
        self.mode = mode
        self.device = device
        self.best_metric = float("inf") if mode == "minimize" else float("-inf")
        self.best_epoch = -1

        # Create directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def is_better(self, current: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == "minimize":
            return current < self.best_metric
        return current > self.best_metric

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        round:int = None
    ) -> None:
        """Save model checkpoint."""
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        # Save regular checkpoint

        checkpoint_path = self.model_dir / f"epoch_{epoch}.pth" if not round else self.model_dir / f"round_{str(round)}"  / f"epoch_{epoch}.pth"
        os.makedirs(checkpoint_path.parent, exist_ok=True)
        torch.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint for epoch {epoch} at {checkpoint_path}")

        # Save best checkpoint if applicable
        if is_best:
            best_path = self.model_dir / "best.pth" if not round else self.model_dir / f"round_{str(round)}"  / "best.pth"
            torch.save(state, best_path)
            
            # Log diagnostics for the saved checkpoint
            diagnostics = self.log_checkpoint_diagnostics(best_path, state["model_state_dict"])
            self.save_checkpoint_metadata(best_path, diagnostics)
            
            logger.info(f"Saved best checkpoint (epoch {epoch})")
            console.print(f"[green]✓[/] New best model saved (epoch {epoch})")

        # Update best metric if needed
        try:
            metric_value = metrics[self.save_metric]
        except KeyError :
            accessor = NestedDictAccess(max_depth=2)
            try:
                metric_value = accessor.get(metrics, ["classification", self.save_metric])
            except AccessError as ae:
                console.print(f"[red]✗[/] {ae}")
                raise ae
        if self.is_better(metric_value):
            self.best_metric = metric_value
            self.best_epoch = epoch

    @staticmethod
    def find_file(start_dir: str, target_filename: str, skip: Optional[str] = None) -> str | None:
        """
        Recursively search for a file with a given name starting from start_dir.
        Returns the most direct path (shortest path) to ensure deterministic results.

        Args:
            start_dir (str): The directory to start the search from.
            target_filename (str): The name of the file to search for.
            skip (Optional[str]): Directory pattern to skip during search.

        Returns:
            str | None: Full path to the file if found, else None.
        """
        found_files = []
        
        for root, dirs, files in os.walk(start_dir):
            # Skip directories if specified
            if skip and skip in root:
                continue
            if target_filename in files:
                found_files.append(os.path.join(root, target_filename))
        
        if not found_files:
            logger.info(f"No {target_filename} found in {start_dir}")
            return None
        
        if len(found_files) > 1:
            # Sort by path length (prefer more direct paths) then alphabetically for deterministic ordering
            found_files.sort(key=lambda x: (len(x.split(os.sep)), x))
            logger.warning(f"Multiple {target_filename} files found: {found_files}")
            logger.warning(f"Selecting most direct path: {found_files[0]}")
        else:
            logger.info(f"Found {target_filename} at: {found_files[0]}")
        
        return found_files[0]
    

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        load_best: bool = False,
        round: Optional[int] = None ,
        ignore: Optional[str]=None
    ) -> dict[str, Any]:
        """Load model checkpoint."""
        try:
            if load_best:
                # Use the improved get_best_checkpoint method
                try:
                    checkpoint_path = self.get_best_checkpoint()
                    if not self.validate_checkpoint(checkpoint_path):
                        raise ValueError(f"Checkpoint validation failed: {checkpoint_path}")
                    print_info(console, f"Best checkpoint found and validated at: {checkpoint_path}")
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"No best checkpoint found in {self.model_dir}. "
                        f"Please ensure a checkpoint has been saved. Details: {e}"
                    )
                console.print(f" [cyan]Loading best checkpoint: {checkpoint_path} ...[/]")
            elif round is not None:
                checkpoint_path = self.model_dir / f"round_{str(round)}" / "best.pth"
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
                console.print(f"[cyan]Loading checkpoint from round {round}, epoch {epoch}...[/]")
            elif epoch is not None:
                checkpoint_path = self.model_dir / f"epoch_{epoch}.pth"
                console.print(f"[cyan]Loading checkpoint from epoch {epoch}...[/]")
            else:
                checkpoint_path = self.model_dir / "best.pth"
                console.print("[cyan]Loading last checkpoint...[/]")

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

            # Log detailed diagnostics before loading
            model_state_dict = checkpoint["model_state_dict"]
            diagnostics = self.log_checkpoint_diagnostics(checkpoint_path, model_state_dict)
            
            # Save metadata for future reference
            self.save_checkpoint_metadata(checkpoint_path, diagnostics)

            # Load model state
            model.load_state_dict(model_state_dict)

            # Load optimizer state if provided
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load scheduler state if provided
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            console.print("[green]✓[/] Successfully loaded checkpoint")

            return checkpoint

        except FileNotFoundError as e:
            error_msg = f"Checkpoint file not found: {checkpoint_path}"
            logger.error(error_msg)
            console.print(f"[red]✗[/] {error_msg}")
            console.print(f"[yellow]Hint:[/] Check if training was completed or if the path is correct")
            raise FileNotFoundError(error_msg) from e
            
        except torch.OutOfMemoryError as e:
            error_msg = f"Out of memory while loading checkpoint: {checkpoint_path}"
            logger.error(error_msg)
            console.print(f"[red]✗[/] {error_msg}")
            console.print(f"[yellow]Hint:[/] Try reducing batch size or using CPU for checkpoint loading")
            raise torch.OutOfMemoryError(error_msg) from e
            
        except (KeyError, RuntimeError) as e:
            error_msg = f"Checkpoint format error: {str(e)}"
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {error_msg}")
            console.print(f"[red]✗[/] Checkpoint format error: {checkpoint_path}")
            console.print(f"[yellow]Details:[/] {str(e)}")
            console.print(f"[yellow]Hint:[/] Checkpoint may be corrupted or from incompatible model")
            raise ValueError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error loading checkpoint: {str(e)}"
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {error_msg}")
            console.print(f"[red]✗[/] Unexpected error loading checkpoint")
            console.print(f"[yellow]Details:[/] {str(e)}")
            console.print(f"[yellow]Path:[/] {checkpoint_path}")
            raise RuntimeError(error_msg) from e

    def get_best_checkpoint(self) -> Optional[Path]:
        """Return the path to the best checkpoint."""
        # First try the direct path
        best_path = self.model_dir / "best.pth"
        if best_path.exists():
            logger.info(f"Found best checkpoint at direct path: {best_path}")
            return best_path
        
        # If direct path doesn't exist, use find_file for recursive search
        found_path = self.find_file(str(self.model_dir), "best.pth")
        if found_path:
            logger.info(f"Found best checkpoint via search: {found_path}")
            return Path(found_path)
        
        # Provide recovery suggestions before raising error
        self.suggest_recovery_options()
        raise FileNotFoundError(f"No best checkpoint found starting from {self.model_dir}")
    
    def validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """Validate that a checkpoint file is valid and loadable."""
        try:
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
                return False
            
            # Try to load the checkpoint to validate it
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            
            # Check required keys
            required_keys = ["model_state_dict"]
            for key in required_keys:
                if key not in checkpoint:
                    logger.error(f"Checkpoint missing required key '{key}': {checkpoint_path}")
                    return False
            
            logger.info(f"Checkpoint validation successful: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {checkpoint_path}, error: {e}")
            return False
    
    @staticmethod
    def calculate_checkpoint_hash(checkpoint_path: Path) -> str:
        """Calculate MD5 hash of checkpoint file for verification."""
        try:
            hash_md5 = hashlib.md5()
            with open(checkpoint_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {checkpoint_path}: {e}")
            return "unknown"
    
    @staticmethod
    def calculate_model_state_hash(model_state_dict: Dict[str, torch.Tensor]) -> str:
        """Calculate hash of model state dict for verification."""
        try:
            # Convert model state to a consistent string representation
            state_str = ""
            for key in sorted(model_state_dict.keys()):
                tensor = model_state_dict[key]
                # Use tensor statistics for hash (more stable than raw values)
                state_str += f"{key}:{tensor.shape}:{tensor.mean().item():.8f}:{tensor.std().item():.8f};"
            
            return hashlib.md5(state_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate model state hash: {e}")
            return "unknown"
    
    def log_checkpoint_diagnostics(self, checkpoint_path: Path, model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """Log detailed checkpoint diagnostics and return info dict."""
        file_hash = self.calculate_checkpoint_hash(checkpoint_path)
        state_hash = self.calculate_model_state_hash(model_state_dict)
        file_size = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
        
        diagnostics = {
            "checkpoint_path": str(checkpoint_path),
            "file_hash": file_hash,
            "model_state_hash": state_hash,
            "file_size_bytes": file_size,
            "num_parameters": len(model_state_dict),
            "parameter_shapes": {k: list(v.shape) for k, v in model_state_dict.items()}
        }
        
        logger.info(f"Checkpoint Diagnostics:")
        logger.info(f"  Path: {checkpoint_path}")
        logger.info(f"  File Hash: {file_hash}")
        logger.info(f"  Model State Hash: {state_hash}")
        logger.info(f"  File Size: {file_size:,} bytes")
        logger.info(f"  Parameters: {len(model_state_dict)} tensors")
        
        console.print(f"[cyan]Checkpoint Diagnostics:[/]")
        console.print(f"  [green]✓[/] File Hash: {file_hash[:16]}...")
        console.print(f"  [green]✓[/] State Hash: {state_hash[:16]}...")
        console.print(f"  [green]✓[/] File Size: {file_size:,} bytes")
        
        return diagnostics
    
    def save_checkpoint_metadata(self, checkpoint_path: Path, diagnostics: Dict[str, str]) -> None:
        """Save checkpoint metadata for future verification."""
        try:
            metadata_path = checkpoint_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(diagnostics, f, indent=2)
            logger.info(f"Saved checkpoint metadata to: {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint metadata: {e}")
    
    def list_available_checkpoints(self) -> List[Path]:
        """List all available checkpoint files in the model directory."""
        checkpoints = []
        try:
            for pattern in ["*.pth", "**/*.pth"]:
                checkpoints.extend(self.model_dir.glob(pattern))
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Sort by modification time
            return checkpoints
        except Exception as e:
            logger.error(f"Failed to list checkpoints in {self.model_dir}: {e}")
            return []
    
    def suggest_recovery_options(self) -> None:
        """Suggest recovery options when checkpoint loading fails."""
        console.print(f"[cyan]Recovery Suggestions:[/]")
        
        available_checkpoints = self.list_available_checkpoints()
        if available_checkpoints:
            console.print(f"  [green]Found {len(available_checkpoints)} checkpoint files:[/]")
            for i, cp in enumerate(available_checkpoints[:5]):  # Show first 5
                console.print(f"    {i+1}. {cp.name} ({cp.stat().st_size:,} bytes)")
            if len(available_checkpoints) > 5:
                console.print(f"    ... and {len(available_checkpoints) - 5} more")
        else:
            console.print(f"  [red]No checkpoint files found in {self.model_dir}[/]")
            console.print(f"  [yellow]→[/] Make sure training was completed successfully")
            console.print(f"  [yellow]→[/] Check if the model output path is correct")
        
        console.print(f"  [yellow]→[/] Verify the experiment configuration paths")
        console.print(f"  [yellow]→[/] Check if the run_id matches the expected checkpoint location")

    def __str__(self) -> str:
        """Return string representation of the CheckpointManager."""
        return (
            f"CheckpointManager("
            f"model_dir='{self.model_dir}', "
            f"save_metric='{self.save_metric}', "
            f"mode='{self.mode}', "
            f"device='{self.device}', "
            f"best_metric={self.best_metric:.4f}, "
            f"best_epoch={self.best_epoch})"
        )
