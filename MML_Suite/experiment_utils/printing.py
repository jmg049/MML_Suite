import os
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import floor
from typing import Any, Dict, List, Optional

import numpy as np
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.progress import BarColumn, Column, Progress, SpinnerColumn, TaskID, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
DEBUG = False  # Global debug flag
@dataclass
class TaskInfo:
    """Store task information including ID and colors"""

    id: TaskID
    description: str
    bar_column: Column


class ProgressManager:
    """Manages progress tracking with Rich's Live display with per-task colors"""

    def __init__(self, console: Console):
        self.console = console
        self.tasks: Dict[str, TaskID] = {}

        # Initialize progress with the default bar column
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="green", pulse_style="blue"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
            disable=False,
        )
        self.progress.start()

    def add_task(
        self,
        description: str,
        total: int,
        style: str = "light_slate_blue",
    ) -> TaskID:
        """
        Add a new task to track with specific colors

        Args:
            description: Task description
            total: Total steps for the task
            complete_style: Color for the completed portion of the bar
            finished_style: Color when task is complete
            pulse_style: Color of the pulse animation
        """
        # Create the task with custom styles via the style parameter
        task_id = self.progress.add_task(
            description,
            total=total,
            style=style,  # This will color the progress bar
        )

        self.tasks[description] = task_id
        self.progress.refresh()
        return task_id

    def stop(self):
        """Stop the progress tracking"""
        if self.progress:
            self.progress.stop()
        self.tasks.clear()

    def update_task(self, description: str, advance: int = 1):
        """Update a task's progress"""
        task_id = self.tasks.get(description)
        if task_id is not None:
            self.progress.update(task_id, advance=advance)

    def complete_task(self, description: str, persist: bool = False):
        """Mark a task as complete"""
        task_id = self.tasks.get(description)
        if task_id is not None:
            if not persist:
                self.progress.remove_task(task_id)
                del self.tasks[description]
            else:
                self.progress.update(task_id=task_id, completed=100)

    def get_task_id(self, description: str) -> Optional[TaskID]:
        """Get the task ID for a given description"""
        return self.tasks.get(description)


class EnhancedConsole:
    """Enhanced console with progress tracking and formatted output"""

    def __init__(
        self,
        theme: Optional[Theme] = None,
        width: Optional[int] = None,
        record: bool = False,
        log_path: Optional[str] = None,
    ):
        # Force terminal output
        console_kwargs = {
            "force_terminal": True,
            "color_system": "auto",
            "highlight": False,  # Disable syntax highlighting which might interfere
        }
        if theme:
            console_kwargs["theme"] = theme
        if width:
            console_kwargs["width"] = width

        self.console = Console(**console_kwargs)
        self.progress_manager = ProgressManager(self.console)

        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.console.file = open(log_path, "w", encoding="utf-8")

    def _print_with_prefix(self, prefix: str, style: str, message: str, **kwargs: Any):
        """Print a message with a styled prefix"""
        prefix_text = Text(f"[{prefix}] ", style=style)
        message_text = Text.from_markup(message)
        self.console.print(prefix_text + message_text, **kwargs)

    def print(self, message, **kwargs):
        self.console.print(message, **kwargs)

    def rule(self, message, **kwargs):
        self.console.rule(message, **kwargs)

    def info(self, message: str, **kwargs: Any):
        self._print_with_prefix("INFO", "info_prefix", message, **kwargs)

    def warning(self, message: str, **kwargs: Any):
        self._print_with_prefix("WARN", "warning_prefix", message, **kwargs)

    def error(self, message: str, **kwargs: Any):
        self._print_with_prefix("ERROR", "error_prefix", message, **kwargs)

    def success(self, message: str, **kwargs: Any):
        self._print_with_prefix("SUCCESS", "success_prefix", message, **kwargs)

    def print_table(self, title: str, columns: List[str], rows: List[List[str]]):
        """Print a formatted table"""
        table = Table(title=title)
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*row)
        self.console.print(table)

    def confusion_matrix_to_rich_table(
        self, confusion_matrix: np.ndarray, condition: str, class_labels: Optional[List[str]] = None
    ):
        """Convert a confusion matrix to a Rich table"""
        if class_labels and len(class_labels) != confusion_matrix.shape[0]:
            raise ValueError("Number of class labels must match matrix dimensions")

        table = Table(
            title=f"Confusion Matrix - {condition.upper()}", box=box.SIMPLE, row_styles=["table_dim", "table_bright"]
        )

        if class_labels is None:
            class_labels = [f"Class {i}" for i in range(confusion_matrix.shape[0])]

        table.add_column("Predicted/Actual", justify="center")
        for label in class_labels:
            table.add_column(label, justify="center")

        for i, label in enumerate(class_labels):
            # each value should be formatted as .2E if the value is greater than or equal to 1E3, otherwise as a regular int

            formatted_values = []
            for j in range(confusion_matrix.shape[1]):
                value = confusion_matrix[i][j]
                if value >= 1e3:
                    formatted_values.append(f"{value:.2E}")
                else:
                    formatted_values.append(f"{int(value)}")

            table.add_row(label, *formatted_values)

        return table

    def display_training_metrics(self, metrics: Dict[str, Any]):
        """Display training metrics in a formatted table"""
        metric_pairs = []
        confusion_table = None

        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                confusion_table = self.confusion_matrix_to_rich_table(value)
                continue

            if isinstance(value, (np.generic, float)):
                value = f"{float(value):.4f}"
            metric_pairs.append((str(key), str(value)))

        # Calculate optimal column layout
        max_columns = max(1, self.console.width // 30 // 2)
        table = Table(title="Training Metrics", show_header=True, header_style="bold white")

        for _ in range(max_columns):
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", justify="right", style="bold yellow")

        for i in range(0, len(metric_pairs), max_columns):
            row = []
            for j in range(max_columns):
                if i + j < len(metric_pairs) and "ConfusionMatrix" not in metric_pairs[i + j]:
                    row.extend(metric_pairs[i + j])
                else:
                    row.extend(["", ""])
            table.add_row(*row)

        self.console.print(table)
        if confusion_table:
            self.console.print(confusion_table)

    def display_validation_metrics(self, metrics: Dict[str, Dict[str, Any]], skip_conditions: Optional[list[str]] = None):
        """Display validation metrics grouped by condition"""

        # for group, group_data in metrics.items():
        grouped_metrics = defaultdict(dict)

        loss_metrics = {}
   
        for key, value in metrics.items():
            if key == "loss" or key  in ["cosine", "mse", "mae", "mmd", "moments_loss"]:
                loss_metrics[key] = value
                continue

            match = re.match(r"(.+?)_([A-Z]+)$", key)
            if match:
                metric_name, condition = match.groups()
                if skip_conditions and (condition.lower() in skip_conditions or condition.upper() in skip_conditions):
                    continue
                grouped_metrics[condition][metric_name] = value

        tables = []

        # Add loss metrics table if present
        if loss_metrics:
            loss_table = Table(title="Loss Metrics", show_header=True, header_style="bold white")
            loss_table.add_column("Metric", style="bold cyan")
            loss_table.add_column("Value", justify="right", style="bold yellow")

            for key, value in loss_metrics.items():
                if isinstance(value, float):
                    value = f"{value:.4f}"
                loss_table.add_row(str(key), str(value))
            tables.append(loss_table)

        # Add condition-specific metric tables
        sorted_conditions = sorted(grouped_metrics.keys(), key=lambda x: (-len(x), x))
        confusion_matrices = {}
        for condition in sorted_conditions:
            condition_metrics = grouped_metrics[condition]
            table = Table(title=f"Metrics - Condition: {condition}", show_header=True, header_style="bold white")
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", justify="right", style="bold yellow")

            for key, value in condition_metrics.items():
                if "ConfusionMatrix" in key:
                    confusion_table = self.confusion_matrix_to_rich_table(value, condition)
                    confusion_matrices[condition] = confusion_table
                    continue

                if isinstance(value, (np.generic, float)):
                    value = f"{float(value):.4f}"
                table.add_row(str(key), str(value))

            tables.append(table)

        if len(confusion_matrices) > 0:
            for condition, confusion_table in confusion_matrices.items():
                tables.append(confusion_table)

        self.console.print(Columns(tables))

    def start_task(self, description: str, total: int, style="light slate blue") -> TaskID:
        return self.progress_manager.add_task(description=description, total=total, style=style)

    def update_task(self, description: str, advance: int = 1):
        self.progress_manager.update_task(description, advance)
        # Force a print to ensure output

    def complete_task(self, description: str, persist: bool = False):
        self.progress_manager.complete_task(description, persist)

    def stop_progress(self):
        self.progress_manager.stop()

# Logging/console printing helpers

def print_debug(console, message: str) -> None:
    global DEBUG
    if DEBUG:
        console.print(f"[blue][DEBUG][/]: {message}")

def print_info(console, message: str) -> None:
    console.print(f"[cyan][INFO][/]: {message}")


def print_warning(console, message: str) -> None:
    console.print(f"[yellow][WARNING][/]: {message}")


def print_success(console, message: str) -> None:
    console.print(f"[green][SUCCESS][/]: {message}")


def print_error(console, message: str) -> None:
    console.print(f"[red][ERROR][/]: {message}")


def print_metric_summary(console: EnhancedConsole, epoch: int, metrics: dict[str, float], split: str, skip_conditions: Optional[list[str]] = None) -> None:
    console.rule(f"Metrics Summary for Epoch {epoch} - {split.upper()}")
    
    console.display_validation_metrics(
        metrics, skip_conditions=skip_conditions
    )


def log_epoch_time(console, logger, client_id: int, epoch: int, time_sec: float) -> None:
    msg = f"Client {client_id} - Epoch {epoch} completed in {int(time_sec)} seconds"
    print_info(console, msg)
    logger.info(msg)


def log_total_time(console, logger, client_id: int, total_time: float) -> None:
    msg = f"Client {client_id} - Total training time: {int(total_time)} seconds"
    print_success(console, msg)
    logger.info(msg)



# Singleton management
class ConsoleSingleton:
    _instance: Optional[EnhancedConsole] = None

    @classmethod
    def get_console(cls, theme: Optional[Theme] = None, width: Optional[int] = None) -> EnhancedConsole:
        if cls._instance is None:
            cls._instance = EnhancedConsole(theme=theme, width=width)
        return cls._instance

    @classmethod
    def configure(
        cls,
        theme: Optional[Theme] = None,
        record: bool = False,
        log_path: Optional[str] = None,
        width: Optional[int] = None,
    ):
        cls._instance = EnhancedConsole(theme=theme, width=width, record=record, log_path=log_path)


@lru_cache(maxsize=10)
def get_table_width(percentage: float = 0.8) -> int:
    """Get a portion of the terminal width"""
    console = Console()
    return floor(console.width * percentage)


def get_console() -> EnhancedConsole:
    from .themes import THEME, WIDTH_SCALE

    return ConsoleSingleton.get_console(theme=THEME, width=get_table_width(WIDTH_SCALE))


def configure_console(
    theme: Optional[Theme] = None, record: bool = False, log_path: Optional[str] = None, width: Optional[int] = None
):
    ConsoleSingleton.configure(theme, record, log_path, width)
