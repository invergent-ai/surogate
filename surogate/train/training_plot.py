import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_log(log_file: str | Path) -> dict:
    """Load and parse a training log JSON file."""
    with open(log_file) as f:
        log_data = json.load(f)

    result = collections.defaultdict(list)
    for entry in log_data:
        kind = entry["log"]
        result[kind].append(entry)

    return result


def extract_over_step(data: list[dict], key: str) -> tuple[list, list]:
    """Extract step numbers and values for a given key from log entries."""
    steps = []
    values = []
    for entry in data:
        steps.append(entry["step"])
        values.append(entry[key])
    return steps, values


def generate_training_plot(log_file: str | Path, output_file: str | Path) -> None:
    """
    Generate a training plot from a log file and save it to the output file.

    Args:
        log_file: Path to the training log JSON file.
        output_file: Path where the plot image will be saved.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    log_data = load_log(log_file)

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    # Plot training loss
    if "step" in log_data and log_data["step"]:
        steps, losses = extract_over_step(log_data["step"], "loss")
        steps = np.asarray(steps, dtype=np.float64)
        losses = np.asarray(losses, dtype=np.float64)
        plt.plot(steps, losses, c=cmap(0), linewidth=1, alpha=0.5)

        # Add smoothed training loss (replace NaN/Inf with interpolated values for convolution)
        smoothing = 10
        if len(losses) > 2 * smoothing + 1:
            clean = losses.copy()
            bad = ~np.isfinite(clean)
            if bad.any() and not bad.all():
                clean[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), clean[~bad])
            smoothed = np.convolve(clean, np.ones(2 * smoothing + 1) / (2 * smoothing + 1), mode="valid")
            plt.plot(steps[smoothing:-smoothing], smoothed, c=cmap(0), linewidth=3, label="Training loss")
        else:
            plt.plot(steps, losses, c=cmap(0), linewidth=3, label="Training loss")

    # Plot validation loss
    if "eval" in log_data and log_data["eval"]:
        steps, losses = extract_over_step(log_data["eval"], "loss")
        plt.plot(steps, losses, c=cmap(1), linewidth=3, marker="o", label="Validation loss")

    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Run")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
