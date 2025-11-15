# surogate/eval/results.py
"""Utilities for viewing and analyzing evaluation results."""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box

from surogate.utils.logger import get_logger

logger = get_logger()
console = Console()


def list_results(results_dir: str = "eval_results") -> List[Path]:
    """
    List all available evaluation results.

    Args:
        results_dir: Directory containing results

    Returns:
        List of result file paths
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return []

    json_files = sorted(results_path.glob("eval_*.json"), reverse=True)
    return json_files


def display_results_list(results: List[Path], results_dir: str):
    """
    Display list of available results.

    Args:
        results: List of result file paths
        results_dir: Results directory
    """
    if not results:
        console.print("[yellow]No evaluation results found[/yellow]")
        return

    console.print(f"\n[bold cyan]Available Evaluation Results[/bold cyan] ({results_dir})\n")

    table = Table(box=box.ROUNDED)
    table.add_column("#", style="dim", width=4)
    table.add_column("Filename", style="cyan")
    table.add_column("Date", style="green")
    table.add_column("Targets", justify="right")
    table.add_column("Metrics", justify="right")

    for i, result_file in enumerate(results, 1):
        # Try to load basic info
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            timestamp = data.get('timestamp', 'N/A')
            num_targets = data.get('num_targets', 'N/A')
            num_metrics = data.get('num_metrics', 'N/A')

            table.add_row(
                str(i),
                result_file.name,
                timestamp[:19] if timestamp != 'N/A' else 'N/A',
                str(num_targets),
                str(num_metrics)
            )
        except:
            table.add_row(
                str(i),
                result_file.name,
                "Error loading",
                "-",
                "-"
            )

    console.print(table)
    console.print(f"\n[dim]Use 'surogate eval --view <filename>' to view details[/dim]")


def load_result(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load evaluation result from file.

    Args:
        filepath: Path to result file

    Returns:
        Result dictionary or None
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load result: {e}")
        return None


def display_results(filepath: str):
    """
    Display evaluation results in a nice format.

    Args:
        filepath: Path to result file
    """
    result = load_result(filepath)
    if not result:
        return

    console.print(f"\n[bold cyan]📊 Evaluation Report[/bold cyan]")
    console.print(f"[dim]File: {filepath}[/dim]\n")

    # Basic info
    console.print(f"[bold]Dataset:[/bold] {result.get('dataset', 'N/A')}")
    console.print(f"[bold]Type:[/bold] {result.get('dataset_type', 'N/A')}")
    console.print(f"[bold]Test Cases:[/bold] {result.get('num_test_cases', 0)}")
    console.print(f"[bold]Timestamp:[/bold] {result.get('timestamp', 'N/A')}\n")

    # Results for each target
    for target_result in result.get('results', []):
        target_name = target_result.get('target', 'Unknown')
        model = target_result.get('model', 'N/A')

        console.print(f"\n[bold green]🎯 Target: {target_name}[/bold green] [dim]({model})[/dim]")

        # Create metrics table
        table = Table(box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Avg Score", justify="right")
        table.add_column("Success Rate", justify="right")
        table.add_column("Status", justify="center")

        metrics_summary = target_result.get('metrics_summary', {})
        for metric_name, metric_data in metrics_summary.items():
            if 'error' in metric_data:
                table.add_row(
                    metric_name,
                    "N/A",
                    "N/A",
                    "[red]❌ Failed[/red]"
                )
            else:
                avg_score = metric_data.get('avg_score', 0)
                success_rate = metric_data.get('success_rate', 0)

                # Color code based on performance
                if success_rate >= 0.8:
                    status = "[green]✅ Excellent[/green]"
                    score_color = "green"
                elif success_rate >= 0.6:
                    status = "[yellow]⚠️  Good[/yellow]"
                    score_color = "yellow"
                else:
                    status = "[red]❌ Needs Work[/red]"
                    score_color = "red"

                table.add_row(
                    metric_name,
                    f"[{score_color}]{avg_score:.3f}[/{score_color}]",
                    f"[{score_color}]{success_rate:.3f}[/{score_color}]",
                    status
                )

        console.print(table)

    console.print()


def compare_results(filepath1: str, filepath2: str):
    """
    Compare two evaluation results.

    Args:
        filepath1: First result file
        filepath2: Second result file
    """
    result1 = load_result(filepath1)
    result2 = load_result(filepath2)

    if not result1 or not result2:
        logger.error("Failed to load one or both results")
        return

    console.print("\n[bold cyan]📊 Comparison Report[/bold cyan]\n")
    console.print(f"[dim]File 1: {Path(filepath1).name}[/dim]")
    console.print(f"[dim]File 2: {Path(filepath2).name}[/dim]\n")

    # Compare each metric
    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Result 1", justify="right")
    table.add_column("Result 2", justify="right")
    table.add_column("Change", justify="right")

    # Get metrics from first target of each result
    metrics1 = result1.get('results', [{}])[0].get('metrics_summary', {})
    metrics2 = result2.get('results', [{}])[0].get('metrics_summary', {})

    for metric_name in metrics1.keys():
        if metric_name in metrics2:
            score1 = metrics1[metric_name].get('avg_score', 0)
            score2 = metrics2[metric_name].get('avg_score', 0)
            change = score2 - score1

            change_str = f"{change:+.3f}"
            if change > 0.01:
                change_color = "green"
                arrow = "↑"
            elif change < -0.01:
                change_color = "red"
                arrow = "↓"
            else:
                change_color = "white"
                arrow = "→"

            table.add_row(
                metric_name,
                f"{score1:.3f}",
                f"{score2:.3f}",
                f"[{change_color}]{arrow} {change_str}[/{change_color}]"
            )

    console.print(table)
    console.print()