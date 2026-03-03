"""Metrics package."""

from biaseval.metrics.aggregate import run as run_aggregate


def run() -> None:
    """Entry function for metrics stage."""
    run_aggregate()
