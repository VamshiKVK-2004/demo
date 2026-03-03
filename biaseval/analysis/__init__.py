"""Analysis package."""

from biaseval.analysis.representation import run as run_representation
from biaseval.analysis.stereotype import run as run_stereotype


def run() -> None:
    """Entry function for analysis stage."""
    run_stereotype()
    run_representation()
