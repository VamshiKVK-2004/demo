"""Pipeline stage registry and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PipelineStage:
    """Represents one named stage in the bias evaluation pipeline."""

    name: str
    handler: Callable[[], None]


def execute(stages: list[PipelineStage], selected: str) -> None:
    """Execute either a single stage or all stages in order."""
    for stage in stages:
        if selected in {"all", stage.name}:
            print(f"[biaseval] running stage: {stage.name}")
            stage.handler()
