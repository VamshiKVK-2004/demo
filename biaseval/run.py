"""CLI entrypoint for running the biaseval pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
import yaml

from biaseval import analysis, dashboard, data, llm, metrics, preprocess, validation, viz
from biaseval.pipeline import PipelineStage, execute

VALID_STAGES = ["all", "data", "preprocess", "llm", "analysis", "metrics", "validation", "viz", "dashboard"]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run biaseval pipeline stages")
    parser.add_argument("--stage", choices=VALID_STAGES, default="all", help="Pipeline stage to run")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="Directory containing weights.yaml and experiments.yaml",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    weights = _load_yaml(args.config_dir / "weights.yaml")
    experiments = _load_yaml(args.config_dir / "experiments.yaml")
    print(f"[biaseval] loaded weights config keys: {list(weights.keys())}")
    print(f"[biaseval] loaded experiment config keys: {list(experiments.keys())}")

    stages = [
        PipelineStage("data", data.run),
        PipelineStage("preprocess", preprocess.run),
        PipelineStage("llm", llm.run),
        PipelineStage("analysis", analysis.run),
        PipelineStage("metrics", metrics.run),
        PipelineStage("validation", validation.run),
        PipelineStage("viz", viz.run),
        PipelineStage("dashboard", dashboard.run),
    ]
    execute(stages, args.stage)


if __name__ == "__main__":
    main()
