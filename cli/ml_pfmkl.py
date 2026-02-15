#!/usr/bin/env python3
"""CLI for patellofemoral KL grade (knee-level)."""

from cli.ml_knee_outcome_runner import run_knee_outcome_cli


def main(argv: list[str] | None = None) -> None:
    run_knee_outcome_cli(outcome_column="Grade", default_sheet="PFM KL", argv=argv)


if __name__ == "__main__":
    main()
