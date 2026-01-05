#!/usr/bin/env python3
"""CLI for patellofemoral KL grade (knee-level)."""

from typing import Optional

from cli.ml_knee_outcome_runner import run_knee_outcome_cli


def main(argv: Optional[list[str]] = None) -> None:
    run_knee_outcome_cli(outcome_column="Grade", default_sheet="PFM KL", argv=argv)


if __name__ == "__main__":
    main()
