#!/usr/bin/env python3
"""Reproduce the maritime measured-MAT experiment 060."""

from _experiment_runner import run_experiments


if __name__ == "__main__":
    run_experiments(
        ["maritime.json"],
        __doc__ or "Run the archived maritime experiment",
    )
