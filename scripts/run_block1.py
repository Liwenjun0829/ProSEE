#!/usr/bin/env python3
"""Reproduce the Block1 experiment 059."""

from _experiment_runner import run_experiments


if __name__ == "__main__":
    run_experiments(
        ["block1.json"],
        __doc__ or "Run the archived Block1 experiment",
    )
