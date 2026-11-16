#!/usr/bin/env python3
"""Reproduce the Square1 experiment 058."""

from _experiment_runner import run_experiments


if __name__ == "__main__":
    run_experiments(
        ["square1.json"],
        __doc__ or "Run the archived Square1 experiment",
    )
