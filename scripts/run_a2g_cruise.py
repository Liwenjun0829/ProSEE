#!/usr/bin/env python3
"""Reproduce experiment 056 using A2G cruise data."""

from _experiment_runner import run_experiments


if __name__ == "__main__":
    run_experiments(
        ["a2g_cruise.json"],
        __doc__ or "Run A2G cruise experiment 056",
    )
