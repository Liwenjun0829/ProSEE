#!/usr/bin/env python3
"""Reproduce experiment 055 using A2G takeoff and landing data."""

from _experiment_runner import run_experiments


if __name__ == "__main__":
    run_experiments(
        ["a2g_takeoff_landing.json"],
        __doc__ or "Run A2G takeoff/landing experiment 055",
    )
