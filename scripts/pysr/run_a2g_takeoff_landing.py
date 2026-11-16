#!/usr/bin/env python3
"""Run the PySR baseline for the A2G takeoff/landing experiment."""

from _scene_runner import run_scene


if __name__ == "__main__":
    run_scene("a2g_takeoff_landing.json")
