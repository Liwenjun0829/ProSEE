"""Dataset loading and splitting."""

from ._core import engine

load_configured_data = engine.load_configured_data
load_joint_csv_data = engine.load_joint_csv_data
load_realdata_mat = engine.load_realdata_mat

__all__ = [
    "load_configured_data",
    "load_joint_csv_data",
    "load_realdata_mat",
]
