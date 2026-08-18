"""Canonical display labels for the seven evaluation metrics."""

from __future__ import annotations


SEVEN_METRIC_LABELS = {
    "ape": "APE (m) ↓",
    "are": "ARE (°) ↓",
    "rpe_trans": "RPE trans (m) ↓",
    "rpe_rot": "RPE rot (deg) ↓",
    "acc": "Acc (m) ↓",
    "comp": "Comp (m) ↓",
    "chamfer_distance": "CD (m) ↓",
    "cd": "CD (m) ↓",
}

SEVEN_METRIC_KEYS = (
    "ape",
    "are",
    "rpe_trans",
    "rpe_rot",
    "acc",
    "comp",
    "chamfer_distance",
)

