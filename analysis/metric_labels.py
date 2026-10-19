"""Canonical display labels for the seven evaluation metrics."""

from __future__ import annotations


SEVEN_METRIC_LABELS = {
    "ape": "APE↓ (m)",
    "are": "ARE↓ (°)",
    # Keep labels as plain Unicode text.  Matplotlib's mathtext parser is not
    # thread-safe, so mixed text/math labels can fail intermittently when
    # Gradio renders multiple figures in worker threads.
    "rpe_trans": "RPEₜ↓ (m)",
    "rpe_rot": "RPEᵣ↓ (°)",
    "acc": "Acc.↓ (m)",
    "comp": "Comp.↓ (m)",
    "chamfer_distance": "CD↓ (m)",
    "cd": "CD↓ (m)",
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
