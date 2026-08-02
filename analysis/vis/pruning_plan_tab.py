from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis.prune_plan import load_prune_plan_bundle
from analysis.prune_plan.eval_plots import (
    PRUNING_MEAN_METRIC_GRID,
    collect_pruning_eval_payload,
    pruning_average_strategy_keys,
    render_strategy_metric_figure,
)


GalleryItems = List[Tuple[np.ndarray, str]]

_BACKBONE_ORDER = ("vggt", "pi3", "vggt_omega")
_AVERAGE_SCOPES = ("training", "testing")
_PLOT_SPECS = (
    ("ours_single", "Ours + GT · Single"),
    ("ours_multi", "Ours + GT · Multi"),
    ("ours", "Ours + GT · Single + Multi"),
    ("others", "Ours_CKA + GT + baselines"),
)


def _backbone_sort_key(name: str) -> Tuple[int, str]:
    key = str(name)
    try:
        return (_BACKBONE_ORDER.index(key), key)
    except ValueError:
        return (len(_BACKBONE_ORDER), key)


def _scope_label(scope: str) -> str:
    if scope == "training":
        return "Training average"
    if scope == "testing":
        return "Testing average"
    return str(scope)


def _bundle_training_datasets(backbone_root: Path) -> List[str]:
    try:
        bundle = load_prune_plan_bundle(Path(backbone_root))
    except Exception:
        return []
    raw = bundle.get("dataset", "")
    if isinstance(raw, str):
        return [name.strip() for name in raw.split(",") if name.strip()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(name).strip() for name in raw if str(name).strip()]
    return []


def _result_datasets(backbone_root: Path) -> List[str]:
    return sorted(
        {
            path.parent.parent.name
            for path in Path(backbone_root).glob("*/eval/pruning")
            if path.is_dir() and any(path.glob("*/summary.json"))
        }
    )


def discover_pruning_plan_backbones(output_root: Path) -> List[str]:
    root = Path(output_root)
    if not root.is_dir():
        return []
    found = [
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and (path / "prune_plan_all.json").is_file()
        and bool(_result_datasets(path))
    ]
    return sorted(set(found), key=_backbone_sort_key)


def discover_pruning_plan_scopes(
    output_root: Path,
    backbone: str,
) -> List[str]:
    datasets = _result_datasets(Path(output_root) / str(backbone))
    return [*_AVERAGE_SCOPES, *datasets]


def resolve_pruning_scope_datasets(
    output_root: Path,
    backbone: str,
    scope: str,
) -> List[str]:
    backbone_root = Path(output_root) / str(backbone)
    available = _result_datasets(backbone_root)
    training = set(_bundle_training_datasets(backbone_root))
    if scope == "training":
        return [dataset for dataset in available if dataset in training]
    if scope == "testing":
        return [dataset for dataset in available if dataset not in training]
    return [scope] if scope in available else []


def _figure_to_rgb(fig: Any) -> np.ndarray:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    with Image.open(buffer) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _scope_plot_payload(payload: Dict[str, Any], scope: str) -> Dict[str, Any]:
    if scope in _AVERAGE_SCOPES:
        average = payload.get("dataset_average", {})
        return average if isinstance(average, dict) else {}
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        return {}
    dataset_payload = datasets.get(scope, {})
    return dataset_payload if isinstance(dataset_payload, dict) else {}


def render_pruning_plan_gallery(
    output_root: Path,
    backbone: str,
    selected_scopes: Sequence[str] | None,
) -> Tuple[GalleryItems, str]:
    """Reload current summaries and render the four pruning views in memory."""
    scopes = list(dict.fromkeys(str(scope) for scope in (selected_scopes or [])))
    if not backbone:
        return [], "Select a backbone."
    if not scopes:
        return [], "Select at least one dataset or average scope."

    backbone_root = Path(output_root) / str(backbone)
    gallery: GalleryItems = []
    messages: List[str] = []
    for scope in scopes:
        datasets = resolve_pruning_scope_datasets(
            Path(output_root), str(backbone), scope
        )
        if not datasets:
            messages.append(f"{scope}: no evaluation summaries")
            continue
        try:
            payload = collect_pruning_eval_payload(
                experiment_root=backbone_root,
                prune_plan_root=backbone_root,
                datasets=datasets,
            )
            plot_payload = _scope_plot_payload(payload, scope)
            strategies = plot_payload.get("strategies", {})
            strategy_keys = tuple(strategies) if isinstance(strategies, dict) else ()
            rendered = 0
            for plot_group, plot_label in _PLOT_SPECS:
                selected = pruning_average_strategy_keys(strategy_keys, plot_group)
                fig = render_strategy_metric_figure(
                    payload=plot_payload,
                    title=(
                        f"{backbone}: pruning strategies · "
                        f"{_scope_label(scope)}"
                    ),
                    metric_grid=PRUNING_MEAN_METRIC_GRID,
                    strategy_keys=selected,
                    compact_pruning_legend=True,
                )
                if fig is None:
                    continue
                try:
                    gallery.append(
                        (
                            _figure_to_rgb(fig),
                            f"{_scope_label(scope)} · {plot_label}",
                        )
                    )
                    rendered += 1
                finally:
                    plt.close(fig)
            messages.append(
                f"{scope}: datasets={','.join(datasets)} | plots={rendered}/4"
            )
        except Exception as exc:
            messages.append(f"{scope}: {type(exc).__name__}: {exc}")
    return gallery, f"backbone={backbone}\n" + "\n".join(messages)


def build_pruning_plan_tab(
    *,
    blocks: gr.Blocks,
    output_root: Path,
    tab_id: str = "pruning_plan",
) -> Dict[str, Any]:
    root = Path(output_root)
    backbones = discover_pruning_plan_backbones(root)
    initial_backbone = backbones[0] if backbones else ""
    initial_scopes = discover_pruning_plan_scopes(root, initial_backbone)
    initial_selection = (
        ["training"] if "training" in initial_scopes else initial_scopes[:1]
    )

    def _scope_choices(backbone: str) -> List[Tuple[str, str]]:
        return [
            (_scope_label(scope), scope)
            for scope in discover_pruning_plan_scopes(root, backbone)
        ]

    def _redraw(backbone: str, selected_scopes: Sequence[str]):
        return render_pruning_plan_gallery(
            root, str(backbone or ""), selected_scopes
        )

    def _backbone_changed(backbone: str, selected_scopes: Sequence[str]):
        scopes = discover_pruning_plan_scopes(root, str(backbone or ""))
        retained = [scope for scope in (selected_scopes or []) if scope in scopes]
        selected = retained or (["training"] if "training" in scopes else scopes[:1])
        gallery, status = render_pruning_plan_gallery(root, backbone, selected)
        return (
            gr.update(choices=_scope_choices(backbone), value=selected),
            gallery,
            status,
        )

    def _refresh(backbone: str, selected_scopes: Sequence[str]):
        latest_backbones = discover_pruning_plan_backbones(root)
        backbone_value = (
            backbone
            if backbone in latest_backbones
            else (latest_backbones[0] if latest_backbones else "")
        )
        scopes = discover_pruning_plan_scopes(root, backbone_value)
        selected = [scope for scope in (selected_scopes or []) if scope in scopes]
        selected = selected or (["training"] if "training" in scopes else scopes[:1])
        gallery, status = render_pruning_plan_gallery(
            root, backbone_value, selected
        )
        return (
            gr.update(choices=latest_backbones, value=backbone_value),
            gr.update(choices=_scope_choices(backbone_value), value=selected),
            gallery,
            status,
        )

    with gr.Tab("Pruning Plan", id=tab_id) as tab:
        gr.Markdown("## Pruning Plan")
        gr.Markdown(
            "Reload current pruning summaries and render all four views in memory. "
            "No saved plot images are used."
        )
        with gr.Row():
            with gr.Column(scale=1):
                backbone_dd = gr.Dropdown(
                    label="Backbone",
                    choices=backbones,
                    value=initial_backbone or None,
                )
                scope_cb = gr.CheckboxGroup(
                    label="Datasets / averages (multi-select)",
                    choices=_scope_choices(initial_backbone),
                    value=initial_selection,
                    type="value",
                    interactive=True,
                    info="Check one or more individual datasets, Training, or Testing.",
                )
                refresh_btn = gr.Button("Reload summaries", variant="primary")
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=5,
                )
            with gr.Column(scale=5):
                gallery = gr.Gallery(
                    label="Live pruning-plan plots",
                    columns=2,
                    height="auto",
                    object_fit="contain",
                    elem_classes=["results-gallery", "pruning-plan-gallery"],
                )

        scope_cb.change(
            _redraw,
            inputs=[backbone_dd, scope_cb],
            outputs=[gallery, status],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        backbone_dd.change(
            _backbone_changed,
            inputs=[backbone_dd, scope_cb],
            outputs=[scope_cb, gallery, status],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        refresh_btn.click(
            _refresh,
            inputs=[backbone_dd, scope_cb],
            outputs=[backbone_dd, scope_cb, gallery, status],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        blocks.load(
            _redraw,
            inputs=[backbone_dd, scope_cb],
            outputs=[gallery, status],
            show_progress="minimal",
        )

    return {
        "tab": tab,
        "backbone_dd": backbone_dd,
        "scope_cb": scope_cb,
        "gallery": gallery,
        "status": status,
        "refresh_btn": refresh_btn,
    }
