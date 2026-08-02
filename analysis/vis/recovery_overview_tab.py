from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from analysis.prune_plan.eval_plots import (
    RECOVERY_METHOD_ORDER,
    RECOVERY_METRIC_GRID,
    add_recovery_comparison_strategies,
    collect_pruning_eval_payload,
    collect_recovery_eval_payload,
    render_strategy_metric_figure,
)


_BACKBONE_ORDER = ("vggt", "pi3", "vggt_omega")
_BASELINE_STRATEGIES = (
    "no_recovery_single",
    "no_recovery_double",
    "no_recovery",
    "gt_drop",
)


def _sort_key(name: str, preferred: Sequence[str]) -> Tuple[int, str]:
    key = str(name)
    try:
        return (preferred.index(key), key)
    except ValueError:
        return (len(preferred), key)


def discover_recovery_overview_options(output_root: Path) -> Tuple[List[str], List[str]]:
    """Discover controls from plans and result folders, including unfinished runs."""
    root = Path(output_root)
    backbones: List[str] = []
    methods = set()
    if root.is_dir():
        for backbone_dir in (path for path in root.iterdir() if path.is_dir()):
            from_results = {
                path.name
                for path in backbone_dir.glob("*/eval/recovery/recovery_*")
                if path.is_dir() and path.name in RECOVERY_METHOD_ORDER
            }
            from_plans = {
                path.stem
                for path in backbone_dir.glob("recovery_*.json")
                if path.stem in RECOVERY_METHOD_ORDER
            }
            from_bundle = set()
            bundle_path = backbone_dir / "recovery_plan_all.json"
            if bundle_path.is_file():
                try:
                    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    bundle = {}
                bundle_methods = bundle.get("methods", {}) if isinstance(bundle, dict) else {}
                if isinstance(bundle_methods, dict):
                    from_bundle = {
                        str(name)
                        for name in bundle_methods
                        if str(name) in RECOVERY_METHOD_ORDER
                    }
            found = from_results | from_plans | from_bundle
            if found or (backbone_dir / "prune_plan_all.json").is_file():
                backbones.append(backbone_dir.name)
                methods.update(found)
    return (
        sorted(set(backbones), key=lambda name: _sort_key(name, _BACKBONE_ORDER)),
        sorted(methods, key=lambda name: _sort_key(name, RECOVERY_METHOD_ORDER)),
    )


def discover_recovery_overview_datasets(output_root: Path) -> List[str]:
    """Discover datasets with a recovery evaluation directory."""
    root = Path(output_root)
    datasets = {
        path.parent.parent.name
        for path in root.glob("*/*/eval/recovery")
        if path.is_dir()
    }
    return ["mean", *sorted(datasets)]


def _figure_to_rgb(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = np.asarray(rgba[..., :3], dtype=np.uint8)
    content = np.any(rgb < 250, axis=2)
    ys, xs = np.nonzero(content)
    if ys.size and xs.size:
        pad = 10
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(rgb.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(rgb.shape[1], int(xs.max()) + pad + 1)
        rgb = rgb[y0:y1, x0:x1]
    return rgb.copy()


def render_recovery_overview(
    output_root: Path,
    selected_backbones: Sequence[str] | None,
    selected_methods: Sequence[str] | None,
    dataset: str = "mean",
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """Reload summaries and redraw one recovery comparison figure per backbone."""
    backbones = list(dict.fromkeys(str(name) for name in (selected_backbones or [])))
    methods = list(dict.fromkeys(str(name) for name in (selected_methods or [])))
    dataset = str(dataset or "mean").strip()
    if not backbones:
        return [], "Select at least one backbone."
    if not methods:
        return [], "Select at least one recovery method."

    gallery: List[Tuple[np.ndarray, str]] = []
    messages: List[str] = []
    for backbone in backbones:
        backbone_root = Path(output_root) / backbone
        try:
            pruning = collect_pruning_eval_payload(
                experiment_root=backbone_root,
                prune_plan_root=backbone_root,
            )
            recovery = collect_recovery_eval_payload(experiment_root=backbone_root)
            recovery = add_recovery_comparison_strategies(recovery, pruning)
            if dataset == "mean":
                plot_payload = recovery.get("dataset_average", {})
                title_scope = "averaged over datasets"
            else:
                dataset_payloads = recovery.get("datasets", {})
                plot_payload = (
                    dataset_payloads.get(dataset, {})
                    if isinstance(dataset_payloads, dict)
                    else {}
                )
                title_scope = dataset
            available = (
                plot_payload.get("strategies", {})
                if isinstance(plot_payload, dict)
                else {}
            )
            selected = [
                key
                for key in (*_BASELINE_STRATEGIES, *methods)
                if isinstance(available, dict) and key in available
            ]
            fig = render_strategy_metric_figure(
                payload=plot_payload,
                title=f"{backbone}: recovery strategies · {title_scope}",
                metric_grid=RECOVERY_METRIC_GRID,
                strategy_keys=selected,
                legend_fontsize=11,
            )
            if fig is None:
                messages.append(f"{backbone}: no matching recovery data")
                continue
            try:
                gallery.append((_figure_to_rgb(fig), backbone))
            finally:
                plt.close(fig)
            missing = [method for method in methods if method not in available]
            suffix = f" (missing: {', '.join(missing)})" if missing else ""
            messages.append(
                f"{backbone} / {dataset}: redrawn from current summaries{suffix}"
            )
        except Exception as exc:
            messages.append(f"{backbone}: {type(exc).__name__}: {exc}")
    return gallery, "\n".join(messages)


def build_recovery_overview_tab(*, blocks: gr.Blocks, output_root: Path) -> Dict[str, Any]:
    backbones, methods = discover_recovery_overview_options(Path(output_root))
    datasets = discover_recovery_overview_datasets(Path(output_root))

    def _redraw(
        selected_backbones: Sequence[str],
        selected_methods: Sequence[str],
        dataset: str,
    ):
        return render_recovery_overview(
            Path(output_root), selected_backbones, selected_methods, dataset
        )

    def _refresh_options(
        selected_backbones: Sequence[str],
        selected_methods: Sequence[str],
        dataset: str,
    ):
        latest_backbones, latest_methods = discover_recovery_overview_options(
            Path(output_root)
        )
        latest_datasets = discover_recovery_overview_datasets(Path(output_root))
        backbone_value = [
            name for name in (selected_backbones or []) if name in latest_backbones
        ] or latest_backbones
        method_value = [
            name for name in (selected_methods or []) if name in latest_methods
        ] or latest_methods
        dataset_value = dataset if dataset in latest_datasets else "mean"
        gallery_value, status_value = render_recovery_overview(
            Path(output_root), backbone_value, method_value, dataset_value
        )
        return (
            gr.update(choices=latest_backbones, value=backbone_value),
            gr.update(
                choices=[
                    (name.removeprefix("recovery_"), name)
                    for name in latest_methods
                ],
                value=method_value,
            ),
            gr.update(choices=latest_datasets, value=dataset_value),
            gallery_value,
            status_value,
        )

    with gr.Tab("Recovery Overview", id="recovery_overview") as tab:
        gr.Markdown("## Recovery Overview")
        gr.Markdown(
            "Compare recovery methods across backbones. Every update reloads the latest "
            "evaluation summaries and redraws the plots; saved PNG files are not used."
        )
        with gr.Row():
            with gr.Column(scale=1):
                backbone_cb = gr.CheckboxGroup(
                    label="Backbones",
                    choices=backbones,
                    value=backbones,
                )
                method_cb = gr.CheckboxGroup(
                    label="Recovery methods",
                    choices=[(name.removeprefix("recovery_"), name) for name in methods],
                    value=methods,
                )
                dataset_dd = gr.Dropdown(
                    label="Dataset",
                    choices=datasets,
                    value="mean",
                    multiselect=False,
                )
                refresh_btn = gr.Button("Refresh latest data", variant="primary")
                status = gr.Textbox(label="Status", interactive=False, lines=4)
            with gr.Column(scale=5):
                gallery = gr.Gallery(
                    label="Recovery comparison by backbone",
                    columns=1,
                    height="auto",
                    object_fit="contain",
                    elem_classes=[
                        "results-gallery",
                        "recovery-eval-gallery",
                        "recovery-overview-gallery",
                    ],
                )

        for event in (backbone_cb.change, method_cb.change, dataset_dd.change):
            event(
                _redraw,
                inputs=[backbone_cb, method_cb, dataset_dd],
                outputs=[gallery, status],
                show_progress="minimal",
                trigger_mode="always_last",
            )
        refresh_btn.click(
            _refresh_options,
            inputs=[backbone_cb, method_cb, dataset_dd],
            outputs=[backbone_cb, method_cb, dataset_dd, gallery, status],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        blocks.load(
            _redraw,
            inputs=[backbone_cb, method_cb, dataset_dd],
            outputs=[gallery, status],
            show_progress="minimal",
        )

    return {
        "tab": tab,
        "backbone_cb": backbone_cb,
        "method_cb": method_cb,
        "dataset_dd": dataset_dd,
        "gallery": gallery,
        "status": status,
    }
