from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import gradio as gr


GalleryItems = List[Tuple[str, str]]
RunsByBackbone = Dict[str, Dict[str, List[str]]]
_DEFAULT_PLOT_GROUP = "all_prune_spans"
_DEFAULT_RECOVERY_VIEW = "strategy_comparison"
_BACKBONE_ORDER = ("vggt", "pi3", "vggt_omega", "mapanything", "mapanything_v1")
_DATASET_ORDER = (
    "mean",
    "scannet_dev50",
    "scannet_test50",
    "7scenes_test",
    "7scenes_train",
    "nrgbd",
    "nuscenes_val",
    "nuscenes_train",
    "waymo_validation",
    "waymo_training",
)
_DATASET_LABELS = {
    "mean": "mean",
    "scannet_dev50": "scans dev",
    "scannet_test50": "scans test",
    "7scenes_test": "7scenes",
    "7scenes_train": "7scenes train",
    "nrgbd": "nrgbd",
    "nuscenes_val": "nuscenes val",
    "nuscenes_train": "nuscenes train",
    "waymo_validation": "waymo validation",
    "waymo_training": "waymo training",
}


def _backbone_sort_key(name: str) -> Tuple[int, str]:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key == "vggtomega":
        key = "vggt_omega"
    try:
        return (_BACKBONE_ORDER.index(key), key)
    except ValueError:
        return (len(_BACKBONE_ORDER), key)


def _sort_runs_by_backbone(runs_by_backbone: RunsByBackbone) -> RunsByBackbone:
    return {
        str(backbone): {
            str(plot_group): sorted(str(recovery_view) for recovery_view in recovery_views)
            for plot_group, recovery_views in sorted(recovery_runs.items())
        }
        for backbone, recovery_runs in sorted(
            runs_by_backbone.items(), key=lambda item: _backbone_sort_key(item[0])
        )
    }


def _dataset_sort_key(name: str) -> Tuple[int, str]:
    key = str(name or "").strip()
    try:
        return (_DATASET_ORDER.index(key), key)
    except ValueError:
        return (len(_DATASET_ORDER), key)


def _dataset_label(dataset_tag: str) -> str:
    key = str(dataset_tag or "").strip()
    if key in _DATASET_LABELS:
        return _DATASET_LABELS[key]
    return key.replace("_", " ")


def _discover_dataset_specs(recovery_root: Path, backbones: Sequence[str]) -> List[Tuple[str, str, str]]:
    root = Path(recovery_root)
    discovered = set()
    for backbone in backbones:
        backbone_dir = root / str(backbone)
        if not backbone_dir.is_dir():
            continue
        for path in backbone_dir.glob("*/eval/recovery/*_recovery_strategies_by_span.png"):
            dataset_tag = path.parent.parent.parent.name
            expected = f"{dataset_tag}_recovery_strategies_by_span.png"
            if path.name == expected:
                discovered.add(dataset_tag)
    datasets = sorted(discovered, key=_dataset_sort_key)
    if datasets:
        datasets = ["mean", *datasets]
    return [(dataset_tag, dataset_tag, _dataset_label(dataset_tag)) for dataset_tag in datasets]


def _plot_path(recovery_root: Path, backbone_name: str, dataset_tag: str) -> Path:
    if str(dataset_tag) == "mean":
        return (
            Path(recovery_root)
            / str(backbone_name)
            / "eval"
            / "recovery"
            / "mean_recovery_strategies_by_span.png"
        )
    return (
        Path(recovery_root)
        / str(backbone_name)
        / str(dataset_tag)
        / "eval"
        / "recovery"
        / f"{dataset_tag}_recovery_strategies_by_span.png"
    )


def _gallery_items_from_disk(recovery_root: Path, backbone_name: str, dataset_tag: str) -> GalleryItems:
    path = _plot_path(recovery_root, backbone_name, dataset_tag)
    if not path.is_file():
        return []
    return [(str(path.resolve()), "lower + recovery strategies")]


def build_recovery_tab(
    *,
    blocks: gr.Blocks,
    recovery_root: Path,
    discover_recovery_runs: Callable[[Path], RunsByBackbone],
    recovery_plot_rows: Callable[[Path, str, str, str], Dict[str, GalleryItems]],
    tab_id: str = "recovery_results",
) -> Dict[str, Any]:
    runs_by_backbone = _sort_runs_by_backbone(discover_recovery_runs(Path(recovery_root)))
    backbone_choices = list(runs_by_backbone.keys())
    initial_backbone = backbone_choices[0] if backbone_choices else None
    plot_group_choices = (
        list(runs_by_backbone.get(str(initial_backbone), {}).keys())
        if initial_backbone
        else []
    )
    initial_plot_group = plot_group_choices[0] if plot_group_choices else None
    recovery_view_choices = (
        runs_by_backbone.get(str(initial_backbone), {}).get(str(initial_plot_group), [])
        if initial_backbone and initial_plot_group
        else []
    )
    initial_recovery_view = recovery_view_choices[0] if recovery_view_choices else None
    dataset_specs = _discover_dataset_specs(Path(recovery_root), backbone_choices)
    dataset_keys = [dataset for dataset, _dataset_tag, _label in dataset_specs]

    def _gallery_values(
        rows: Dict[str, GalleryItems],
        selected_datasets: Sequence[str],
    ) -> Tuple[Any, ...]:
        selected = {str(dataset) for dataset in (selected_datasets or [])}
        out: List[Any] = []
        for dataset, _dataset_tag, _label in dataset_specs:
            out.append(
                gr.update(
                    value=rows.get(dataset, []),
                    visible=dataset in selected,
                )
            )
        return tuple(out)

    def _rows_for_display(
        backbone_name: str,
        plot_group_name: str,
        recovery_view_name: str,
    ) -> Dict[str, GalleryItems]:
        rows = dict(
            recovery_plot_rows(
                Path(recovery_root),
                plot_group_name,
                recovery_view_name,
                backbone_name,
            )
        )
        for row_key, dataset_tag, _label in dataset_specs:
            if not rows.get(row_key):
                rows[row_key] = _gallery_items_from_disk(
                    Path(recovery_root),
                    backbone_name,
                    dataset_tag,
                )
        return rows

    def _status(
        backbone_name: str,
        plot_group_name: str,
        recovery_view_name: str,
        rows: Dict[str, GalleryItems],
        *,
        empty_suffix: str = "no images found",
    ) -> str:
        image_count = sum(len(v) for v in rows.values())
        suffix = f"images={image_count}" if image_count else empty_suffix
        return (
            f"backbone={backbone_name} | plot_group={plot_group_name} | "
            f"recovery_view={recovery_view_name} | {suffix}"
        )

    def _refresh(
        backbone_name: str,
        plot_group_name: str,
        recovery_view_name: str,
        selected_datasets: Sequence[str],
    ):
        backbone_name = str(backbone_name or "").strip()
        plot_group_name = str(plot_group_name or "").strip()
        recovery_view_name = str(recovery_view_name or "").strip()
        if not backbone_name or not plot_group_name or not recovery_view_name:
            return (
                *_gallery_values({}, selected_datasets),
                "No backbone/recovery plot group selected.",
            )
        rows = _rows_for_display(backbone_name, plot_group_name, recovery_view_name)
        return (
            *_gallery_values(rows, selected_datasets),
            _status(backbone_name, plot_group_name, recovery_view_name, rows),
        )

    def _backbone_changed(backbone_name: str, selected_datasets: Sequence[str]):
        backbone_name = str(backbone_name or "").strip()
        plot_group_choices = list(runs_by_backbone.get(backbone_name, {}).keys())
        plot_group_name = plot_group_choices[0] if plot_group_choices else ""
        recovery_view_choices = (
            runs_by_backbone.get(backbone_name, {}).get(plot_group_name, [])
            if plot_group_name
            else []
        )
        recovery_view_name = recovery_view_choices[0] if recovery_view_choices else ""
        rows = (
            _rows_for_display(backbone_name, plot_group_name, recovery_view_name)
            if recovery_view_name
            else {}
        )
        return (
            gr.update(choices=plot_group_choices, value=plot_group_name),
            gr.update(choices=recovery_view_choices, value=recovery_view_name),
            *_gallery_values(rows, selected_datasets),
            (
                _status(backbone_name, plot_group_name, recovery_view_name, rows)
                if recovery_view_name
                else f"backbone={backbone_name} | no recovery plots found"
            ),
        )

    def _plot_group_changed(
        backbone_name: str,
        plot_group_name: str,
        selected_datasets: Sequence[str],
    ):
        backbone_name = str(backbone_name or "").strip()
        plot_group_name = str(plot_group_name or "").strip()
        recovery_view_choices = runs_by_backbone.get(backbone_name, {}).get(plot_group_name, [])
        recovery_view_name = recovery_view_choices[0] if recovery_view_choices else ""
        rows = (
            _rows_for_display(backbone_name, plot_group_name, recovery_view_name)
            if recovery_view_name
            else {}
        )
        return (
            gr.update(choices=recovery_view_choices, value=recovery_view_name),
            *_gallery_values(rows, selected_datasets),
            (
                _status(backbone_name, plot_group_name, recovery_view_name, rows)
                if recovery_view_name
                else (
                    f"backbone={backbone_name} | plot_group={plot_group_name} | "
                    "no recovery views found"
                )
            ),
        )

    def _recovery_view_changed(
        backbone_name: str,
        plot_group_name: str,
        recovery_view_name: str,
        selected_datasets: Sequence[str],
    ):
        backbone_name = str(backbone_name or "").strip()
        plot_group_name = str(plot_group_name or "").strip()
        recovery_view_name = str(recovery_view_name or "").strip()
        rows = (
            _rows_for_display(backbone_name, plot_group_name, recovery_view_name)
            if backbone_name and plot_group_name and recovery_view_name
            else {}
        )
        return (
            *_gallery_values(rows, selected_datasets),
            (
                _status(backbone_name, plot_group_name, recovery_view_name, rows)
                if rows
                else (
                    f"backbone={backbone_name} | plot_group={plot_group_name} | "
                    f"recovery_view={recovery_view_name} | no images found"
                )
            ),
        )

    with gr.Tab("Recovery Results", id=tab_id) as tab_recovery:
        gr.Markdown("## Recovery Results")
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                recovery_backbone_dd = gr.Dropdown(
                    label="Backbone",
                    choices=backbone_choices,
                    value=initial_backbone,
                )
                recovery_plot_group_dd = gr.Dropdown(
                    label="Plot group",
                    choices=plot_group_choices,
                    value=initial_plot_group,
                )
                recovery_view_dd = gr.Dropdown(
                    label="Recovery view",
                    choices=recovery_view_choices,
                    value=initial_recovery_view,
                )
                recovery_dataset_cb = gr.CheckboxGroup(
                    label="Dataset",
                    choices=[
                        (label, dataset) for dataset, _dataset_tag, label in dataset_specs
                    ],
                    value=list(dataset_keys),
                )
                recovery_info = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=5):
                eval_galleries: List[gr.Gallery] = []
                for _dataset, _dataset_tag, label in dataset_specs:
                    eval_galleries.append(
                        gr.Gallery(
                            label=f"{label} · lower/recovery comparison",
                            columns=1,
                            rows=1,
                            height="auto",
                            object_fit="contain",
                            elem_classes=["results-gallery", "recovery-eval-gallery"],
                        )
                    )

        gallery_outputs = [
            *eval_galleries,
            recovery_info,
        ]
        recovery_backbone_dd.change(
            _backbone_changed,
            inputs=[recovery_backbone_dd, recovery_dataset_cb],
            outputs=[recovery_plot_group_dd, recovery_view_dd, *gallery_outputs],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        recovery_plot_group_dd.change(
            _plot_group_changed,
            inputs=[recovery_backbone_dd, recovery_plot_group_dd, recovery_dataset_cb],
            outputs=[recovery_view_dd, *gallery_outputs],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        recovery_view_dd.change(
            _recovery_view_changed,
            inputs=[
                recovery_backbone_dd,
                recovery_plot_group_dd,
                recovery_view_dd,
                recovery_dataset_cb,
            ],
            outputs=gallery_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        recovery_dataset_cb.change(
            _refresh,
            inputs=[
                recovery_backbone_dd,
                recovery_plot_group_dd,
                recovery_view_dd,
                recovery_dataset_cb,
            ],
            outputs=gallery_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        blocks.load(
            _refresh,
            inputs=[
                recovery_backbone_dd,
                recovery_plot_group_dd,
                recovery_view_dd,
                recovery_dataset_cb,
            ],
            outputs=gallery_outputs,
            show_progress="minimal",
        )

    return {"tab": tab_recovery}


def build_lora_tab(
    *,
    blocks: gr.Blocks,
    lora_root: Path,
    discover_lora_runs: Callable[[Path], RunsByBackbone],
    lora_plot_rows: Callable[[Path, str, str, str], Dict[str, GalleryItems]],
    tab_id: str = "recovery_results",
) -> Dict[str, Any]:
    return build_recovery_tab(
        blocks=blocks,
        recovery_root=lora_root,
        discover_recovery_runs=discover_lora_runs,
        recovery_plot_rows=lora_plot_rows,
        tab_id=tab_id,
    )
