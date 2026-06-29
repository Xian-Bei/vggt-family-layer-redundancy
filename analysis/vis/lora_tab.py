from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import gradio as gr


GalleryItems = List[Tuple[str, str]]
_POSE_METRICS = ("ape", "are", "rpe_trans", "rpe_rot")
_POINT_METRICS = ("acc", "comp", "cd", "nc")
_DEPTH_METRICS = ("depth_abs_rel", "depth_delta1")
_METRIC_ORDER = (
    *_POSE_METRICS,
    *_DEPTH_METRICS,
    *_POINT_METRICS,
    "chamfer_distance",
)
_BACKBONE_ORDER = ("vggt", "pi3", "vggt_omega")
_DATASETS = (
    ("scans_dev50", "scannet_dev50", "scans dev"),
    ("scans_test50", "scannet_test50", "scans test"),
    ("7scenes_test", "7scenes_test", "7scenes"),
    ("nrgbd", "nrgbd", "nrgbd"),
    ("nuscenes_val", "nuscenes_val", "nuscenes val"),
    ("waymo_validation", "waymo_validation", "waymo validation"),
)
_DATASET_KEYS = tuple(dataset for dataset, _dataset_tag, _label in _DATASETS)


def _backbone_sort_key(name: str) -> Tuple[int, str]:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key == "vggtomega":
        key = "vggt_omega"
    try:
        return (_BACKBONE_ORDER.index(key), key)
    except ValueError:
        return (len(_BACKBONE_ORDER), key)


def _sort_models(models: Sequence[str]) -> List[str]:
    return sorted([str(m) for m in models], key=_backbone_sort_key)


def _sort_runs_by_plan(runs_by_plan: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        str(plan): _sort_models(models)
        for plan, models in sorted(runs_by_plan.items())
    }


def _metric_sort_key(metric: str) -> Tuple[int, str]:
    key = str(metric)
    try:
        return (_METRIC_ORDER.index(key), key)
    except ValueError:
        return (len(_METRIC_ORDER), key)


def _items_for_metrics(items: GalleryItems, metrics: Tuple[str, ...]) -> GalleryItems:
    by_metric = {str(item[1]): item for item in items}
    return [by_metric[m] for m in metrics if m in by_metric]


def _gallery_items_from_disk(run_dir: Path, dataset_tag: str) -> GalleryItems:
    prefix = f"eval_plot_{dataset_tag}_"
    items: GalleryItems = []
    for path in sorted(run_dir.glob(f"{prefix}*.png")):
        metric = path.stem[len(prefix):]
        items.append((str(path.resolve()), metric))
    return sorted(items, key=lambda item: _metric_sort_key(item[1]))


def build_lora_tab(
    *,
    blocks: gr.Blocks,
    lora_root: Path,
    discover_lora_runs: Callable[[Path], Dict[str, List[str]]],
    lora_plot_rows: Callable[[Path, str, str], Dict[str, GalleryItems]],
) -> Dict[str, Any]:
    runs_by_plan = _sort_runs_by_plan(discover_lora_runs(Path(lora_root)))
    plan_choices = list(runs_by_plan.keys())
    initial_plan = plan_choices[0] if plan_choices else None
    model_choices = runs_by_plan.get(str(initial_plan), []) if initial_plan else []
    initial_model = model_choices[0] if model_choices else None

    def _gallery_values(
        rows: Dict[str, GalleryItems],
        selected_datasets: Sequence[str],
    ) -> Tuple[Any, ...]:
        selected = {str(dataset) for dataset in (selected_datasets or [])}
        out: List[Any] = []
        for dataset, _dataset_tag, _label in _DATASETS:
            items = rows.get(dataset, [])
            visible = dataset in selected
            out.extend(
                [
                    gr.update(
                        value=_items_for_metrics(items, _POSE_METRICS), visible=visible
                    ),
                    gr.update(
                        value=_items_for_metrics(items, _POINT_METRICS), visible=visible
                    ),
                    gr.update(
                        value=_items_for_metrics(items, _DEPTH_METRICS), visible=visible
                    ),
                ]
            )
        out.append(rows.get("training", []))
        return tuple(out)

    def _rows_for_display(plan_name: str, model_name: str) -> Dict[str, GalleryItems]:
        rows = dict(lora_plot_rows(Path(lora_root), plan_name, model_name))
        run_dir = Path(lora_root) / str(model_name) / "scannet_dev50" / "train" / str(plan_name)
        for row_key, dataset_tag, _label in _DATASETS:
            if not rows.get(row_key):
                rows[row_key] = _gallery_items_from_disk(run_dir, dataset_tag)
        return rows

    def _refresh(
        plan_name: str,
        model_name: str,
        selected_datasets: Sequence[str],
    ):
        plan_name = str(plan_name or "").strip()
        model_name = str(model_name or "").strip()
        if not plan_name or not model_name:
            return (
                *_gallery_values({}, selected_datasets),
                "No pruning plan/model selected.",
            )
        rows = _rows_for_display(plan_name, model_name)
        return (
            *_gallery_values(rows, selected_datasets),
            (
                f"plan={plan_name} | model={model_name} | "
                f"images={sum(len(v) for v in rows.values())}"
            ),
        )

    def _plan_changed(plan_name: str, selected_datasets: Sequence[str]):
        plan_name = str(plan_name or "").strip()
        choices = _sort_models(runs_by_plan.get(plan_name, []))
        model_name = choices[0] if choices else ""
        rows = _rows_for_display(plan_name, model_name) if model_name else {}
        return (
            gr.update(choices=choices, value=model_name),
            *_gallery_values(rows, selected_datasets),
            (
                f"plan={plan_name} | model={model_name} | "
                f"images={sum(len(v) for v in rows.values())}"
                if model_name
                else f"plan={plan_name} | no trained models found"
            ),
        )

    with gr.Tab("LoRA Results") as tab_lora:
        gr.Markdown("## LoRA Results")
        with gr.Row():
            with gr.Column(scale=1):
                lora_plan_dd = gr.Dropdown(
                    label="Pruning plan",
                    choices=plan_choices,
                    value=initial_plan,
                )
                lora_model_dd = gr.Dropdown(
                    label="Model",
                    choices=model_choices,
                    value=initial_model,
                )
                lora_dataset_cb = gr.CheckboxGroup(
                    label="Dataset",
                    choices=[
                        (label, dataset) for dataset, _dataset_tag, label in _DATASETS
                    ],
                    value=list(_DATASET_KEYS),
                )
                lora_info = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=5):
                eval_galleries: List[gr.Gallery] = []
                for _dataset, _dataset_tag, label in _DATASETS:
                    eval_galleries.append(
                        gr.Gallery(
                            label=f"{label} · pose",
                            columns=4,
                            rows=1,
                            height="auto",
                            object_fit="contain",
                            elem_classes=["results-gallery", "lora-eval-gallery"],
                        )
                    )
                    eval_galleries.append(
                        gr.Gallery(
                            label=f"{label} · point",
                            columns=4,
                            rows=1,
                            height="auto",
                            object_fit="contain",
                            elem_classes=["results-gallery", "lora-eval-gallery"],
                        )
                    )
                    eval_galleries.append(
                        gr.Gallery(
                            label=f"{label} · depth",
                            columns=4,
                            rows=1,
                            height="auto",
                            object_fit="contain",
                            elem_classes=["results-gallery", "lora-eval-gallery"],
                        )
                    )
                training_gallery = gr.Gallery(
                    label="training",
                    columns=2,
                    rows=1,
                    height="auto",
                    object_fit="contain",
                    elem_classes=["results-gallery", "lora-training-gallery"],
                )

        gallery_outputs = [
            *eval_galleries,
            training_gallery,
            lora_info,
        ]
        lora_plan_dd.change(
            _plan_changed,
            inputs=[lora_plan_dd, lora_dataset_cb],
            outputs=[lora_model_dd, *gallery_outputs],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        lora_model_dd.change(
            _refresh,
            inputs=[lora_plan_dd, lora_model_dd, lora_dataset_cb],
            outputs=gallery_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        lora_dataset_cb.change(
            _refresh,
            inputs=[lora_plan_dd, lora_model_dd, lora_dataset_cb],
            outputs=gallery_outputs,
            show_progress="hidden",
            trigger_mode="always_last",
        )
        tab_lora.select(
            _refresh,
            inputs=[lora_plan_dd, lora_model_dd, lora_dataset_cb],
            outputs=gallery_outputs,
            show_progress="minimal",
        )
        blocks.load(
            _refresh,
            inputs=[lora_plan_dd, lora_model_dd, lora_dataset_cb],
            outputs=gallery_outputs,
            show_progress="minimal",
        )

    return {"tab": tab_lora}
