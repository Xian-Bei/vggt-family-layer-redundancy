from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr

from analysis.redundancy.similarity_v3 import (
    DEFAULT_AGGREGATION_EXCLUDE_TOKEN_REGIONS,
    REDUNDANCY_GRAPH_REGIONS,
)

_BACKBONE_ORDER: Tuple[str, ...] = (
    "vggt",
    "pi3",
    "vggt_omega",
)

_TOKEN_REGION_ORDER: Tuple[str, ...] = (
    "camera",
    "special",
    "all",
    "patch",
    "mean",
    "min",
)


def _token_region_sort_key(name: str) -> Tuple[int, str]:
    key = str(name or "").strip().lower()
    try:
        return (_TOKEN_REGION_ORDER.index(key), key)
    except ValueError:
        return (len(_TOKEN_REGION_ORDER), key)


_DATASET_ORDER: Tuple[str, ...] = (
    "scannet_dev50",
    "scans_dev50",
    "scannet_test50",
    "scans_test50",
    "7scenes_test",
    "7scenes",
    "7scenes_train",
    "nrgbd",
    "nuscenes_val",
    "waymo_validation",
    "nuscenes_train",
    "waymo_training",
)

_BACKBONE_DISPLAY_NAMES = {
    "vggt": "VGGT",
    "pi3": "Pi3",
    "vggt_omega": "VGGT-Ω",
}

_DATASET_DISPLAY_NAMES = {
    "scannet_dev50": "ScanNet",
    "scans_dev50": "ScanS",
    "scannet_test50": "ScanNet Test50",
    "scans_test50": "ScanS Test50",
    "7scenes_test": "7Scenes Test",
    "7scenes": "7Scenes",
    "7scenes_train": "7Scenes",
    "nrgbd": "NRGBD",
    "nuscenes_val": "nuScenes Val",
    "waymo_validation": "Waymo Validation",
    "nuscenes_train": "nuScenes",
    "waymo_training": "Waymo",
}


def _backbone_display_name(name: str) -> str:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key == "vggtomega":
        key = "vggt_omega"
    return _BACKBONE_DISPLAY_NAMES.get(key, str(name))


def _dataset_display_name(name: str) -> str:
    key = str(name or "").strip().lower()
    return _DATASET_DISPLAY_NAMES.get(key, str(name).replace("_", " ").title())


def _backbone_sort_key(name: str) -> Tuple[int, str]:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key == "vggtomega":
        key = "vggt_omega"
    try:
        return (_BACKBONE_ORDER.index(key), key)
    except ValueError:
        return (len(_BACKBONE_ORDER), key)


def _dataset_sort_key(name: str) -> Tuple[int, str]:
    key = str(name or "").strip()
    try:
        return (_DATASET_ORDER.index(key), key)
    except ValueError:
        return (len(_DATASET_ORDER), key)


def build_cka_heatmaps_tab(
    *,
    datasets: Sequence[str],
    models: Sequence[str],
    token_regions: Sequence[str],
    refresh_cka_heatmaps_cb: Any,
) -> Dict[str, Any]:
    source_dataset_names = [str(x) for x in datasets]
    source_model_names = [str(x) for x in models]
    dataset_names = sorted(source_dataset_names, key=_dataset_sort_key)
    model_names = sorted(source_model_names, key=_backbone_sort_key)
    token_region_names = sorted({str(x).strip().lower() for x in token_regions if str(x).strip()}, key=_token_region_sort_key)
    default_token_region = token_region_names[0] if token_region_names else "special"

    def _refresh_sorted(
        token_region: str,
        sim_cutoff_value: float,
        exclude_token_regions: List[str],
    ) -> Tuple[Any, ...]:
        raw = tuple(refresh_cka_heatmaps_cb(token_region, sim_cutoff_value, exclude_token_regions))
        cells_per_model = len(source_dataset_names) + 1
        total = len(source_model_names) * cells_per_model
        if len(raw) < total + 1:
            return raw
        images = list(raw[:total])
        info = raw[total:]
        source_index = {
            (model, dataset): model_i * cells_per_model + ds_i
            for model_i, model in enumerate(source_model_names)
            for ds_i, dataset in enumerate(source_dataset_names)
        }
        for model_i, model in enumerate(source_model_names):
            source_index[(model, "__mean__")] = model_i * cells_per_model + len(source_dataset_names)
        sorted_images: List[Any] = []
        for model in model_names:
            for dataset in dataset_names:
                key = (model, dataset)
                if key not in source_index:
                    continue
                sorted_images.append(images[source_index[key]])
            mean_key = (model, "__mean__")
            if mean_key in source_index:
                sorted_images.append(images[source_index[mean_key]])
        if len(sorted_images) != len(model_names) * cells_per_model:
            return raw
        return tuple([*sorted_images, *info])

    with gr.Tab("CKA Heatmaps") as tab_cka:
        gr.Markdown("## CKA Heatmaps")
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                cka_token_region = gr.Radio(
                    label="Token",
                    choices=token_region_names or ["special"],
                    value=default_token_region,
                    interactive=bool(token_region_names),
                )
                cka_sim_cutoff_slider = gr.Slider(
                    label="cutoff (0 disables)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01,
                    interactive=bool(dataset_names and model_names and token_region_names),
                )
                cka_exclude_token_regions = gr.CheckboxGroup(
                    label="Exclude from mean/min",
                    choices=list(REDUNDANCY_GRAPH_REGIONS),
                    value=list(DEFAULT_AGGREGATION_EXCLUDE_TOKEN_REGIONS),
                    interactive=bool(dataset_names and model_names and token_region_names),
                )
                cka_info = gr.Textbox(label="Displayed counts", interactive=False)
            with gr.Column(scale=5, elem_classes=["cka-heatmap-matrix"]):
                cka_images: List[gr.Image] = []
                if dataset_names and model_names:
                    with gr.Row(elem_classes=["cka-grid-header"]):
                        with gr.Column(scale=0, min_width=0, elem_classes=["cka-grid-corner"]):
                            pass
                        for dataset in dataset_names:
                            with gr.Column(scale=1, min_width=0, elem_classes=["cka-grid-dataset-label"]):
                                gr.Markdown(f"**{_dataset_display_name(dataset)}**")
                        with gr.Column(scale=1, min_width=0, elem_classes=["cka-grid-dataset-label"]):
                            gr.Markdown("**Mean**")
                    for model in model_names:
                        with gr.Row(elem_classes=["cka-grid-row"], equal_height=True):
                            with gr.Column(scale=0, min_width=0, elem_classes=["cka-grid-model-label"]):
                                gr.Markdown(f"**{_backbone_display_name(model)}**")
                            for dataset in dataset_names:
                                with gr.Column(scale=1, min_width=0, elem_classes=["cka-grid-cell"]):
                                    img = gr.Image(
                                        show_label=False,
                                        type="numpy",
                                        interactive=False,
                                        elem_classes=["results-media", "cka-heatmap-image"],
                                    )
                                cka_images.append(img)
                            with gr.Column(scale=1, min_width=0, elem_classes=["cka-grid-cell"]):
                                img = gr.Image(
                                    show_label=False,
                                    type="numpy",
                                    interactive=False,
                                    elem_classes=["results-media", "cka-heatmap-image"],
                                )
                            cka_images.append(img)
                else:
                    gr.Markdown("No v3 redundancy outputs found for CKA heatmaps.")

        cka_inputs = [cka_token_region, cka_sim_cutoff_slider, cka_exclude_token_regions]
        cka_outputs = [*cka_images, cka_info]

        if cka_images:
            cka_refresh_events: List[Any] = []

            def _bind_refresh(
                trigger: Any,
                *,
                with_trigger_mode: bool = True,
                use_cancels: bool = True,
                track_event: bool = True,
            ) -> None:
                kwargs: Dict[str, Any] = {
                    "inputs": cka_inputs,
                    "outputs": cka_outputs,
                    "show_progress": "minimal",
                }
                if with_trigger_mode:
                    kwargs["trigger_mode"] = "always_last"
                if use_cancels and cka_refresh_events:
                    kwargs["cancels"] = list(cka_refresh_events)
                event = trigger(_refresh_sorted, **kwargs)
                if track_event:
                    cka_refresh_events.append(event)

            _bind_refresh(cka_token_region.change)
            _bind_refresh(cka_sim_cutoff_slider.release)
            _bind_refresh(cka_exclude_token_regions.change)
            _bind_refresh(tab_cka.select, with_trigger_mode=False)
        else:
            cka_info.value = "No v3 redundancy outputs found for CKA heatmaps."

    return {
        "tab": tab_cka,
        "cka_token_region": cka_token_region,
        "cka_sim_cutoff_slider": cka_sim_cutoff_slider,
        "cka_exclude_token_regions": cka_exclude_token_regions,
        "cka_images": cka_images,
        "cka_info": cka_info,
        "cka_inputs": cka_inputs,
        "cka_outputs": cka_outputs,
    }
