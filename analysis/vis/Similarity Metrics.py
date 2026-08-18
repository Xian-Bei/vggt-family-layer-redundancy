from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr

from analysis.vis.display_names import (
    backbone_display_name,
    backbone_publication_sort_key,
)
from analysis.vis.composite import (
    Cell,
    Section,
    compose_sections,
    gallery_cells,
    update_value,
)

# Similarity Metrics typography and outer-label spacing (pixels).
SIMILARITY_SUBPLOT_FONT_SIZE = 22
SIMILARITY_OUTER_FONT_SIZE = 26
SIMILARITY_LEFT_LABEL_TO_IMAGE_GAP = 0
SIMILARITY_TOP_LABEL_TO_IMAGE_GAP = 0
SIMILARITY_BOTTOM_LABEL_TO_IMAGE_GAP = 0

HEATMAP_REAL_DATASETS: Tuple[str, ...] = (
    "nuscenes_train",
    "scannet_dev50",
)

SIMILARITY_COLUMN_LABELS: Tuple[str, ...] = (
    "Normalized Retention",
    "CKA",
    "Cosine",
    "CCA",
    "Inv_MSE",
)

_DATASET_ORDER: Tuple[str, ...] = (
    "mean",
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


def _compose_similarity_metrics(value: Any) -> Any:
    cells = gallery_cells(value)
    columns = len(SIMILARITY_COLUMN_LABELS)
    rows: List[List[Cell]] = []
    labels: List[str] = []
    for start in range(0, len(cells), columns):
        row = cells[start : start + columns]
        labels.append(next((cell.caption for cell in row if cell.caption), ""))
        for cell in row:
            cell.caption = ""
        row.extend(Cell(None) for _ in range(columns - len(row)))
        rows.append(row)
    return compose_sections(
        [
            Section(
                columns=columns,
                cells=[],
                fixed_rows=rows,
                row_labels=labels,
                row_labels_bold=True,
                column_labels=list(SIMILARITY_COLUMN_LABELS),
                column_labels_bold=True,
            )
        ],
        outer_font_size=SIMILARITY_OUTER_FONT_SIZE,
        left_label_to_image_gap=SIMILARITY_LEFT_LABEL_TO_IMAGE_GAP,
        top_label_to_image_gap=SIMILARITY_TOP_LABEL_TO_IMAGE_GAP,
        bottom_label_to_image_gap=SIMILARITY_BOTTOM_LABEL_TO_IMAGE_GAP,
    )


def _dataset_sort_key(name: str) -> Tuple[int, str]:
    key = str(name or "").strip()
    try:
        return (_DATASET_ORDER.index(key), key)
    except ValueError:
        return (len(_DATASET_ORDER), key)


def build_heatmaps_tab(
    *,
    controller: Any,
    dataset_choices: Sequence[str],
    backbone_choices: Sequence[str],
    initial_backbone: str | None,
    stage2_modes: List[str],
    stage2_scene_ids: List[str],
    initial_stage2_scene: str | None,
    stage2_metric_row_pose: str,
    stage2_metric_row_point: str,
    stage2_token_row_special: str,
    stage2_token_row_camera: str,
    stage2_token_row_patch: str,
    stage2_token_row_all: str,
    stage2_token_row_mean: str,
    heatmaps_resolve_filters_cb: Any,
    refresh_stage2_full_cb: Any,
) -> Dict[str, Any]:
    sorted_backbone_choices = sorted(
        [str(x) for x in backbone_choices],
        key=backbone_publication_sort_key,
    )
    allowed_datasets = {"mean", *HEATMAP_REAL_DATASETS}
    sorted_dataset_choices = sorted(
        [str(x) for x in dataset_choices if str(x) in allowed_datasets],
        key=_dataset_sort_key,
    )
    default_dataset = (
        "mean"
        if "mean" in sorted_dataset_choices
        else (sorted_dataset_choices[0] if sorted_dataset_choices else None)
    )
    default_scene_ids = (
        ["mean"] if default_dataset == "mean" else list(stage2_scene_ids)
    )
    default_scene = (
        "mean" if default_dataset == "mean" else initial_stage2_scene
    )

    with gr.Tab("Similarity Metrics") as tab_stage2:
        gr.Markdown("## Similarity Metrics")
        gr.Markdown(
            "The all-metric mean always normalizes each metric to `[0,1]` before averaging. "
            "similarity; Scene locked to `mean`). "
            "Dataset **`mean`**: virtual cross-dataset view (element-wise mean of each dataset's `mean` "
            "similarity; Scene locked to `mean`). "
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                stage2_backbone_filter = gr.CheckboxGroup(
                    label="Backbones",
                    choices=[
                        (backbone_display_name(item), item)
                        for item in sorted_backbone_choices
                    ],
                    value=sorted_backbone_choices,
                )
                stage2_dataset_dd = gr.Dropdown(
                    label="Dataset",
                    choices=sorted_dataset_choices,
                    value=default_dataset,
                )
                stage2_scene_dd = gr.Dropdown(
                    label="Scene",
                    choices=default_scene_ids,
                    value=default_scene,
                    interactive=default_dataset != "mean",
                )
                _hm_vm, _hm_m, _hm_t, _, _hm_tc, _hm_hs, _hm_hi, _hm_po = heatmaps_resolve_filters_cb(
                    controller, default_scene, "absolute", None, None
                )
                _hm_tc = [item for item in _hm_tc if str(item) != "min"]
                _hm_t = [item for item in _hm_t if str(item) != "min"]
                stage2_metric_rows_filter = gr.CheckboxGroup(
                    choices=[],
                    value=[],
                    visible=False,
                )
                stage2_value_mode = gr.Dropdown(
                    choices=["absolute"],
                    value="absolute",
                    visible=False,
                )
                stage2_pct_clip_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=100,
                    visible=False,
                )
                stage2_token_rows_filter = gr.CheckboxGroup(
                    label="Token similarity regions",
                    choices=_hm_tc,
                    value=_hm_t,
                    interactive=_hm_hi,
                )
                stage2_colorbar_histogram = gr.Checkbox(
                    label="Show row/column mean plots",
                    value=False,
                )
                stage2_sim_gamma_slider = gr.State(1.0)
                stage2_sim_cutoff_slider = gr.State(0.0)
                stage2_info = gr.Textbox(label="Displayed counts", interactive=False)
            with gr.Column(scale=5, elem_classes=["stage2-right"]):
                stage2_image = gr.Image(
                    label="mixed_units: 7-metric mean · selected similarity maps",
                    show_label=False,
                    type="numpy",
                    interactive=False,
                    elem_classes=["results-media", "composite-plot-image"],
                )

        stage2_inputs = [
            stage2_dataset_dd,
            stage2_backbone_filter,
            stage2_scene_dd,
            stage2_value_mode,
            stage2_pct_clip_slider,
            stage2_metric_rows_filter,
            stage2_token_rows_filter,
            stage2_sim_gamma_slider,
            stage2_sim_cutoff_slider,
            stage2_colorbar_histogram,
        ]
        raw_gallery_count = 8 * len(stage2_modes)

        def _compose_raw_result(raw: Sequence[Any]) -> Tuple[Any, ...]:
            combined: List[Any] = []
            for index in range(raw_gallery_count):
                if index % 8 != 2:
                    continue
                value, _ = update_value(raw[index])
                if isinstance(value, list):
                    combined.extend(value)
            return (
                _compose_similarity_metrics(combined),
                raw[raw_gallery_count * 2],
                *raw[raw_gallery_count * 2 + 1 :],
            )

        def _compose_refresh(*args: Any) -> Tuple[Any, ...]:
            return _compose_raw_result(refresh_stage2_full_cb(*args))

        stage2_outputs = [
            stage2_image,
            stage2_info,
            stage2_value_mode,
            stage2_pct_clip_slider,
            stage2_metric_rows_filter,
            stage2_token_rows_filter,
            stage2_sim_gamma_slider,
            stage2_sim_cutoff_slider,
        ]

        if default_scene_ids:
            stage2_refresh_events: List[Any] = []

            def _bind_refresh(
                trigger: Any,
                *,
                with_trigger_mode: bool = True,
                use_cancels: bool = True,
                track_event: bool = True,
            ) -> None:
                kwargs: Dict[str, Any] = {
                    "inputs": stage2_inputs,
                    "outputs": stage2_outputs,
                    "show_progress": "minimal",
                }
                if with_trigger_mode:
                    kwargs["trigger_mode"] = "always_last"
                if use_cancels and stage2_refresh_events:
                    kwargs["cancels"] = list(stage2_refresh_events)
                event = trigger(_compose_refresh, **kwargs)
                if track_event:
                    stage2_refresh_events.append(event)

            _bind_refresh(stage2_scene_dd.change)
            _bind_refresh(stage2_token_rows_filter.change)
            _bind_refresh(stage2_colorbar_histogram.change)
            _bind_refresh(tab_stage2.select, with_trigger_mode=False)
        else:
            stage2_info.value = (
                f"No Stage2/Stage3 reports found under "
                f"sim={controller.analysis_similarity_dir} and skip={controller.analysis_skip_dir}"
            )

    return {
        "tab": tab_stage2,
        "stage2_backbone_filter": stage2_backbone_filter,
        "stage2_dataset_dd": stage2_dataset_dd,
        "stage2_scene_dd": stage2_scene_dd,
        "stage2_image": stage2_image,
        "compose_refresh_result": _compose_refresh,
        "compose_raw_result": _compose_raw_result,
        "stage2_info": stage2_info,
        "stage2_value_mode": stage2_value_mode,
        "stage2_pct_clip_slider": stage2_pct_clip_slider,
        "stage2_metric_rows_filter": stage2_metric_rows_filter,
        "stage2_token_rows_filter": stage2_token_rows_filter,
        "stage2_sim_gamma_slider": stage2_sim_gamma_slider,
        "stage2_sim_cutoff_slider": stage2_sim_cutoff_slider,
        "stage2_colorbar_histogram": stage2_colorbar_histogram,
        "stage2_inputs": stage2_inputs,
        "stage2_outputs": stage2_outputs,
    }
