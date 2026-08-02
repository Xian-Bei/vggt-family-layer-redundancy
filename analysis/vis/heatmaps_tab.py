from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr

HEATMAP_REAL_DATASETS: Tuple[str, ...] = (
    "nuscenes_train",
    "scannet_dev50",
)

_BACKBONE_ORDER: Tuple[str, ...] = (
    "vggt",
    "pi3",
    "vggt_omega",
)

_DATASET_ORDER: Tuple[str, ...] = (
    "min",
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
    stage2_metric_row_depth: str,
    stage2_metric_row_point: str,
    stage2_token_row_special: str,
    stage2_token_row_camera: str,
    stage2_token_row_patch: str,
    stage2_token_row_all: str,
    stage2_token_row_mean: str,
    stage2_token_row_min: str,
    heatmaps_resolve_filters_cb: Any,
    refresh_stage2_full_cb: Any,
) -> Dict[str, Any]:
    sorted_backbone_choices = sorted([str(x) for x in backbone_choices], key=_backbone_sort_key)
    allowed_datasets = {"min", "mean", *HEATMAP_REAL_DATASETS}
    sorted_dataset_choices = sorted(
        [str(x) for x in dataset_choices if str(x) in allowed_datasets],
        key=_dataset_sort_key,
    )

    with gr.Tab("Metrics & Remaining") as tab_stage2:
        gr.Markdown("## Metrics & Remaining")
        gr.Markdown(
            "Choose `mean` or a scene. Each skip mode shows one row containing the "
            "**7-metric mean** followed by the four similarity heatmaps for each "
            "selected token region and backbone. "
            "Dataset **`min`**: virtual cross-dataset view (element-wise min of each dataset's `mean` "
            "similarity; Scene locked to `mean`). "
            "Dataset **`mean`**: virtual cross-dataset view (element-wise mean of each dataset's `mean` "
            "similarity; Scene locked to `mean`). "
            "Skip absolute/multiple applies only to the 7-metric mean. "
            "**Show remaining** adds remaining-vs-full versions of only the "
            "7-metric mean and similarity maps, plus marginal 7-metric mean "
            "and similarity maps rebased by the first prune interval's "
            "`1 - similarity` degradation, clipped to `[0,1]`. "
            "Heatmap titles report strict-upper Pearson scores: **H r** compares "
            "each non-first column with the first column in the same row; "
            "**V r** compares each non-first row with the first row in the same column. "
            "Diagonal and NaN entries are excluded."
        )
        with gr.Row():
            with gr.Column(scale=1):
                stage2_backbone_filter = gr.CheckboxGroup(
                    label="Backbones",
                    choices=sorted_backbone_choices,
                    value=(
                        ["vggt"]
                        if "vggt" in sorted_backbone_choices
                        else sorted_backbone_choices[:1]
                    ),
                )
                stage2_dataset_dd = gr.Dropdown(
                    label="Dataset",
                    choices=sorted_dataset_choices,
                    value=controller.dataset,
                )
                stage2_scene_dd = gr.Dropdown(label="Scene", choices=stage2_scene_ids, value=initial_stage2_scene)
                _hm_vm, _hm_m, _hm_t, _, _hm_tc, _hm_hs, _hm_hi, _hm_po = heatmaps_resolve_filters_cb(
                    controller, initial_stage2_scene, "absolute", None, None
                )
                stage2_metric_rows_filter = gr.CheckboxGroup(
                    choices=[],
                    value=[],
                    visible=False,
                )
                stage2_value_mode = gr.Radio(
                    label="Skip heatmap value",
                    choices=["absolute", "multiple"],
                    value=_hm_vm,
                    elem_classes=["hm-value-mode"] + (["hm-value-mode-pct-locked"] if not _hm_po else []),
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
                stage2_normalize_skip_metrics = gr.Checkbox(
                    label="Normalize 7-metric mean to [0,1]",
                    value=True,
                )
                stage2_show_remaining = gr.Checkbox(
                    label="Show remaining",
                    value=False,
                    info=(
                        "For scene=mean, append remaining 7-metric mean and "
                        "similarity maps, then marginal mean and similarity."
                    ),
                )
                stage2_sim_gamma_slider = gr.State(1.0)
                stage2_sim_cutoff_slider = gr.State(0.0)
                stage2_info = gr.Textbox(label="Displayed counts", interactive=False)
            with gr.Column(scale=5, elem_classes=["stage2-right"]):
                stage2_mode_galleries: List[gr.Gallery] = []
                stage2_row_body_cols: List[gr.Column] = []
                for mode in stage2_modes:
                    for index, label in enumerate((
                        f"{mode}: hidden pose",
                        f"{mode}: hidden depth",
                        (
                            f"{mode}: 7-metric mean · selected similarity maps "
                            "· original / remaining / marginal"
                        ),
                        f"{mode}: hidden sim special",
                        f"{mode}: hidden sim camera",
                        f"{mode}: hidden sim patch",
                        f"{mode}: hidden sim all",
                        f"{mode}: hidden sim mean",
                        f"{mode}: hidden sim min",
                    )):
                        visible = index == 2
                        gallery_classes = ["results-gallery", "stage2-heatmap-gallery"]
                        with gr.Row(
                            elem_classes=["stage2-gallery-row"],
                            equal_height=False,
                            visible=visible,
                        ):
                            with gr.Column(scale=1, min_width=280) as row_body:
                                gal = gr.Gallery(
                                    label=label,
                                    columns=5 if visible else 4,
                                    height="auto",
                                    object_fit="contain",
                                    visible=visible,
                                    elem_classes=gallery_classes,
                                )
                            stage2_row_body_cols.append(row_body)
                            stage2_mode_galleries.append(gal)

        stage2_inputs = [
            stage2_backbone_filter,
            stage2_scene_dd,
            stage2_value_mode,
            stage2_pct_clip_slider,
            stage2_metric_rows_filter,
            stage2_token_rows_filter,
            stage2_sim_gamma_slider,
            stage2_sim_cutoff_slider,
            stage2_colorbar_histogram,
            stage2_normalize_skip_metrics,
            stage2_show_remaining,
        ]
        stage2_outputs = [
            *stage2_mode_galleries,
            *stage2_row_body_cols,
            stage2_info,
            stage2_value_mode,
            stage2_pct_clip_slider,
            stage2_metric_rows_filter,
            stage2_token_rows_filter,
            stage2_sim_gamma_slider,
            stage2_sim_cutoff_slider,
        ]

        if stage2_scene_ids:
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
                event = trigger(refresh_stage2_full_cb, **kwargs)
                if track_event:
                    stage2_refresh_events.append(event)

            _bind_refresh(stage2_scene_dd.change)
            _bind_refresh(stage2_value_mode.change)
            _bind_refresh(stage2_token_rows_filter.change)
            _bind_refresh(stage2_colorbar_histogram.change)
            _bind_refresh(stage2_normalize_skip_metrics.change)
            _bind_refresh(stage2_show_remaining.change)
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
        "stage2_mode_galleries": stage2_mode_galleries,
        "stage2_row_body_cols": stage2_row_body_cols,
        "stage2_info": stage2_info,
        "stage2_value_mode": stage2_value_mode,
        "stage2_pct_clip_slider": stage2_pct_clip_slider,
        "stage2_metric_rows_filter": stage2_metric_rows_filter,
        "stage2_token_rows_filter": stage2_token_rows_filter,
        "stage2_sim_gamma_slider": stage2_sim_gamma_slider,
        "stage2_sim_cutoff_slider": stage2_sim_cutoff_slider,
        "stage2_colorbar_histogram": stage2_colorbar_histogram,
        "stage2_normalize_skip_metrics": stage2_normalize_skip_metrics,
        "stage2_show_remaining": stage2_show_remaining,
        "stage2_inputs": stage2_inputs,
        "stage2_outputs": stage2_outputs,
    }
