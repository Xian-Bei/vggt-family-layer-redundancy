from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr

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
    sorted_dataset_choices = sorted([str(x) for x in dataset_choices], key=_dataset_sort_key)

    with gr.Tab("Heatmaps") as tab_stage2:
        gr.Markdown("## Heatmaps")
        gr.Markdown(
            "Choose `mean` or a scene. Per skip mode: pose/depth/point-cloud degradation plus the token-similarity "
            "regions actually present in the saved redundancy graphs. "
            "Dataset **`min`**: virtual cross-dataset view (element-wise min of each dataset's `mean` "
            "all-token similarity; Scene locked to `mean`). "
            "Dataset **`mean`**: virtual cross-dataset view (element-wise mean of each dataset's `mean` "
            "all-token similarity; Scene locked to `mean`). "
            "Use the **left filters** for skip row groups. "
            "Skip value/pct applies only to skip rows; darker means worse degradation."
        )
        with gr.Row():
            with gr.Column(scale=1):
                stage2_backbone_dd = None
                if sorted_backbone_choices:
                    stage2_backbone_dd = gr.Dropdown(
                        label="Backbone",
                        choices=sorted_backbone_choices,
                        value=(
                            initial_backbone
                            if initial_backbone in sorted_backbone_choices
                            else sorted_backbone_choices[0]
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
                    label="Skip Δ rows",
                    choices=[
                        stage2_metric_row_pose,
                        stage2_metric_row_depth,
                        stage2_metric_row_point,
                    ],
                    value=_hm_m,
                    interactive=_hm_hs,
                )
                stage2_value_mode = gr.Radio(
                    label="Skip heatmap value",
                    choices=["absolute", "pct"],
                    value=_hm_vm,
                    elem_classes=["hm-value-mode"] + (["hm-value-mode-pct-locked"] if not _hm_po else []),
                )
                stage2_pct_clip_slider = gr.Slider(
                    label="Pct clip threshold (%)",
                    minimum=0,
                    maximum=100,
                    value=100,
                    step=1,
                    interactive=_hm_po and _hm_vm == "pct",
                )
                stage2_token_rows_filter = gr.CheckboxGroup(
                    label="Token similarity rows",
                    choices=_hm_tc,
                    value=_hm_t,
                    interactive=_hm_hi,
                )
                stage2_sim_gamma_slider = gr.Slider(
                    label="gamma (1 disables)",
                    minimum=0.1,
                    maximum=3.0,
                    value=1.0,
                    step=0.05,
                    interactive=_hm_hi,
                )
                stage2_sim_cutoff_slider = gr.Slider(
                    label="cutoff (0 disables)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01,
                    interactive=_hm_hi or _hm_hs,
                )
                gr.Markdown("### Prune plan (dense triangles)")
                stage2_prune_show = gr.Checkbox(
                    label="Show dense-triangle markers",
                    value=False,
                    interactive=_hm_hi,
                )
                stage2_prune_metric = gr.Radio(
                    label="Prune metric",
                    choices=["cka", "cosine"],
                    value="cka",
                    interactive=_hm_hi,
                )
                stage2_prune_n = gr.Number(
                    label="n (triangles)",
                    value=2,
                    precision=0,
                    minimum=0,
                    maximum=20,
                    interactive=_hm_hi,
                )
                with gr.Row():
                    stage2_prune_save_btn = gr.Button(
                        value="Save prune plan",
                        variant="primary",
                        interactive=_hm_hi,
                    )
                stage2_prune_status = gr.Textbox(
                    label="Prune status",
                    interactive=False,
                    value="",
                    lines=4,
                )
                stage2_info = gr.Textbox(label="Displayed counts", interactive=False)
            with gr.Column(scale=5, elem_classes=["stage2-right"]):
                stage2_mode_galleries: List[gr.Gallery] = []
                stage2_row_body_cols: List[gr.Column] = []
                for mode in stage2_modes:
                    for label, n_columns in (
                        (f"{mode}: pose", 4),
                        (f"{mode}: depth", 2),
                        (f"{mode}: point cloud", 4),
                        (f"{mode}: sim · special", 4),
                        (f"{mode}: sim · camera", 4),
                        (f"{mode}: sim · patch", 4),
                        (f"{mode}: sim · all tokens", 4),
                        (f"{mode}: sim · token mean", 4),
                        (f"{mode}: sim · token min", 4),
                    ):
                        gallery_classes = ["results-gallery", "stage2-heatmap-gallery"]
                        with gr.Row(elem_classes=["stage2-gallery-row"], equal_height=False):
                            with gr.Column(scale=1, min_width=280) as row_body:
                                gal = gr.Gallery(
                                    label=label,
                                    columns=n_columns,
                                    height="auto",
                                    object_fit="contain",
                                    elem_classes=gallery_classes,
                                )
                            stage2_row_body_cols.append(row_body)
                            stage2_mode_galleries.append(gal)

        stage2_inputs = [
            stage2_scene_dd,
            stage2_value_mode,
            stage2_pct_clip_slider,
            stage2_metric_rows_filter,
            stage2_token_rows_filter,
            stage2_sim_gamma_slider,
            stage2_sim_cutoff_slider,
            stage2_prune_show,
            stage2_prune_metric,
            stage2_prune_n,
        ]
        stage2_outputs = [
            *stage2_mode_galleries,
            *stage2_row_body_cols,
            stage2_prune_status,
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
            _bind_refresh(stage2_pct_clip_slider.release)
            _bind_refresh(stage2_metric_rows_filter.change)
            _bind_refresh(stage2_token_rows_filter.change)
            _bind_refresh(stage2_sim_gamma_slider.release)
            _bind_refresh(stage2_sim_cutoff_slider.release)
            _bind_refresh(stage2_prune_show.change)
            _bind_refresh(stage2_prune_metric.change)
            # Number has no stable .input; binding it + cancels aborts SSE and spams 404.
            _bind_refresh(stage2_prune_n.change, use_cancels=False, track_event=False)
            _bind_refresh(tab_stage2.select, with_trigger_mode=False)
        else:
            stage2_info.value = (
                f"No Stage2/Stage3 reports found under "
                f"sim={controller.analysis_similarity_dir} and skip={controller.analysis_skip_dir}"
            )

    return {
        "tab": tab_stage2,
        "stage2_backbone_dd": stage2_backbone_dd,
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
        "stage2_prune_show": stage2_prune_show,
        "stage2_prune_metric": stage2_prune_metric,
        "stage2_prune_n": stage2_prune_n,
        "stage2_prune_save_btn": stage2_prune_save_btn,
        "stage2_prune_status": stage2_prune_status,
        "stage2_inputs": stage2_inputs,
        "stage2_outputs": stage2_outputs,
    }
