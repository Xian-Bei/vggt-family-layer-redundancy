from __future__ import annotations

from typing import Any, Dict, List, Sequence

import gradio as gr


def build_heatmaps_tab(
    *,
    controller: Any,
    dataset_choices: Sequence[str],
    stage2_modes: List[str],
    initial_mode_values: List[str],
    stage2_scene_ids: List[str],
    initial_stage2_scene: str | None,
    stage2_metric_row_pose: str,
    stage2_metric_row_point: str,
    stage2_token_row_special: str,
    stage2_token_row_patch: str,
    stage2_token_row_all: str,
    heatmaps_resolve_filters_cb: Any,
    refresh_stage2_full_cb: Any,
) -> Dict[str, Any]:
    with gr.Tab("Heatmaps") as tab_stage2:
        gr.Markdown("## Heatmaps")
        gr.Markdown(
            "Choose `mean` or a scene. Per skip mode: pose -> special sim -> depth -> patch sim -> all-token sim. "
            "Use the **left filters** to show or hide fusion modes and row groups. "
            "Skip value/pct controls apply only to skip rows (mean uses full_report Δ then pct+clip vs mean Stage1)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                stage2_dataset_dd = gr.Dropdown(
                    label="Dataset",
                    choices=list(dataset_choices),
                    value=controller.dataset,
                    info="Switch Stage2 report roots to <sim/skip>/<dataset>/",
                )
                stage2_scene_dd = gr.Dropdown(label="Scene", choices=stage2_scene_ids, value=initial_stage2_scene)
                _hm_vm, _hm_m, _hm_t, _, _hm_hs, _hm_hi, _hm_po = heatmaps_resolve_filters_cb(
                    controller, initial_stage2_scene, "absolute", None, None
                )
                stage2_mode_filter = gr.CheckboxGroup(
                    label="Fusion / skip modes",
                    choices=list(stage2_modes),
                    value=list(initial_mode_values),
                )
                stage2_metric_rows_filter = gr.CheckboxGroup(
                    label="Skip Δ rows",
                    choices=[stage2_metric_row_pose, stage2_metric_row_point],
                    value=_hm_m,
                    interactive=_hm_hs,
                    info="pose: ARE/ATE/RPE. point: depth & Chamfer.",
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
                    choices=[stage2_token_row_special, stage2_token_row_patch, stage2_token_row_all],
                    value=_hm_t,
                    interactive=_hm_hi,
                    info="special-, patch-, and all-token heatmaps.",
                )
                stage2_sim_gamma_slider = gr.Slider(
                    label="CKA/cosine gamma",
                    minimum=0.1,
                    maximum=3.0,
                    value=1.0,
                    step=0.05,
                    interactive=_hm_hi,
                    info="Only affects CKA and cosine in token similarity rows; gamma=1 disables mapping.",
                )
                stage2_sim_cutoff_slider = gr.Slider(
                    label="CKA/cosine cutoff",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.01,
                    interactive=_hm_hi,
                    info="Only affects CKA and cosine in token similarity rows; 0 disables cutoff.",
                )
                stage2_info = gr.Textbox(label="Displayed counts", interactive=False)
            with gr.Column(scale=5, elem_classes=["stage2-right"]):
                stage2_mode_galleries: List[gr.Gallery] = []
                stage2_row_body_cols: List[gr.Column] = []
                for mode in stage2_modes:
                    for label, n_columns in (
                        (f"{mode}: pose", 4),
                        (f"{mode}: sim · special", 4),
                        (f"{mode}: depth", 3),
                        (f"{mode}: sim · patch", 4),
                        (f"{mode}: sim · all tokens", 4),
                    ):
                        with gr.Row(elem_classes=["stage2-gallery-row"], equal_height=False):
                            with gr.Column(scale=1, min_width=280) as row_body:
                                gal = gr.Gallery(
                                    label=label,
                                    columns=n_columns,
                                    height="auto",
                                    object_fit="contain",
                                    elem_classes=["results-gallery", "stage2-heatmap-gallery"],
                                )
                            stage2_row_body_cols.append(row_body)
                            stage2_mode_galleries.append(gal)

        stage2_inputs = [
            stage2_scene_dd,
            stage2_value_mode,
            stage2_pct_clip_slider,
            stage2_mode_filter,
            stage2_metric_rows_filter,
            stage2_token_rows_filter,
            stage2_sim_gamma_slider,
            stage2_sim_cutoff_slider,
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

            def _bind_refresh(trigger: Any, *, with_trigger_mode: bool = True) -> None:
                kwargs: Dict[str, Any] = {
                    "inputs": stage2_inputs,
                    "outputs": stage2_outputs,
                    "show_progress": "minimal",
                }
                if with_trigger_mode:
                    kwargs["trigger_mode"] = "always_last"
                if stage2_refresh_events:
                    kwargs["cancels"] = list(stage2_refresh_events)
                event = trigger(refresh_stage2_full_cb, **kwargs)
                stage2_refresh_events.append(event)

            _bind_refresh(stage2_scene_dd.change)
            _bind_refresh(stage2_value_mode.change)
            _bind_refresh(stage2_pct_clip_slider.release)
            _bind_refresh(stage2_mode_filter.change)
            _bind_refresh(stage2_metric_rows_filter.change)
            _bind_refresh(stage2_token_rows_filter.change)
            _bind_refresh(stage2_sim_gamma_slider.release)
            _bind_refresh(stage2_sim_cutoff_slider.release)
            _bind_refresh(tab_stage2.select, with_trigger_mode=False)
        else:
            stage2_info.value = (
                f"No Stage2/Stage3 reports found under "
                f"sim={controller.analysis_similarity_dir} and skip={controller.analysis_skip_dir}"
            )

    return {
        "tab": tab_stage2,
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
        "stage2_mode_filter": stage2_mode_filter,
        "stage2_inputs": stage2_inputs,
        "stage2_outputs": stage2_outputs,
    }
