from __future__ import annotations

from typing import Any, Dict, List, Sequence

import gradio as gr


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
    stage2_token_row_patch: str,
    stage2_token_row_all: str,
    heatmaps_resolve_filters_cb: Any,
    refresh_stage2_full_cb: Any,
) -> Dict[str, Any]:
    with gr.Tab("Heatmaps") as tab_stage2:
        gr.Markdown("## Heatmaps")
        gr.Markdown(
            "Choose `mean` or a scene. Per skip mode: pose skip Δ -> depth Δ -> **all-token** similarity "
            "(v3 redundancy; special/patch token rows are disabled). "
            "Dataset **`min`**: virtual cross-dataset view (element-wise min of each dataset's `mean` "
            "all-token similarity; Scene locked to `mean`). "
            "A second row repeats the same metrics but drops 1-based odd layer rows and reindexes "
            "the remaining layers contiguously (e.g. 0–47 → 0–23). "
            "Use the **left filters** for skip row groups. "
            "Skip value/pct applies only to skip rows."
        )
        with gr.Row():
            with gr.Column(scale=1):
                stage2_backbone_dd = None
                if backbone_choices:
                    stage2_backbone_dd = gr.Dropdown(
                        label="Backbone",
                        choices=list(backbone_choices),
                        value=initial_backbone if initial_backbone in backbone_choices else backbone_choices[0],
                    )
                stage2_dataset_dd = gr.Dropdown(
                    label="Dataset",
                    choices=list(dataset_choices),
                    value=controller.dataset,
                )
                stage2_scene_dd = gr.Dropdown(label="Scene", choices=stage2_scene_ids, value=initial_stage2_scene)
                _hm_vm, _hm_m, _hm_t, _, _hm_hs, _hm_hi, _hm_po = heatmaps_resolve_filters_cb(
                    controller, initial_stage2_scene, "absolute", None, None
                )
                stage2_metric_rows_filter = gr.CheckboxGroup(
                    label="Skip Δ rows",
                    choices=[stage2_metric_row_pose, stage2_metric_row_point],
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
                    choices=[stage2_token_row_all],
                    value=[stage2_token_row_all] if _hm_hi else [],
                    interactive=False,
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
                    interactive=_hm_hi,
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
                    for label, n_columns, odd_row in (
                        (f"{mode}: pose", 4, False),
                        (f"{mode}: sim · special", 4, False),
                        (f"{mode}: depth", 3, False),
                        (f"{mode}: sim · patch", 4, False),
                        (f"{mode}: sim · all tokens", 4, False),
                        (f"{mode}: sim · all tokens (drop odd rows, reindexed)", 4, True),
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
