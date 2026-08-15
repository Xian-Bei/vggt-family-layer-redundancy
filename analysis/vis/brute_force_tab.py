from __future__ import annotations

from typing import Any, Dict, List, Sequence

import gradio as gr

from analysis.vis.display_names import backbone_display_name, dataset_display_name


def _bind_refresh(
    *,
    tab: Any,
    inputs: List[Any],
    outputs: List[Any],
    refresh_cb: Any,
    triggers: Sequence[tuple[Any, bool]],
) -> None:
    events: List[Any] = []
    for trigger, release in triggers:
        kwargs: Dict[str, Any] = {
            "inputs": inputs,
            "outputs": outputs,
            "show_progress": "minimal",
            "cancels": list(events),
        }
        if not release:
            kwargs["trigger_mode"] = "always_last"
        events.append(trigger(refresh_cb, **kwargs))


def build_brute_force_tab(
    *,
    backbone_choices: Sequence[str],
    dataset_choices: Sequence[str],
    initial_skip_datasets: Sequence[str],
    stage2_modes: List[str],
    refresh_overview_cb: Any,
    refresh_detail_cb: Any,
) -> Dict[str, Any]:
    """Build overview and detailed brute-force pruning tabs."""
    backbones = list(dict.fromkeys(str(x) for x in backbone_choices))
    datasets = list(dict.fromkeys(str(x) for x in dataset_choices))
    dataset_scopes = [*datasets, "mean"]
    available_skip_datasets = {
        str(dataset) for dataset in initial_skip_datasets
    }
    with gr.Tab("Brute-Force Overview") as overview_tab:
        gr.Markdown("## Brute-Force Overview")
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                overview_backbone_filter = gr.CheckboxGroup(
                    label="Backbones",
                    choices=[(backbone_display_name(item), item) for item in backbones],
                    value=backbones,
                )
                overview_histogram = gr.Checkbox(
                    label="Show row/column mean plots",
                    value=False,
                )
                overview_info = gr.Textbox(
                    label="Displayed scopes",
                    lines=10,
                    interactive=False,
                )
            with gr.Column(scale=5, elem_classes=["stage2-right"]):
                overview_galleries: List[gr.Gallery] = []
                overview_rows: List[gr.Column] = []
                for backbone in backbones:
                    with gr.Column(elem_classes=["brute-force-backbone-block"]) as row:
                        gr.Markdown(
                            f"### {backbone_display_name(backbone)}",
                            elem_classes=["backbone-separator"],
                        )
                        gallery = gr.Gallery(
                            label="Brute-force overview heatmaps",
                            show_label=False,
                            columns=4,
                            height="auto",
                            object_fit="contain",
                            elem_classes=[
                                "results-gallery",
                                "stage2-heatmap-gallery",
                                "brute-force-overview-gallery",
                            ],
                        )
                    overview_rows.append(row)
                    overview_galleries.append(gallery)

        overview_inputs = [
            overview_backbone_filter,
            overview_histogram,
        ]
        overview_outputs = [*overview_galleries, *overview_rows, overview_info]
        _bind_refresh(
            tab=overview_tab,
            inputs=overview_inputs,
            outputs=overview_outputs,
            refresh_cb=refresh_overview_cb,
            triggers=[
                (overview_backbone_filter.change, False),
                (overview_histogram.change, False),
                (overview_tab.select, True),
            ],
        )

    with gr.Tab("Brute-Force Detail") as detail_tab:
        gr.Markdown("## Brute-Force Detail")
        gr.Markdown(
            "Detailed mean-scene pruning results. Each skip scope uses two rows: "
            "the four pose metrics, followed by accuracy, completeness, Chamfer "
            "distance, and the mean of all seven metrics."
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                backbone_filter = gr.CheckboxGroup(
                    label="Backbones",
                    choices=[(backbone_display_name(item), item) for item in backbones],
                    value=backbones,
                )
                gr.Markdown("#### Skip results")
                skip_dataset_checks: List[gr.Checkbox] = []
                for dataset in datasets:
                    available = dataset in available_skip_datasets
                    skip_dataset_checks.append(
                        gr.Checkbox(
                            label=dataset,
                            value=available,
                            interactive=available,
                            info=(
                                None
                                if available
                                else "No experiment record; excluded from mean."
                            ),
                            elem_classes=(
                                []
                                if available
                                else ["brute-force-unavailable-dataset"]
                            ),
                        )
                    )
                skip_mean_check = gr.Checkbox(
                    label="mean",
                    value=False,
                    info="Mean includes only datasets with experiment records.",
                )
                colorbar_histogram = gr.Checkbox(
                    label="Show row/column mean plots",
                    value=False,
                )
                normalize_skip_metrics = gr.Checkbox(
                    label="Normalize individual skip metrics to [0,1]",
                    value=False,
                    info="The 7-metric mean is always normalized per metric before averaging.",
                )
                info = gr.Textbox(label="Displayed scopes", interactive=False)

            with gr.Column(scale=5, elem_classes=["stage2-right"]):
                galleries: List[gr.Gallery] = []
                row_cols: List[gr.Column] = []
                backbone_blocks: List[gr.Column] = []
                for backbone in backbones:
                    with gr.Column(elem_classes=["brute-force-backbone-block"]) as backbone_block:
                        gr.Markdown(
                            f"### {backbone_display_name(backbone)}",
                            elem_classes=["backbone-separator"],
                        )
                        for mode in stage2_modes:
                            for dataset in dataset_scopes:
                                for row_index, label in enumerate((
                                    f"{mode} · {dataset} · pose heatmaps",
                                    f"{mode} · {dataset} · point-cloud heatmaps",
                                )):
                                    with gr.Row(
                                        elem_classes=["stage2-gallery-row"],
                                        equal_height=False,
                                    ):
                                        with gr.Column(scale=1, min_width=280) as row_col:
                                            if row_index == 0:
                                                gr.Markdown(
                                                    f"#### {dataset_display_name(dataset)}",
                                                    elem_classes=[
                                                        "brute-force-dataset-heading"
                                                    ],
                                                )
                                            gallery = gr.Gallery(
                                                label=label,
                                                show_label=False,
                                                columns=4,
                                                height="auto",
                                                object_fit="contain",
                                                elem_classes=[
                                                    "results-gallery",
                                                    "stage2-heatmap-gallery",
                                                    "brute-force-detail-gallery",
                                                    *(
                                                        ["brute-force-dataset-start"]
                                                        if row_index == 0
                                                        else []
                                                    ),
                                                ],
                                            )
                                    row_cols.append(row_col)
                                    galleries.append(gallery)
                    backbone_blocks.append(backbone_block)

        inputs = [
            backbone_filter,
            *skip_dataset_checks,
            skip_mean_check,
            colorbar_histogram,
            normalize_skip_metrics,
        ]
        outputs = [*galleries, *row_cols, *backbone_blocks, info]

        def _refresh_detail_from_checkboxes(
            selected_backbones: List[str] | None,
            *values: Any,
        ) -> Any:
            scope_count = len(dataset_scopes)
            scope_flags = values[:scope_count]
            detail_inputs = values[scope_count:]
            selected_skip_datasets = [
                dataset
                for dataset, selected in zip(dataset_scopes, scope_flags)
                if bool(selected)
            ]
            return refresh_detail_cb(
                selected_backbones,
                selected_skip_datasets,
                *detail_inputs,
            )

        _bind_refresh(
            tab=detail_tab,
            inputs=inputs,
            outputs=outputs,
            refresh_cb=_refresh_detail_from_checkboxes,
            triggers=[
                (backbone_filter.change, False),
                *[
                    (checkbox.change, False)
                    for checkbox in [*skip_dataset_checks, skip_mean_check]
                ],
                (colorbar_histogram.change, False),
                (normalize_skip_metrics.change, False),
                (detail_tab.select, True),
            ],
        )

    return {
        "overview_tab": overview_tab,
        "detail_tab": detail_tab,
        "overview_galleries": overview_galleries,
        "detail_galleries": galleries,
    }
