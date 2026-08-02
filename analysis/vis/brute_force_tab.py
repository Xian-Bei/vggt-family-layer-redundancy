from __future__ import annotations

from typing import Any, Dict, List, Sequence

import gradio as gr


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
        gr.Markdown(
            "Each backbone is one row. The four heatmaps average, from left to right: "
            "**indoor pose**, **indoor point cloud**, **outdoor pose**, and "
            "**outdoor point cloud**. Indoor combines ScanNet and 7Scenes; outdoor "
            "combines nuScenes and Waymo. Pose uses ARE, APE, RPE rotation, and RPE "
            "translation; point cloud uses accuracy, completeness, and Chamfer distance. "
            "Pose heatmaps use the viridis color scale and point-cloud heatmaps use coolwarm. "
            "The status box lists the datasets actually included and any incomplete results."
        )
        with gr.Row():
            with gr.Column(scale=1):
                overview_backbone_filter = gr.CheckboxGroup(
                    label="Backbones",
                    choices=backbones,
                    value=backbones,
                )
                overview_value_mode = gr.Radio(
                    label="Skip heatmap value",
                    choices=["absolute", "multiple"],
                    value="absolute",
                )
                overview_histogram = gr.Checkbox(
                    label="Show row/column mean plots",
                    value=False,
                )
                overview_normalize = gr.Checkbox(
                    label="Normalize metrics to [0,1] before averaging",
                    value=True,
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
                        gr.Markdown(f"### {backbone}")
                        gallery = gr.Gallery(
                            label="Indoor pose · Indoor point cloud · Outdoor pose · Outdoor point cloud",
                            columns=4,
                            height="auto",
                            object_fit="contain",
                            elem_classes=["results-gallery", "stage2-heatmap-gallery"],
                        )
                    overview_rows.append(row)
                    overview_galleries.append(gallery)

        overview_inputs = [
            overview_backbone_filter,
            overview_value_mode,
            overview_histogram,
            overview_normalize,
        ]
        overview_outputs = [*overview_galleries, *overview_rows, overview_info]
        _bind_refresh(
            tab=overview_tab,
            inputs=overview_inputs,
            outputs=overview_outputs,
            refresh_cb=refresh_overview_cb,
            triggers=[
                (overview_backbone_filter.change, False),
                (overview_value_mode.change, False),
                (overview_histogram.change, False),
                (overview_normalize.change, False),
                (overview_tab.select, True),
            ],
        )

    with gr.Tab("Brute-Force Detail") as detail_tab:
        gr.Markdown("## Brute-Force Detail")
        gr.Markdown(
            "Detailed mean-scene pruning results. Each skip scope uses two rows: "
            "the four pose metrics, followed by accuracy, completeness, Chamfer "
            "distance, and the mean of all seven metrics. Remaining-pruning "
            "visualization is available in the **Heatmaps** tab."
        )
        with gr.Row():
            with gr.Column(scale=1):
                backbone_filter = gr.CheckboxGroup(
                    label="Backbones",
                    choices=backbones,
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
                similarity_dataset_filter = gr.CheckboxGroup(
                    label="Similarity heatmaps",
                    choices=dataset_scopes,
                    value=["mean"],
                )
                value_mode = gr.Radio(
                    label="Skip heatmap value",
                    choices=["absolute", "multiple"],
                    value="absolute",
                )
                colorbar_histogram = gr.Checkbox(
                    label="Show row/column mean plots",
                    value=False,
                )
                normalize_skip_metrics = gr.Checkbox(
                    label="Normalize skip metrics to [0,1]",
                    value=True,
                )
                info = gr.Textbox(label="Displayed scopes", interactive=False)

            with gr.Column(scale=5, elem_classes=["stage2-right"]):
                galleries: List[gr.Gallery] = []
                row_cols: List[gr.Column] = []
                backbone_blocks: List[gr.Column] = []
                for backbone in backbones:
                    with gr.Column(elem_classes=["brute-force-backbone-block"]) as backbone_block:
                        gr.Markdown(f"### {backbone}")
                        for mode in stage2_modes:
                            for dataset in dataset_scopes:
                                for label in (
                                    f"{mode} · {dataset} · pose",
                                    f"{mode} · {dataset} · point cloud + 7-metric mean",
                                ):
                                    with gr.Row(
                                        elem_classes=["stage2-gallery-row"],
                                        equal_height=False,
                                    ):
                                        with gr.Column(scale=1, min_width=280) as row_col:
                                            gallery = gr.Gallery(
                                                label=label,
                                                columns=4,
                                                height="auto",
                                                object_fit="contain",
                                                elem_classes=[
                                                    "results-gallery",
                                                    "stage2-heatmap-gallery",
                                                    "brute-force-detail-gallery",
                                                ],
                                            )
                                    row_cols.append(row_col)
                                    galleries.append(gallery)
                        for dataset in dataset_scopes:
                            with gr.Row(
                                elem_classes=["stage2-gallery-row"],
                                equal_height=False,
                            ):
                                with gr.Column(scale=1, min_width=280) as similarity_row_col:
                                    similarity_gallery = gr.Gallery(
                                        label=f"Similarity · {dataset} · token mean",
                                        columns=4,
                                        rows=1,
                                        height="auto",
                                        object_fit="contain",
                                        elem_classes=[
                                            "results-gallery",
                                            "stage2-heatmap-gallery",
                                            "brute-force-detail-gallery",
                                            "brute-force-similarity-gallery",
                                        ],
                                    )
                            row_cols.append(similarity_row_col)
                            galleries.append(similarity_gallery)
                    backbone_blocks.append(backbone_block)

        inputs = [
            backbone_filter,
            *skip_dataset_checks,
            skip_mean_check,
            similarity_dataset_filter,
            value_mode,
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
                (similarity_dataset_filter.change, False),
                (value_mode.change, False),
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
