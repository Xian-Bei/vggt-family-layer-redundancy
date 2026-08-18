from __future__ import annotations

from typing import Any, Dict, List, Sequence

import gradio as gr

from analysis.metric_labels import SEVEN_METRIC_KEYS, SEVEN_METRIC_LABELS
from analysis.vis.display_names import backbone_display_name, dataset_display_name
from analysis.vis.composite import (
    Cell,
    Section,
    compose_sections,
    gallery_cells,
)

# Brute-Force Overview typography and outer-label spacing (pixels).
BRUTE_FORCE_OVERVIEW_SUBPLOT_FONT_SIZE = 20
BRUTE_FORCE_OVERVIEW_OUTER_FONT_SIZE = 26
BRUTE_FORCE_OVERVIEW_LEFT_LABEL_TO_IMAGE_GAP = 0
BRUTE_FORCE_OVERVIEW_TOP_LABEL_TO_IMAGE_GAP = 0
BRUTE_FORCE_OVERVIEW_BOTTOM_LABEL_TO_IMAGE_GAP = 0

# Brute-Force Detail typography and outer-label spacing (pixels).
BRUTE_FORCE_DETAIL_SUBPLOT_FONT_SIZE = 26
BRUTE_FORCE_DETAIL_OUTER_FONT_SIZE = 30
BRUTE_FORCE_DETAIL_LEFT_LABEL_TO_IMAGE_GAP = 0
BRUTE_FORCE_DETAIL_TOP_LABEL_TO_IMAGE_GAP = 0
BRUTE_FORCE_DETAIL_BOTTOM_LABEL_TO_IMAGE_GAP = 0


BRUTE_FORCE_OVERVIEW_COLUMN_LABELS = (
    "Indoor Pose Metrics \n Normalized Retention",
    "Indoor Geometry Metrics \n Normalized Retention",
    "Outdoor Pose Metrics \n Normalized Retention",
    "Outdoor Geometry Metrics \n Normalized Retention",
)

BRUTE_FORCE_DETAIL_COLUMN_LABELS = tuple(
    SEVEN_METRIC_LABELS[key] for key in SEVEN_METRIC_KEYS
)

BRUTE_FORCE_DETAIL_EXCLUDED_DATASETS = {
    "7scenes_test",
    "nuscenes_test",
    "waymo_validation",
    "mean",
}
BRUTE_FORCE_DETAIL_DATASET_ORDER = (
    "scannet_dev50",
    "nuscenes_train",
)

BRUTE_FORCE_OVERVIEW_INFO = """\
### Mean metric sources

pose mean: mean(ARE, APE, RPE_rot, RPE_trans)<br>
point-cloud mean: mean(Accuracy, Completeness, Chamfer)

### Averaged datasets

**vggt_omega**<br>
&nbsp;&nbsp;indoor: scannet_dev50<br>
&nbsp;&nbsp;outdoor: nuscenes_train

**pi3**<br>
&nbsp;&nbsp;indoor: scannet_dev50<br>
&nbsp;&nbsp;outdoor: nuscenes_train

**vggt**<br>
&nbsp;&nbsp;indoor: scannet_dev50<br>
&nbsp;&nbsp;outdoor: nuscenes_train
"""


def _compose_brute_force_overview(
    gallery_values: Sequence[Any],
    backbone_labels: Sequence[str],
) -> Any:
    columns = len(BRUTE_FORCE_OVERVIEW_COLUMN_LABELS)
    rows = [gallery_cells(value) for value in gallery_values]
    fixed_rows = [
        row[:columns] + [Cell(None)] * max(0, columns - len(row))
        for row in rows
    ]
    return compose_sections(
        [
            Section(
                columns=columns,
                cells=[],
                fixed_rows=fixed_rows,
                row_labels=list(backbone_labels),
                row_labels_bold=True,
                column_labels=list(BRUTE_FORCE_OVERVIEW_COLUMN_LABELS),
                column_labels_bold=True,
            )
        ],
        outer_font_size=BRUTE_FORCE_OVERVIEW_OUTER_FONT_SIZE,
        left_label_to_image_gap=BRUTE_FORCE_OVERVIEW_LEFT_LABEL_TO_IMAGE_GAP,
        top_label_to_image_gap=BRUTE_FORCE_OVERVIEW_TOP_LABEL_TO_IMAGE_GAP,
        bottom_label_to_image_gap=BRUTE_FORCE_OVERVIEW_BOTTOM_LABEL_TO_IMAGE_GAP,
    )


def _compose_brute_force_detail(
    gallery_values: Sequence[Any],
    backbones: Sequence[str],
    modes: Sequence[str],
    datasets: Sequence[str],
) -> Any | None:
    """Compose one seven-metric row per backbone and dataset."""
    value_index = 0
    sections: List[Section] = []
    for backbone in backbones:
        for _mode in modes:
            for dataset in datasets:
                pose_cells = gallery_cells(gallery_values[value_index])
                value_index += 1
                point_cells = gallery_cells(gallery_values[value_index])
                value_index += 1
                cells = [*pose_cells[:4], *point_cells[:3]]
                if not cells:
                    continue
                cells.extend(Cell(None) for _ in range(7 - len(cells)))
                sections.append(
                    Section(
                        columns=7,
                        cells=cells,
                        heading=(
                            f"{backbone_display_name(backbone).replace('Ω', 'Omega')} "
                            f"{dataset_display_name(dataset)}"
                        ),
                        column_labels_bold=True,
                    )
                )
    if not sections:
        return None
    sections[-1].column_labels = list(BRUTE_FORCE_DETAIL_COLUMN_LABELS)
    sections[-1].column_labels_below = True
    return compose_sections(
        sections,
        outer_font_size=BRUTE_FORCE_DETAIL_OUTER_FONT_SIZE,
        left_label_to_image_gap=BRUTE_FORCE_DETAIL_LEFT_LABEL_TO_IMAGE_GAP,
        top_label_to_image_gap=BRUTE_FORCE_DETAIL_TOP_LABEL_TO_IMAGE_GAP,
        bottom_label_to_image_gap=BRUTE_FORCE_DETAIL_BOTTOM_LABEL_TO_IMAGE_GAP,
    )


def filter_brute_force_detail_datasets(
    dataset_choices: Sequence[str],
) -> List[str]:
    """Return the dataset scopes represented by Detail UI/output rows."""
    datasets = [
        str(dataset)
        for dataset in dataset_choices
        if str(dataset).strip().lower()
        not in BRUTE_FORCE_DETAIL_EXCLUDED_DATASETS
    ]
    return sorted(
        datasets,
        key=lambda dataset: (
            BRUTE_FORCE_DETAIL_DATASET_ORDER.index(dataset)
            if dataset in BRUTE_FORCE_DETAIL_DATASET_ORDER
            else len(BRUTE_FORCE_DETAIL_DATASET_ORDER),
            dataset,
        ),
    )


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
    detail_datasets = filter_brute_force_detail_datasets(datasets)
    dataset_scopes = list(detail_datasets)
    available_skip_datasets = {
        str(dataset) for dataset in initial_skip_datasets
    }
    with gr.Tab("Brute-Force Overview") as overview_tab:
        gr.Markdown("## Brute-Force Overview")
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                overview_info = gr.Markdown(
                    BRUTE_FORCE_OVERVIEW_INFO,
                    elem_classes=["brute-force-overview-info"],
                )
            with gr.Column(scale=5, elem_classes=["stage2-right"]):
                overview_image = gr.Image(
                    label="Brute-force overview heatmaps",
                    show_label=False,
                    type="numpy",
                    interactive=False,
                    elem_classes=["results-media", "composite-plot-image"],
                )

        overview_outputs = [overview_image]

        def _refresh_overview() -> Any:
            # Overview intentionally has no controls: it always presents every
            # discovered backbone using the publication dataset sources.
            raw = refresh_overview_cb(backbones, False)
            return _compose_brute_force_overview(
                raw[: len(backbones)],
                [backbone_display_name(item) for item in backbones],
            )

        _bind_refresh(
            tab=overview_tab,
            inputs=[],
            outputs=overview_outputs,
            refresh_cb=_refresh_overview,
            triggers=[
                (overview_tab.select, True),
            ],
        )

    with gr.Tab("Brute-Force Detail") as detail_tab:
        gr.Markdown("## Brute-Force Detail")
        gr.Markdown(
            "Detailed mean-scene pruning results. Each backbone-dataset scope uses "
            "one row containing four pose metrics and three geometry metrics."
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
                for dataset in detail_datasets:
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
                colorbar_histogram = gr.Checkbox(
                    label="Show row/column mean plots",
                    value=False,
                )
                normalize_skip_metrics = gr.Checkbox(
                    label="Normalize individual skip metrics to [0,1]",
                    value=False,
                )
                info = gr.Textbox(label="Displayed scopes", interactive=False)

            with gr.Column(scale=5, elem_classes=["stage2-right"]):
                detail_image = gr.Image(
                    label="Brute-force detail heatmaps",
                    show_label=False,
                    type="numpy",
                    interactive=False,
                    elem_classes=["results-media", "composite-plot-image"],
                )

        inputs = [
            backbone_filter,
            *skip_dataset_checks,
            colorbar_histogram,
            normalize_skip_metrics,
        ]
        outputs = [detail_image, info]

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
            raw = refresh_detail_cb(
                selected_backbones,
                selected_skip_datasets,
                *detail_inputs,
            )
            gallery_count = (
                len(backbones)
                * len(stage2_modes)
                * len(dataset_scopes)
                * 2
            )
            return (
                _compose_brute_force_detail(
                    raw[:gallery_count],
                    backbones,
                    stage2_modes,
                    dataset_scopes,
                ),
                raw[-1],
            )

        _bind_refresh(
            tab=detail_tab,
            inputs=inputs,
            outputs=outputs,
            refresh_cb=_refresh_detail_from_checkboxes,
            triggers=[
                (backbone_filter.change, False),
                *[(checkbox.change, False) for checkbox in skip_dataset_checks],
                (colorbar_histogram.change, False),
                (normalize_skip_metrics.change, False),
                (detail_tab.select, True),
            ],
        )

    return {
        "overview_tab": overview_tab,
        "detail_tab": detail_tab,
        "overview_info": overview_info,
        "detail_datasets": detail_datasets,
        "overview_image": overview_image,
        "detail_image": detail_image,
    }
