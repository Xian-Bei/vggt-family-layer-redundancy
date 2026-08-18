from __future__ import annotations

from typing import Any, Dict, Sequence

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
)

# Remaining typography and outer-label spacing (pixels).
REMAINING_SUBPLOT_FONT_SIZE = 24
REMAINING_OUTER_FONT_SIZE = 28
REMAINING_LEFT_LABEL_TO_IMAGE_GAP = 0
REMAINING_TOP_LABEL_TO_IMAGE_GAP = 0
REMAINING_BOTTOM_LABEL_TO_IMAGE_GAP = 0

REMAINING_COLUMN_LABELS = (
    "Normalized Retention",
    "Marginal Normalized Retention",
)


def _compose_remaining(
    gallery_values: Sequence[Any],
    backbone_labels: Sequence[str],
) -> Any:
    cells: list[Cell] = []
    groups: list[tuple[str, int]] = []
    columns: list[str] = []
    for backbone, value in zip(backbone_labels, gallery_values):
        group = gallery_cells(value)[: len(REMAINING_COLUMN_LABELS)]
        group.extend(Cell(None) for _ in range(len(REMAINING_COLUMN_LABELS) - len(group)))
        cells.extend(group)
        groups.append((backbone, len(group)))
        columns.extend(REMAINING_COLUMN_LABELS)
    return compose_sections(
        [
            Section(
                columns=len(cells),
                cells=[],
                fixed_rows=[cells],
                group_headings=groups,
                column_labels=columns,
                column_labels_below=True,
                column_labels_bold=True,
            )
        ],
        outer_font_size=REMAINING_OUTER_FONT_SIZE,
        left_label_to_image_gap=REMAINING_LEFT_LABEL_TO_IMAGE_GAP,
        top_label_to_image_gap=REMAINING_TOP_LABEL_TO_IMAGE_GAP,
        bottom_label_to_image_gap=REMAINING_BOTTOM_LABEL_TO_IMAGE_GAP,
    )


def build_remaining_tab(
    *,
    dataset_choices: Sequence[str],
    backbone_choices: Sequence[str],
    initial_dataset: str,
    refresh_remaining_cb: Any,
) -> Dict[str, Any]:
    """Build the compact original-vs-marginal 7-metric comparison tab."""

    choices = [
        str(dataset)
        for dataset in dataset_choices
        if str(dataset).strip().lower() != "min"
    ]
    backbones = sorted(
        [str(backbone) for backbone in backbone_choices],
        key=backbone_publication_sort_key,
    )
    selected = "mean" if "mean" in choices else (choices[0] if choices else "")

    with gr.Tab("Remaining") as tab:
        gr.Markdown("## Remaining")
        gr.Markdown(
            "All backbones share one row. Each model heading spans two columns that "
            "show the original **7-metric mean** and its **remaining marginal** "
            "heatmap. Both use scene `mean`."
        )
        dataset = gr.Dropdown(
            label="Dataset",
            choices=choices,
            value=selected,
        )
        image = gr.Image(
            label="Original / marginal 7-metric mean",
            show_label=False,
            type="numpy",
            interactive=False,
            elem_classes=["results-media", "composite-plot-image"],
        )
        info = gr.Textbox(label="Displayed counts", interactive=False)

        def refresh(dataset_value: str):
            values = refresh_remaining_cb(dataset_value)
            gallery_values = values[: len(backbones)]
            return (
                _compose_remaining(
                    gallery_values,
                    [backbone_display_name(item) for item in backbones],
                ),
                values[-1],
            )

        outputs = [image, info]
        dataset.change(
            refresh,
            inputs=[dataset],
            outputs=outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        tab.select(
            refresh,
            inputs=[dataset],
            outputs=outputs,
            show_progress="minimal",
        )

    return {
        "tab": tab,
        "dataset": dataset,
        "image": image,
        "info": info,
    }
