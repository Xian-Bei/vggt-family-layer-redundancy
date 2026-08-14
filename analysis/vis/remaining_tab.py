from __future__ import annotations

from typing import Any, Dict, Sequence

import gradio as gr

from analysis.vis.display_names import (
    backbone_display_name,
    backbone_publication_sort_key,
)

REMAINING_COLUMN_LABELS = (
    "7-Metric Normalized Retention",
    "Marginal 7-Metric Mean",
)


def build_remaining_tab(
    *,
    dataset_choices: Sequence[str],
    backbone_choices: Sequence[str],
    initial_dataset: str,
    refresh_remaining_cb: Any,
) -> Dict[str, Any]:
    """Build the compact original-vs-marginal 7-metric comparison tab."""

    choices = [str(dataset) for dataset in dataset_choices]
    backbones = sorted(
        [str(backbone) for backbone in backbone_choices],
        key=backbone_publication_sort_key,
    )
    selected = str(initial_dataset)
    if selected not in choices:
        selected = "mean" if "mean" in choices else (choices[0] if choices else "")

    with gr.Tab("Remaining") as tab:
        gr.Markdown("## Remaining")
        gr.Markdown(
            "Each backbone is one row with a shared model heading. The two columns "
            "show the original **7-metric mean** and its **remaining marginal** "
            "heatmap. Both use scene `mean`."
        )
        dataset = gr.Dropdown(
            label="Dataset",
            choices=choices,
            value=selected,
        )
        galleries = []
        backbone_rows = []
        for backbone in backbones:
            with gr.Column(
                elem_classes=["publication-backbone-block"]
            ) as backbone_row:
                gr.Markdown(
                    f"### {backbone_display_name(backbone)}",
                    elem_classes=["backbone-separator"],
                )
                gallery = gr.Gallery(
                    label="Original / marginal 7-metric mean",
                    show_label=False,
                    columns=2,
                    height="auto",
                    object_fit="contain",
                    elem_classes=[
                        "results-gallery",
                        "stage2-heatmap-gallery",
                        "remaining-gallery",
                    ],
                )
            galleries.append(gallery)
            backbone_rows.append(backbone_row)
        with gr.Row(
            elem_classes=["remaining-column-label-row"],
            equal_height=True,
        ):
            for column_label in REMAINING_COLUMN_LABELS:
                gr.Markdown(
                    column_label,
                    elem_classes=["remaining-column-label"],
                )
        info = gr.Textbox(label="Displayed counts", interactive=False)

        outputs = [*galleries, *backbone_rows, info]
        dataset.change(
            refresh_remaining_cb,
            inputs=[dataset],
            outputs=outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        tab.select(
            refresh_remaining_cb,
            inputs=[dataset],
            outputs=outputs,
            show_progress="minimal",
        )

    return {
        "tab": tab,
        "dataset": dataset,
        "galleries": galleries,
        "backbone_rows": backbone_rows,
        "info": info,
    }
