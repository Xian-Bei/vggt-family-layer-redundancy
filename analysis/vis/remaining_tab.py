from __future__ import annotations

from typing import Any, Dict, Sequence

import gradio as gr


def build_remaining_tab(
    *,
    dataset_choices: Sequence[str],
    initial_dataset: str,
    refresh_remaining_cb: Any,
) -> Dict[str, Any]:
    """Build the compact original-vs-marginal 7-metric comparison tab."""

    choices = [str(dataset) for dataset in dataset_choices]
    selected = str(initial_dataset)
    if selected not in choices:
        selected = "mean" if "mean" in choices else (choices[0] if choices else "")

    with gr.Tab("Remaining") as tab:
        gr.Markdown("## Remaining")
        gr.Markdown(
            "Each column is one backbone. The first row is the original "
            "**7-metric mean**; the "
            "second row is its **remaining marginal** heatmap. Both use scene `mean`."
        )
        dataset = gr.Dropdown(
            label="Dataset",
            choices=choices,
            value=selected,
        )
        with gr.Row(elem_classes=["stage2-gallery-row"], equal_height=False):
            original_gallery = gr.Gallery(
                label="7-metric mean",
                columns=max(1, len(choices)),
                height="auto",
                object_fit="contain",
                elem_classes=["results-gallery", "stage2-heatmap-gallery"],
            )
        with gr.Row(elem_classes=["stage2-gallery-row"], equal_height=False):
            marginal_gallery = gr.Gallery(
                label="Remaining marginal 7-metric mean",
                columns=max(1, len(choices)),
                height="auto",
                object_fit="contain",
                elem_classes=["results-gallery", "stage2-heatmap-gallery"],
            )
        info = gr.Textbox(label="Displayed counts", interactive=False)

        outputs = [original_gallery, marginal_gallery, info]
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
        "original_gallery": original_gallery,
        "marginal_gallery": marginal_gallery,
        "info": info,
    }
