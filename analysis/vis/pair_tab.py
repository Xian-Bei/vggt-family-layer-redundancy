from __future__ import annotations

from typing import Any, Dict, List, Sequence

import gradio as gr


def build_pair_tab(
    *,
    controller: Any,
    dataset_choices: Sequence[str],
    scene_ids: List[str],
    initial_scene: str | None,
    num_layers: int,
    num_frames: int,
    frame_labels: List[str],
    initial_frame_a: int,
    initial_frame_b: int,
    num_heads: int,
    scene_changed_cb: Any,
    refresh_outputs_cb: Any,
) -> Dict[str, Any]:
    pair_sliders_live = controller.cache_root_exists and bool(scene_ids)
    with gr.Tab("Pair Attention", interactive=controller.cache_root_exists) as tab_pair:
        gr.Markdown("## Pair Attention Panel")
        gr.Markdown(
            "Choose a Stage1-cached scene, global layer, two frames, score space, and head view. "
            "The panel refreshes automatically and shows overlay summaries for both directions: `i -> j` and `j -> i`."
        )
        if not controller.cache_root_exists:
            gr.Markdown(
                "This tab is disabled because `--cache-dir` does not exist or is not a directory. "
                "Heatmaps and other tabs can still use similarity/skip outputs."
            )
        elif not bool(scene_ids):
            gr.Markdown(
                "Pair Attention unavailable for this dataset. "
                f"Expected cache manifest at `{controller.cache_dir / 'manifest.json'}` with usable scene entries "
                "and a valid HF model id."
            )

        with gr.Row():
            with gr.Column(scale=1):
                dataset_dd = gr.Dropdown(
                    label="Dataset",
                    choices=list(dataset_choices),
                    value=controller.dataset,
                    info="Select cache/similarity/skip subdirectory under analysis_outputs/*/<dataset>/",
                )
                scene_dd = gr.Dropdown(
                    label="Scene",
                    choices=scene_ids,
                    value=initial_scene,
                    interactive=pair_sliders_live,
                )
                scene_info = gr.Textbox(
                    label="Scene metadata",
                    value=f"layers: {num_layers}, frames: {num_frames}",
                    interactive=False,
                )
                layer_slider = gr.Slider(
                    label="Global layer",
                    minimum=0,
                    maximum=max(0, num_layers - 1),
                    value=0,
                    step=1,
                    interactive=pair_sliders_live,
                )
                frame_a_slider = gr.Slider(
                    label="Frame A",
                    minimum=0,
                    maximum=max(0, num_frames - 1),
                    value=initial_frame_a,
                    step=1,
                    interactive=pair_sliders_live,
                )
                frame_a_info = gr.Textbox(
                    label="Frame A selection",
                    value=frame_labels[initial_frame_a] if frame_labels else "",
                    interactive=False,
                )
                frame_b_slider = gr.Slider(
                    label="Frame B",
                    minimum=0,
                    maximum=max(0, num_frames - 1),
                    value=initial_frame_b,
                    step=1,
                    interactive=pair_sliders_live,
                )
                frame_b_info = gr.Textbox(
                    label="Frame B selection",
                    value=frame_labels[initial_frame_b] if frame_labels else "",
                    interactive=False,
                )
                score_space = gr.Radio(
                    label="Score space",
                    choices=["before_softmax", "after_softmax"],
                    value="before_softmax",
                    interactive=pair_sliders_live,
                )
                head_mode = gr.Radio(label="Head view", choices=["mean", "single"], value="mean", interactive=pair_sliders_live)
                head_index = gr.Slider(
                    label="Head index",
                    minimum=0,
                    maximum=max(0, num_heads - 1),
                    value=0,
                    step=1,
                    interactive=pair_sliders_live,
                )
                summary = gr.Textbox(label="Selection", interactive=False)
            with gr.Column(scale=5):
                panel = gr.Image(label="Bidirectional pair attention panel", type="numpy", elem_classes=["results-media"])

        refresh_inputs = [scene_dd, layer_slider, frame_a_slider, frame_b_slider, score_space, head_mode, head_index]
        refresh_outputs = [panel, summary, frame_a_info, frame_b_info]

        scene_dd.change(
            scene_changed_cb,
            inputs=[scene_dd, score_space, head_mode, head_index],
            outputs=[layer_slider, frame_a_slider, frame_b_slider, scene_info, panel, summary, frame_a_info, frame_b_info],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        layer_slider.change(
            refresh_outputs_cb,
            inputs=refresh_inputs,
            outputs=refresh_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        frame_a_slider.change(
            refresh_outputs_cb,
            inputs=refresh_inputs,
            outputs=refresh_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        frame_b_slider.change(
            refresh_outputs_cb,
            inputs=refresh_inputs,
            outputs=refresh_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        score_space.change(
            refresh_outputs_cb,
            inputs=refresh_inputs,
            outputs=refresh_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        head_mode.change(
            refresh_outputs_cb,
            inputs=refresh_inputs,
            outputs=refresh_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        head_index.change(
            refresh_outputs_cb,
            inputs=refresh_inputs,
            outputs=refresh_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        tab_pair.select(
            refresh_outputs_cb,
            inputs=refresh_inputs,
            outputs=refresh_outputs,
            show_progress="minimal",
        )

    return {
        "tab": tab_pair,
        "dataset_dd": dataset_dd,
        "scene_dd": scene_dd,
        "scene_info": scene_info,
        "layer_slider": layer_slider,
        "frame_a_slider": frame_a_slider,
        "frame_b_slider": frame_b_slider,
        "frame_a_info": frame_a_info,
        "frame_b_info": frame_b_info,
        "score_space": score_space,
        "head_mode": head_mode,
        "head_index": head_index,
        "panel": panel,
        "summary": summary,
    }
