from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gradio as gr
import numpy as np


def build_lora_tab(
    *,
    lora_root: Path,
    discover_lora_experiments: Callable[[Path], List[str]],
    lora_expected_plot_paths: Callable[[Path], Dict[str, Path]],
    load_image_if_exists: Callable[[Path], Optional[np.ndarray]],
) -> Dict[str, Any]:
    lora_exps = discover_lora_experiments(Path(lora_root))
    initial_lora_exp = lora_exps[0] if lora_exps else None

    with gr.Tab("LoRA results") as tab_lora:
        gr.Markdown("## LoRA results")
        gr.Markdown(
            "Each subfolder under `--lora-dir` is treated as one experiment run. "
            "Select one on the left to display its saved plots on the right."
        )

        def _refresh_lora(exp_name: str):
            exp_name = str(exp_name or "").strip()
            if not exp_name:
                empty: List[Any] = [None] * 9
                return (*empty, "No experiment selected.")
            exp_dir = Path(lora_root) / exp_name
            paths = lora_expected_plot_paths(exp_dir)
            images: Dict[str, Optional[np.ndarray]] = {k: load_image_if_exists(p) for k, p in paths.items()}
            missing = [k for k, v in images.items() if v is None]
            details = f"exp={exp_name} | dir={exp_dir}"
            details += " | all plots found" if not missing else f" | missing: {', '.join(missing)}"
            return (
                images["eval_plot_are"],
                images["eval_plot_ate"],
                images["eval_plot_rpe_rot"],
                images["eval_plot_rpe_trans"],
                images["eval_plot_depth_abs_delta1"],
                images["eval_plot_depth_abs_abs_rel"],
                images["eval_plot_chamfer_distance"],
                images["train_curves"],
                images["train_loss_weighted_components"],
                details,
            )

        with gr.Row():
            with gr.Column(scale=1):
                lora_exp_dd = gr.Dropdown(label="Experiment", choices=lora_exps, value=initial_lora_exp)
                lora_info = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=5):
                with gr.Row():
                    lora_eval_are = gr.Image(label="eval_plot_are", type="numpy", elem_classes=["results-media"])
                    lora_eval_ate = gr.Image(label="eval_plot_ate", type="numpy", elem_classes=["results-media"])
                    lora_eval_rpe_rot = gr.Image(label="eval_plot_rpe_rot", type="numpy", elem_classes=["results-media"])
                    lora_eval_rpe_trans = gr.Image(label="eval_plot_rpe_trans", type="numpy", elem_classes=["results-media"])
                with gr.Row():
                    lora_eval_depth_delta1 = gr.Image(
                        label="eval_plot_depth_abs_delta1", type="numpy", elem_classes=["results-media"]
                    )
                    lora_eval_depth_abs_rel = gr.Image(
                        label="eval_plot_depth_abs_abs_rel", type="numpy", elem_classes=["results-media"]
                    )
                    lora_eval_chamfer = gr.Image(
                        label="eval_plot_chamfer_distance", type="numpy", elem_classes=["results-media"]
                    )
                with gr.Row():
                    lora_train_curves = gr.Image(label="train_curves.png", type="numpy", elem_classes=["results-media"])
                    lora_train_components = gr.Image(
                        label="train_loss_weighted_components.png", type="numpy", elem_classes=["results-media"]
                    )

        lora_outputs = [
            lora_eval_are,
            lora_eval_ate,
            lora_eval_rpe_rot,
            lora_eval_rpe_trans,
            lora_eval_depth_delta1,
            lora_eval_depth_abs_rel,
            lora_eval_chamfer,
            lora_train_curves,
            lora_train_components,
            lora_info,
        ]
        lora_exp_dd.change(
            _refresh_lora,
            inputs=[lora_exp_dd],
            outputs=lora_outputs,
            show_progress="minimal",
            trigger_mode="always_last",
        )
        tab_lora.select(
            _refresh_lora,
            inputs=[lora_exp_dd],
            outputs=lora_outputs,
            show_progress="minimal",
        )

    return {"tab": tab_lora}
