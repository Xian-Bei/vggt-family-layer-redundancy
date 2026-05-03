from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gradio as gr


def build_lora_tab(
    *,
    blocks: gr.Blocks,
    lora_root: Path,
    discover_lora_experiments: Callable[[Path], List[str]],
    lora_expected_plot_paths: Callable[[Path], Dict[str, Path]],
) -> Dict[str, Any]:
    lora_exps = discover_lora_experiments(Path(lora_root))
    initial_lora_exp = lora_exps[0] if lora_exps else None

    with gr.Tab("LoRA results") as tab_lora:
        gr.Markdown("## LoRA results")
        gr.Markdown(
            "Each subfolder under `--lora-dir` is treated as one experiment run. "
            "Select one on the left to display its saved plots on the right."
        )

        def _path_if_file(p: Path) -> Optional[str]:
            p = Path(p).resolve()
            return str(p) if p.is_file() else None

        def _refresh_lora(exp_name: str):
            exp_name = str(exp_name or "").strip()
            if not exp_name:
                empty: List[Any] = [None] * 12
                return (*empty, "No experiment selected.")
            exp_dir = (Path(lora_root) / exp_name).resolve()
            paths = lora_expected_plot_paths(exp_dir)
            # Gradio 5: serve images by absolute filepath (avoids numpy payload / init refresh gaps).
            imgs = {k: _path_if_file(p) for k, p in paths.items()}
            missing = [k for k, v in imgs.items() if v is None]
            details = f"exp={exp_name} | dir={exp_dir}"
            details += " | all plots found" if not missing else f" | missing: {', '.join(missing)}"
            return (
                imgs["eval_plot_are"],
                imgs["eval_plot_ate"],
                imgs["eval_plot_rpe_rot"],
                imgs["eval_plot_rpe_trans"],
                imgs["eval_plot_depth_abs_delta1"],
                imgs["eval_plot_depth_abs_abs_rel"],
                imgs["eval_plot_chamfer_distance"],
                imgs["eval_plot_7andn_acc_med"],
                imgs["eval_plot_7andn_comp_med"],
                imgs["eval_plot_7andn_nc_med"],
                imgs["train_curves"],
                imgs["train_loss_weighted_components"],
                details,
            )

        with gr.Row():
            with gr.Column(scale=1):
                lora_exp_dd = gr.Dropdown(label="Experiment", choices=lora_exps, value=initial_lora_exp)
                lora_info = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=5):
                with gr.Row():
                    lora_eval_are = gr.Image(
                        label="eval_plot_scannet_are", type="filepath", elem_classes=["results-media"]
                    )
                    lora_eval_ate = gr.Image(
                        label="eval_plot_scannet_ate", type="filepath", elem_classes=["results-media"]
                    )
                    lora_eval_rpe_rot = gr.Image(
                        label="eval_plot_scannet_rpe_rot", type="filepath", elem_classes=["results-media"]
                    )
                    lora_eval_rpe_trans = gr.Image(
                        label="eval_plot_scannet_rpe_trans", type="filepath", elem_classes=["results-media"]
                    )
                with gr.Row():
                    lora_eval_depth_delta1 = gr.Image(
                        label="eval_plot_scannet_depth_delta1", type="filepath", elem_classes=["results-media"]
                    )
                    lora_eval_depth_abs_rel = gr.Image(
                        label="eval_plot_scannet_depth_abs_rel", type="filepath", elem_classes=["results-media"]
                    )
                    lora_eval_chamfer = gr.Image(
                        label="eval_plot_scannet_chamfer_distance", type="filepath", elem_classes=["results-media"]
                    )
                gr.Markdown(
                    "**7-scenes / NRGBD** (eval_plot_7andn_*.png). Slots stay empty if that periodic-eval group was not run."
                )
                with gr.Row():
                    lora_eval_7n_acc_med = gr.Image(
                        label="eval_plot_7andn_acc_med", type="filepath", elem_classes=["results-media"]
                    )
                    lora_eval_7n_comp_med = gr.Image(
                        label="eval_plot_7andn_comp_med", type="filepath", elem_classes=["results-media"]
                    )
                    lora_eval_7n_nc_med = gr.Image(
                        label="eval_plot_7andn_nc_med", type="filepath", elem_classes=["results-media"]
                    )
                with gr.Row():
                    lora_train_curves = gr.Image(
                        label="train_curves.png", type="filepath", elem_classes=["results-media"]
                    )
                    lora_train_components = gr.Image(
                        label="train_loss_weighted_components.png", type="filepath", elem_classes=["results-media"]
                    )

        lora_outputs = [
            lora_eval_are,
            lora_eval_ate,
            lora_eval_rpe_rot,
            lora_eval_rpe_trans,
            lora_eval_depth_delta1,
            lora_eval_depth_abs_rel,
            lora_eval_chamfer,
            lora_eval_7n_acc_med,
            lora_eval_7n_comp_med,
            lora_eval_7n_nc_med,
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
        # Dropdown ``change`` does not run for the initial default value; ``Tab.select`` is
        # also unreliable on first paint. Populate once when the Blocks app loads.
        blocks.load(
            _refresh_lora,
            inputs=[lora_exp_dd],
            outputs=lora_outputs,
            show_progress="minimal",
        )

    return {"tab": tab_lora}
