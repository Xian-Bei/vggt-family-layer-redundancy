from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gradio as gr
import numpy as np


def build_eval_tab(
    *,
    controller: Any,
    eval_viser_controller_cls: Any,
    load_image_if_exists: Callable[[Path], Optional[np.ndarray]],
) -> Dict[str, Any]:
    with gr.Tab("Eval Visualization") as tab_eval:
        gr.Markdown("## Eval visualization (2D + 3D)")
        gr.Markdown("2D: per-scene `plot.png` (if present). 3D: Viser pointcloud + trajectory (requires exported vis assets).")

        eval_root = getattr(controller, "eval_root", None)
        viser_host = getattr(controller, "viser_host", "127.0.0.1")
        viser_port = getattr(controller, "viser_port", 8090)
        eval_ctrl = None
        if eval_root:
            try:
                eval_ctrl = eval_viser_controller_cls(eval_root=Path(eval_root), host=str(viser_host), port=int(viser_port))
            except Exception:
                eval_ctrl = None

        def _discover_eval_scenes(root: str | None) -> List[str]:
            if not root:
                return []
            p = Path(root)
            if not p.exists():
                return []
            out = []
            for d in sorted(p.iterdir()):
                if d.is_dir() and (d / "metrics.json").exists():
                    out.append(d.name)
            return out

        eval_scenes = _discover_eval_scenes(str(eval_root) if eval_root else None)
        iframe_html = f'<iframe src="http://{viser_host}:{int(viser_port)}" style="width:100%; height:650px; border:0;"></iframe>'
        with gr.Row():
            with gr.Column(scale=1):
                eval_scene_dd = gr.Dropdown(label="Scene", choices=eval_scenes, value=eval_scenes[0] if eval_scenes else None)
                eval_view_mode = gr.Radio(label="View mode", choices=["2D", "3D"], value="2D")
                eval_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=5):
                eval_plot_img = gr.Image(label="plot.png", type="numpy", elem_classes=["results-media"])
                eval_3d_html = gr.HTML(value=iframe_html, visible=False)

        def _load_plot(scene_id: str | None):
            if not eval_root or not scene_id:
                return None, "Eval root not set."
            path = Path(eval_root) / str(scene_id) / "plot.png"
            if not path.exists():
                return None, f"Missing {path}"
            arr = load_image_if_exists(path)
            return arr, f"Loaded {path}"

        def _refresh_eval_view(scene_id: str | None, view_mode: str):
            if view_mode == "2D":
                img, status = _load_plot(scene_id)
                return gr.update(visible=True, value=img), gr.update(visible=False), status
            if eval_ctrl is None:
                return gr.update(visible=False, value=None), gr.update(visible=True), "Eval visualization not configured (missing --eval-root)."
            if not scene_id:
                return gr.update(visible=False, value=None), gr.update(visible=True), "No scene selected."
            status = eval_ctrl.load_scene(str(scene_id), show_gt=True)
            return gr.update(visible=False, value=None), gr.update(visible=True), status

        eval_scene_dd.change(
            _refresh_eval_view,
            inputs=[eval_scene_dd, eval_view_mode],
            outputs=[eval_plot_img, eval_3d_html, eval_status],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        eval_view_mode.change(
            _refresh_eval_view,
            inputs=[eval_scene_dd, eval_view_mode],
            outputs=[eval_plot_img, eval_3d_html, eval_status],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        tab_eval.select(
            _refresh_eval_view,
            inputs=[eval_scene_dd, eval_view_mode],
            outputs=[eval_plot_img, eval_3d_html, eval_status],
            show_progress="minimal",
        )

    return {"tab": tab_eval}
