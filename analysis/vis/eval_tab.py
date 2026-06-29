from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gradio as gr


def _eval_is_bundle_root(root: Path) -> bool:
    """True when ``root`` is the shared ``evaluation`` folder (suites: ``7andn``, ``scans_*``)."""
    if not root.is_dir():
        return False
    for ch in root.iterdir():
        if ch.is_dir() and (ch.name == "7andn" or ch.name.startswith("scans_")):
            return True
    return False


def _looks_like_model_subdir(path: Path) -> bool:
    """Model run dirs contain either ScanNet-style scenes or 7andn numeric frame folders."""
    if not path.is_dir() or path.name in ("vis", "__pycache__"):
        return False
    for ch in path.iterdir():
        if not ch.is_dir():
            continue
        if ch.name.isdigit():
            return True
        if (ch / "metrics.json").exists() or (ch / "vis" / "pred" / "viser_points.npz").exists():
            return True
    return False


def _list_suites(bundle: Path) -> List[str]:
    return sorted(
        c.name
        for c in bundle.iterdir()
        if c.is_dir() and (c.name == "7andn" or c.name.startswith("scans_"))
    )


def _list_models(bundle: Path, suite: str) -> List[str]:
    d = bundle / str(suite)
    if not d.is_dir():
        return []
    return sorted(x.name for x in d.iterdir() if x.is_dir() and _looks_like_model_subdir(x))


def _list_7andn_frames(bundle: Path, suite: str, model: str) -> List[str]:
    md = bundle / suite / model
    if not md.is_dir():
        return []
    return sorted((x.name for x in md.iterdir() if x.is_dir() and x.name.isdigit()), key=lambda s: int(s))


def _list_7andn_datasets(bundle: Path, suite: str, model: str, frame: str) -> List[str]:
    base = bundle / suite / model / str(frame)
    if not base.is_dir():
        return []
    names: List[str] = []
    for x in base.iterdir():
        if x.is_dir() and x.name not in ("__pycache__",):
            names.append(x.name)
    return sorted(names)


def _eval_compute_run_root(
    *,
    bundle: Path,
    suite: str,
    model: str,
    frame: str,
    dataset: str,
) -> Path:
    s, m = str(suite or "").strip(), str(model or "").strip()
    if not s or not m:
        return bundle.resolve()
    if s == "7andn":
        f, d = str(frame or "").strip(), str(dataset or "").strip()
        if f and d:
            return (bundle / s / m / f / d).resolve()
        return (bundle / s / m).resolve()
    return (bundle / s / m).resolve()


def _discover_vis_scenes(run_root: Path) -> List[str]:
    rels: set[str] = set()
    if not run_root.is_dir():
        return []
    for npz in run_root.rglob("viser_points.npz"):
        if npz.parent.name != "pred" or npz.parent.parent.name != "vis":
            continue
        scene_dir = npz.parent.parent.parent
        try:
            rels.add(scene_dir.relative_to(run_root).as_posix())
        except ValueError:
            continue
    return sorted(rels)


def _discover_metric_scenes(run_root: Path) -> List[str]:
    if not run_root.is_dir():
        return []
    return sorted(d.name for d in run_root.iterdir() if d.is_dir() and (d / "metrics.json").exists())


def _discover_scenes_union(run_root: Path) -> List[str]:
    return sorted(set(_discover_vis_scenes(run_root)) | set(_discover_metric_scenes(run_root)))


def _path_if_file(p: Path) -> Optional[str]:
    p = Path(p).resolve()
    return str(p) if p.is_file() else None


def build_eval_tab(
    *,
    blocks: gr.Blocks,
    controller: Any,
    eval_viser_controller_cls: Any,
    load_image_if_exists: Callable[[Path], Optional[Any]],
) -> Dict[str, Any]:
    _ = load_image_if_exists  # 2D uses filepath; signature kept for ``view.py`` compatibility.

    eval_root_cfg = getattr(controller, "eval_root", None)
    viser_host = getattr(controller, "viser_host", "127.0.0.1")
    viser_port = getattr(controller, "viser_port", 8090)

    bundle_path = Path(eval_root_cfg).resolve() if eval_root_cfg else Path(".").resolve()
    is_bundle = bool(eval_root_cfg) and _eval_is_bundle_root(bundle_path)

    suites_init = _list_suites(bundle_path) if is_bundle else []
    suite0 = suites_init[0] if suites_init else ""
    models_init = _list_models(bundle_path, suite0) if suite0 else []
    model0 = models_init[0] if models_init else ""
    frames_init = _list_7andn_frames(bundle_path, suite0, model0) if (is_bundle and suite0 == "7andn") else []
    frame0 = frames_init[0] if frames_init else ""
    datasets_init = (
        _list_7andn_datasets(bundle_path, suite0, model0, frame0)
        if (is_bundle and suite0 == "7andn" and frame0)
        else []
    )
    dataset0 = datasets_init[0] if datasets_init else ""

    if is_bundle and suite0 and model0:
        if suite0 == "7andn" and frame0 and dataset0:
            initial_run_root = _eval_compute_run_root(
                bundle=bundle_path, suite=suite0, model=model0, frame=frame0, dataset=dataset0
            )
        else:
            initial_run_root = _eval_compute_run_root(
                bundle=bundle_path, suite=suite0, model=model0, frame="", dataset=""
            )
    elif eval_root_cfg:
        initial_run_root = bundle_path
    else:
        initial_run_root = bundle_path

    eval_ctrl = None
    try:
        if initial_run_root.is_dir():
            eval_ctrl = eval_viser_controller_cls(
                eval_root=initial_run_root, host=str(viser_host), port=int(viser_port)
            )
    except Exception:
        eval_ctrl = None

    with gr.Tab("Eval Results") as tab_eval:
        gr.Markdown("## Eval Results (2D + 3D)")
        gr.Markdown(
            "2D: per-scene `plot.png` when present. 3D: Viser point cloud + trajectory "
            "(requires `vis/pred/viser_points.npz`). "
            "With `--eval-root` set to `analysis_outputs/evaluation`, pick **eval suite → model → "
            "(7andn: frame + dataset) → scene**; with a single run folder, only **Scene** appears."
        )

        iframe_html = (
            f'<iframe src="http://{viser_host}:{int(viser_port)}" '
            'style="width:100%; height:650px; border:0;"></iframe>'
        )

        with gr.Row(visible=is_bundle) as eval_bundle_row:
            eval_suite_dd = gr.Dropdown(label="Eval suite", choices=suites_init, value=suite0 or None)
            eval_model_dd = gr.Dropdown(label="Model / run", choices=models_init, value=model0 or None)
            eval_frame_dd = gr.Dropdown(
                label="7andn frame idx",
                choices=frames_init,
                value=frame0 or None,
                visible=is_bundle and suite0 == "7andn",
            )
            eval_dataset_dd = gr.Dropdown(
                label="Dataset",
                choices=datasets_init,
                value=dataset0 or None,
                visible=is_bundle and suite0 == "7andn",
            )

        with gr.Row():
            with gr.Column(scale=1):
                eval_scene_dd = gr.Dropdown(
                    label="Scene (or scene/seq path)",
                    choices=[],
                    value=None,
                )
                eval_view_mode = gr.Radio(label="View mode", choices=["2D", "3D"], value="2D")
                eval_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=5):
                eval_plot_img = gr.Image(label="plot.png", type="filepath", elem_classes=["results-media"])
                eval_3d_html = gr.HTML(value=iframe_html, visible=False)

        def _full_refresh(suite: str, model: str, frame: str, dataset: str, scene_id: str | None, view_mode: str):
            if not eval_root_cfg:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(choices=[], value=None),
                    None,
                    gr.update(visible=False),
                    "Eval root not set (--eval-root).",
                )

            if is_bundle:
                suites = _list_suites(bundle_path)
                suite_s = str(suite or "").strip()
                if suite_s not in suites:
                    suite_s = suites[0] if suites else ""
                models = _list_models(bundle_path, suite_s)
                model_s = str(model or "").strip()
                if model_s not in models:
                    model_s = models[0] if models else ""
                frames = _list_7andn_frames(bundle_path, suite_s, model_s) if suite_s == "7andn" else []
                frame_s = str(frame or "").strip()
                if suite_s == "7andn" and frames and frame_s not in frames:
                    frame_s = frames[0]
                datasets = (
                    _list_7andn_datasets(bundle_path, suite_s, model_s, frame_s)
                    if (suite_s == "7andn" and frame_s)
                    else []
                )
                dataset_s = str(dataset or "").strip()
                if suite_s == "7andn" and datasets and dataset_s not in datasets:
                    dataset_s = datasets[0]
                run_root = _eval_compute_run_root(
                    bundle=bundle_path, suite=suite_s, model=model_s, frame=frame_s, dataset=dataset_s
                )
                upd_suite = gr.update(choices=suites, value=suite_s)
                upd_model = gr.update(choices=models, value=model_s)
                upd_frame = gr.update(
                    choices=frames,
                    value=frame_s if frame_s in frames else (frames[0] if frames else None),
                    visible=suite_s == "7andn",
                )
                upd_dataset = gr.update(
                    choices=datasets,
                    value=dataset_s if dataset_s in datasets else (datasets[0] if datasets else None),
                    visible=suite_s == "7andn",
                )
            else:
                run_root = bundle_path
                suite_s = model_s = frame_s = dataset_s = ""
                upd_suite = gr.update()
                upd_model = gr.update()
                upd_frame = gr.update()
                upd_dataset = gr.update()

            scenes = _discover_scenes_union(run_root)
            scene_s = str(scene_id or "").strip()
            if scene_s not in scenes:
                scene_s = scenes[0] if scenes else None

            if eval_ctrl is not None:
                try:
                    eval_ctrl.set_eval_root(run_root)
                except Exception:
                    pass

            plot_path: Optional[str] = None
            status_extra = f"run_root={run_root}"
            if scene_s:
                p = run_root / scene_s / "plot.png"
                plot_path = _path_if_file(p)
                if plot_path is None:
                    status_extra += f" | no plot.png for {scene_s!r}"
            else:
                status_extra += " | no scenes under run root"

            vm = str(view_mode or "2D")
            if vm == "2D":
                return (
                    upd_suite,
                    upd_model,
                    upd_frame,
                    upd_dataset,
                    gr.update(choices=scenes, value=scene_s),
                    plot_path,
                    gr.update(visible=False),
                    f"{status_extra} | 2D",
                )
            if eval_ctrl is None:
                return (
                    upd_suite,
                    upd_model,
                    upd_frame,
                    upd_dataset,
                    gr.update(choices=scenes, value=scene_s),
                    None,
                    gr.update(visible=True),
                    f"{status_extra} | Viser not available.",
                )
            if not scene_s:
                return (
                    upd_suite,
                    upd_model,
                    upd_frame,
                    upd_dataset,
                    gr.update(choices=scenes, value=None),
                    None,
                    gr.update(visible=True),
                    f"{status_extra} | select a scene for 3D",
                )
            msg = eval_ctrl.load_scene(scene_s, show_gt=True)
            return (
                upd_suite,
                upd_model,
                upd_frame,
                upd_dataset,
                gr.update(choices=scenes, value=scene_s),
                None,
                gr.update(visible=True),
                f"{status_extra} | {msg}",
            )

        inputs_all = [
            eval_suite_dd,
            eval_model_dd,
            eval_frame_dd,
            eval_dataset_dd,
            eval_scene_dd,
            eval_view_mode,
        ]
        outputs_all = [
            eval_suite_dd,
            eval_model_dd,
            eval_frame_dd,
            eval_dataset_dd,
            eval_scene_dd,
            eval_plot_img,
            eval_3d_html,
            eval_status,
        ]

        for comp in (
            eval_suite_dd,
            eval_model_dd,
            eval_frame_dd,
            eval_dataset_dd,
            eval_scene_dd,
            eval_view_mode,
        ):
            comp.change(
                _full_refresh,
                inputs=inputs_all,
                outputs=outputs_all,
                show_progress="minimal",
                trigger_mode="always_last",
            )
        tab_eval.select(
            _full_refresh,
            inputs=inputs_all,
            outputs=outputs_all,
            show_progress="minimal",
        )
        blocks.load(
            _full_refresh,
            inputs=inputs_all,
            outputs=outputs_all,
            show_progress="minimal",
        )

    return {"tab": tab_eval}
