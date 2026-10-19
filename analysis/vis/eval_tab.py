from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import html
import io
import json
import math
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .display_names import backbone_display_name


# These are the three deployment budgets selected by ``main.sh``.
VISUALIZATION_BUDGETS: Dict[str, int] = {
    "vggt": 10,
    "pi3": 16,
    "vggt_omega": 18,
}
VISUALIZATION_BACKBONES = tuple(VISUALIZATION_BUDGETS)
VISUALIZATION_RECOVERY_PLAN = "recovery_ghost_interval_sequential_autogroup"
VISUALIZATION_REPLACEME_RECOVERY_PLAN = "recovery_replaceme_ls"
VISUALIZATION_SELECTION_DATASETS = (
    "scannet_dev50",
    "7scenes_train",
    "nuscenes_train",
    "waymo_training",
)


@dataclass(frozen=True)
class VisualizationMethod:
    key: str
    label: str
    run_root: Path
    asset_root: Optional[Path] = None
    show_in_2d: bool = True


def _method_asset_root(method: VisualizationMethod) -> Path:
    return method.asset_root if method.asset_root is not None else method.run_root


def _looks_like_run_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "eval_record_v3.json").is_file() or (path / "summary.json").is_file():
        return True
    return any(child.is_dir() for child in path.iterdir())


def _compression_label(name: str) -> str:
    value = str(name).strip()
    if value.startswith("fastvggt"):
        return "FastVGGT"
    if value.startswith("quantvggt"):
        suffix = value.removeprefix("quantvggt_")
        return f"QuantVGGT ({suffix.upper()})" if suffix != value and suffix else "QuantVGGT"
    return value.replace("_", " ")


@lru_cache(maxsize=32)
def _selected_recovery_method(
    eval_root: Path, budget: int
) -> Optional[VisualizationMethod]:
    """Resolve Ours directly from its budget selection, without test-time runs."""
    try:
        from analysis.recovery.shared_results import best_recovery_result

        # .../<output>/<backbone>/<dataset>/eval -> .../<output>/<backbone>
        plan_root = Path(eval_root).parents[1]
        selected = best_recovery_result(
            plan_root,
            drop_count=int(budget),
            datasets=VISUALIZATION_SELECTION_DATASETS,
            recovery_plan_name=VISUALIZATION_RECOVERY_PLAN,
            strategies=("retention_single", "retention_double"),
        )
        selector = str(selected["selector"])
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None
    candidate = Path(eval_root) / "recovery" / selector / VISUALIZATION_RECOVERY_PLAN
    if not _looks_like_run_root(candidate):
        return None
    return VisualizationMethod(
        key=f"budget:none:{VISUALIZATION_RECOVERY_PLAN}:{selector}",
        label=f"Budget {budget} · No test-time",
        run_root=candidate.resolve(),
    )


@lru_cache(maxsize=32)
def _selected_replaceme_method(
    eval_root: Path, budget: int
) -> Optional[VisualizationMethod]:
    """Resolve the calibrated ReplaceMe winner recorded by its main run."""
    plan_root = Path(eval_root).parents[1]
    selection_path = plan_root / f"replaceme_selection_budget{int(budget)}.json"
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        if int(payload.get("budget", -1)) != int(budget):
            return None
        winner = payload["winner"]
        selector = str(winner["selector"]).strip()
        strategy = str(winner.get("strategy", "")).strip()
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not selector or strategy not in {"replaceme_single", "replaceme_double"}:
        return None
    candidate = (
        Path(eval_root)
        / "recovery"
        / selector
        / VISUALIZATION_REPLACEME_RECOVERY_PLAN
    )
    if not _looks_like_run_root(candidate):
        return None
    structure = "Single" if strategy.endswith("_single") else "Double"
    return VisualizationMethod(
        key=(
            f"replaceme:{VISUALIZATION_REPLACEME_RECOVERY_PLAN}:"
            f"{selector}"
        ),
        label=f"Budget {budget} · ReplaceMe ({structure})",
        run_root=candidate.resolve(),
        show_in_2d=False,
    )


def _selected_budget_runs(eval_root: Path, budget: int) -> tuple[Optional[VisualizationMethod], List[VisualizationMethod]]:
    """Find the uncompressed and test-time runs selected for one budget."""
    testtime_root = eval_root / "testtime"
    compressed: List[VisualizationMethod] = []
    selected: List[tuple[str, str]] = []
    if testtime_root.is_dir():
        for recovery_dir in sorted(p for p in testtime_root.iterdir() if p.is_dir()):
            for compression_dir in sorted(p for p in recovery_dir.iterdir() if p.is_dir()):
                for selector_dir in sorted(p for p in compression_dir.iterdir() if p.is_dir()):
                    if not _looks_like_run_root(selector_dir):
                        continue
                    selected.append((selector_dir.name, recovery_dir.name))
                    compression = compression_dir.name
                    compressed.append(
                        VisualizationMethod(
                            key=f"budget:{recovery_dir.name}:{compression}:{selector_dir.name}",
                            label=f"Budget {budget} + {_compression_label(compression)}",
                            run_root=selector_dir.resolve(),
                        )
                    )

    # A test-time run records the physical selector chosen for prune_spanN.
    # Reuse it here instead of showing another recovery candidate at that budget.
    baseline = _selected_recovery_method(eval_root, budget)
    if baseline is None:
        # Compatibility fallback for incomplete/legacy plan bundles: a
        # test-time result still records which physical recovery selector was
        # chosen for prune_spanN.
        for selector, recovery in dict.fromkeys(selected):
            candidate = eval_root / "recovery" / selector / recovery
            if _looks_like_run_root(candidate):
                baseline = VisualizationMethod(
                    key=f"budget:none:{recovery}:{selector}",
                    label=f"Budget {budget} · No test-time",
                    run_root=candidate.resolve(),
                )
                break
    return baseline, compressed


def _discover_visualization_methods(
    output_root: Path,
    backbone: str,
    dataset: str,
    *,
    reference_output_root: Optional[Path] = None,
) -> List[VisualizationMethod]:
    eval_root = Path(output_root) / str(backbone) / str(dataset) / "eval"
    budget = VISUALIZATION_BUDGETS.get(str(backbone))
    if budget is None or not eval_root.is_dir():
        return []

    budget_baseline, budget_testtime = _selected_budget_runs(eval_root, budget)
    methods: List[VisualizationMethod] = []
    if budget_baseline is not None:
        methods.append(budget_baseline)
    replaceme = _selected_replaceme_method(eval_root, budget)
    if replaceme is not None:
        methods.append(replaceme)
    methods.extend(budget_testtime)

    full = eval_root / "upper"
    if _looks_like_run_root(full):
        asset_root: Optional[Path] = None
        if not _discover_scenes(full) and reference_output_root is not None:
            reference_full = (
                Path(reference_output_root) / str(backbone) / str(dataset) / "eval" / "upper"
            )
            if _discover_scenes(reference_full):
                asset_root = reference_full.resolve()
        methods.append(
            VisualizationMethod(
                "original:none",
                "Original · No test-time",
                full.resolve(),
                asset_root=asset_root,
            )
        )
    for candidate in sorted(eval_root.glob("upper_*")):
        if not _looks_like_run_root(candidate):
            continue
        compression = candidate.name.removeprefix("upper_")
        methods.append(
            VisualizationMethod(
                key=f"original:{compression}",
                label=f"Original + {_compression_label(compression)}",
                run_root=candidate.resolve(),
            )
        )
    return methods


@lru_cache(maxsize=256)
def _discover_scenes(run_root: Path) -> tuple[str, ...]:
    scenes: set[str] = set()
    if not run_root.is_dir():
        return ()
    for filename in ("plot.png", "viser_points.npz"):
        for asset in run_root.rglob(filename):
            if filename == "viser_points.npz":
                if asset.parent.name != "pred" or asset.parent.parent.name != "vis":
                    continue
                scene_dir = asset.parent.parent.parent
            else:
                scene_dir = asset.parent
            try:
                scenes.add(scene_dir.relative_to(run_root).as_posix())
            except ValueError:
                continue
    return tuple(sorted(scenes))


def _discover_datasets(output_root: Path) -> List[str]:
    names: set[str] = set()
    preferred = ["scannet_test50", "7scenes_test", "nuscenes_val", "waymo_validation"]
    for backbone in VISUALIZATION_BACKBONES:
        backbone_root = Path(output_root) / backbone
        if not backbone_root.is_dir():
            continue
        for dataset_dir in backbone_root.iterdir():
            # The comparison is meaningful only for final evaluation datasets
            # that contain test-time runs, or one of the canonical final-test
            # datasets with recovery outputs.  The latter keeps discovery
            # independent from deleted deployment-compression baselines without
            # admitting calibration datasets such as scannet_dev50.
            if dataset_dir.is_dir() and (
                (dataset_dir / "eval" / "testtime").is_dir()
                or (
                    dataset_dir.name in preferred
                    and (dataset_dir / "eval" / "recovery").is_dir()
                )
            ):
                names.add(dataset_dir.name)
    return [name for name in preferred if name in names] + sorted(names.difference(preferred))


def _methods_by_backbone(
    output_root: Path,
    dataset: str,
    *,
    reference_output_root: Optional[Path] = None,
) -> Dict[str, List[VisualizationMethod]]:
    return {
        backbone: _discover_visualization_methods(
            output_root,
            backbone,
            dataset,
            reference_output_root=reference_output_root,
        )
        for backbone in VISUALIZATION_BACKBONES
    }


def _scene_union(methods_by_backbone: Dict[str, List[VisualizationMethod]]) -> List[str]:
    scenes: set[str] = set()
    for methods in methods_by_backbone.values():
        for method in methods:
            scenes.update(_discover_scenes(_method_asset_root(method)))
    return sorted(scenes)


TRAJECTORY_METRICS = ("ape", "are", "rpe_trans", "rpe_rot")
TRAJECTORY_METRIC_LABELS = ("APE", "ARE", "RPEₜ", "RPEᵣ")
VISUALIZATION_METRICS = (
    ("ape", "APE"),
    ("are", "ARE"),
    ("rpe_trans", "RPEₜ"),
    ("rpe_rot", "RPEᵣ"),
    ("acc", "Acc"),
    ("comp", "Comp"),
    ("cd", "CD"),
)
_TRAJECTORY_RENDER_LOCK = threading.Lock()


@lru_cache(maxsize=256)
def _record_scene_metrics(run_root: Path) -> Dict[str, Dict[str, float]]:
    path = run_root / "eval_record_v3.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows = payload.get("scenes", {})
    if not isinstance(rows, dict):
        return {}
    result: Dict[str, Dict[str, float]] = {}
    for row in rows.values():
        if not isinstance(row, dict):
            continue
        scene_id = str(row.get("scene_id", "")).strip()
        if scene_id:
            result[scene_id] = row
    return result


def _scene_metric_row(
    method: VisualizationMethod, scene: str
) -> Optional[Dict[str, Any]]:
    row: Any = None
    path = method.run_root / str(scene) / "metrics.json"
    if path.is_file():
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            row = None
    if not isinstance(row, dict):
        row = _record_scene_metrics(method.run_root).get(str(scene))
    return row if isinstance(row, dict) else None


def _trajectory_values(method: VisualizationMethod, scene: str) -> Optional[tuple[float, ...]]:
    row = _scene_metric_row(method, scene)
    if row is None:
        return None
    try:
        values = tuple(float(row[key]) for key in TRAJECTORY_METRICS)
    except (KeyError, TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def _visualization_metric_values(
    method: VisualizationMethod, scene: str
) -> Optional[tuple[float, ...]]:
    row = _scene_metric_row(method, scene)
    if row is None:
        return None
    try:
        values = tuple(float(row[key]) for key, _label in VISUALIZATION_METRICS)
    except (KeyError, TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def _sort_scenes_by_replaceme_advantage(
    scenes: List[str], methods_by_backbone: Dict[str, List[VisualizationMethod]]
) -> List[str]:
    comparisons: List[tuple[VisualizationMethod, VisualizationMethod]] = []
    for methods in methods_by_backbone.values():
        replaceme = next(
            (method for method in methods if method.key.startswith("replaceme:")),
            None,
        )
        ours = next(
            (method for method in methods if method.key.startswith("budget:none:")),
            None,
        )
        if replaceme is not None and ours is not None:
            comparisons.append((replaceme, ours))

    def key(scene: str) -> tuple[float, str]:
        advantages = []
        for replaceme, ours in comparisons:
            replaceme_values = _visualization_metric_values(replaceme, scene)
            our_values = _visualization_metric_values(ours, scene)
            if replaceme_values is None or our_values is None:
                continue
            advantages.append(sum(replaceme_values) - sum(our_values))
        mean_advantage = (
            sum(advantages) / len(advantages) if advantages else -math.inf
        )
        return (-mean_advantage, scene)

    return sorted(scenes, key=key)


def _method_2d_group(method: VisualizationMethod) -> tuple[int, int, str]:
    """Return row, order, and concise publication label for a method."""
    if method.key.startswith("original:fastvggt"):
        return 1, 0, "FastVGGT"
    if method.key.startswith("original:quantvggt"):
        return 1, 1, "QuantVGGT"
    if method.key == "original:none":
        return 1, 3, "Original"
    if method.key.startswith("budget:none:"):
        return 1, 2, "Ours"
    if method.key.startswith("replaceme:"):
        return 2, 2, "ReplaceMe"
    if ":fastvggt" in method.key:
        return 2, 0, "Ours + FastVGGT"
    if ":quantvggt" in method.key:
        return 2, 1, "Ours + QuantVGGT"
    return 2, 99, method.label


def _visualization_metrics_table(
    methods: List[VisualizationMethod], scene: str
) -> str:
    ordered = [
        method
        for _group, method in sorted(
            ((_method_2d_group(method), method) for method in methods),
            key=lambda item: (item[0][0], item[0][1], item[1].label),
        )
    ]
    if not ordered:
        return "<div><strong>Metrics (7)</strong><p>Unavailable.</p></div>"
    headers = [html.escape(_method_2d_group(method)[2]) for method in ordered]
    rows = {method.key: _scene_metric_row(method, scene) for method in ordered}
    header_html = "".join(f"<th>{label}</th>" for label in headers)
    body = []
    for key, label in VISUALIZATION_METRICS:
        cells = []
        for method in ordered:
            row = rows[method.key]
            value = None if row is None else row.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = float("nan")
            cells.append(f"{number:.2f}" if math.isfinite(number) else "n/a")
        body.append(
            f"<tr><th>{html.escape(label)}</th>"
            + "".join(f"<td>{value}</td>" for value in cells)
            + "</tr>"
        )
    return (
        '<style>.eval-metrics-table{overflow-x:auto}.eval-metrics-table table{'
        'width:100%;border-collapse:collapse;margin-top:6px;font-size:12px;'
        'font-variant-numeric:tabular-nums}.eval-metrics-table th,'
        '.eval-metrics-table td{padding:4px 6px;border:1px solid #d8dee9;'
        'text-align:right;white-space:nowrap}.eval-metrics-table th:first-child{'
        'text-align:left}.eval-metrics-table thead th{background:#f6f7f9}</style>'
        '<div class="eval-metrics-table"><strong>Metrics (7)</strong><table>'
        '<thead><tr><th>Metric</th>'
        f"{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _redraw_trajectory_plot(method: VisualizationMethod, scene: str) -> Optional[Image.Image]:
    path = _method_asset_root(method) / str(scene) / "trajectory_plot_data.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path) as payload:
            reference = np.asarray(payload["reference_xyz"], dtype=np.float64)
            estimate = np.asarray(payload["estimate_xyz"], dtype=np.float64)
            error = np.asarray(payload["error"], dtype=np.float64).reshape(-1)
    except (OSError, KeyError, ValueError):
        return None
    n = min(len(reference), len(estimate), len(error))
    if n < 2:
        return None
    reference = reference[:n]
    estimate = estimate[:n]
    error = error[:n]
    values = _trajectory_values(method, scene)

    with _TRAJECTORY_RENDER_LOCK:
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=90)
        ax.plot(reference[:, 0], reference[:, 2], "--", color="gray", label="gt")
        points = estimate[:, (0, 2)]
        segments = np.stack([points[:-1], points[1:]], axis=1)
        finite_error = error[np.isfinite(error)]
        vmin = float(np.min(finite_error)) if finite_error.size else 0.0
        vmax = float(np.max(finite_error)) if finite_error.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-12
        colored = LineCollection(
            segments,
            cmap="jet",
            norm=Normalize(vmin=vmin, vmax=vmax),
            linewidths=2.0,
        )
        colored.set_array(error[1:])
        ax.add_collection(colored)
        fig.colorbar(colored, ax=ax)
        combined = np.concatenate([reference[:, (0, 2)], estimate[:, (0, 2)]], axis=0)
        low = np.nanmin(combined, axis=0)
        high = np.nanmax(combined, axis=0)
        center = (low + high) / 2.0
        radius = max(float(np.max(high - low)) * 0.58, 0.05)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        if values is not None:
            ax.set_title(f"ATE: {values[0]:.2f}, ARE: {values[1]:.2f}")
        ax.legend()
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=90)
        plt.close(fig)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
        image.load()
        buffer.close()
        return image


def _annotated_plot(method: VisualizationMethod, scene: str, title: str) -> Image.Image:
    plot = _method_asset_root(method) / str(scene) / "plot.png"
    redrawn = _redraw_trajectory_plot(method, scene)
    if redrawn is not None:
        body = redrawn
    elif plot.is_file():
        with Image.open(plot) as source:
            body = source.convert("RGB")
    else:
        body = Image.new("RGB", (900, 600), "white")
        missing_draw = ImageDraw.Draw(body)
        missing_draw.text((24, 24), "plot.png unavailable", fill=(150, 30, 30))

    banner_height = max(96, int(round(body.height * 0.22)))
    canvas = Image.new("RGB", (body.width, body.height + banner_height), "white")
    canvas.paste(body, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    font_size = max(14, min(24, body.width // 36))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size + 2)
    except OSError:
        font = title_font = ImageFont.load_default()
    values = _trajectory_values(method, scene)
    if values is None:
        metric_parts = [f"{label}: n/a" for label in TRAJECTORY_METRIC_LABELS]
    else:
        metric_parts = [
            f"{label}: {value:.2f}"
            for label, value in zip(TRAJECTORY_METRIC_LABELS, values)
        ]
    draw.text((12, 8), title, fill=(0, 0, 0), font=title_font)
    draw.text((12, 38), "     ".join(metric_parts[:2]), fill=(0, 0, 0), font=font)
    draw.text((12, 64), "     ".join(metric_parts[2:]), fill=(0, 0, 0), font=font)
    return canvas


def _gallery_for_scene(
    methods: List[VisualizationMethod], scene: str, *, row: Optional[int] = None
) -> List[tuple[Image.Image, str]]:
    grouped = sorted(
        (
            (_method_2d_group(method), method)
            for method in methods
            if method.show_in_2d
        ),
        key=lambda item: (item[0][0], item[0][1]),
    )
    items: List[tuple[Image.Image, str]] = []
    for (method_row, _order, title), method in grouped:
        if row is not None and method_row != row:
            continue
        values = _trajectory_values(method, scene)
        metric_caption = (
            " · ".join(
                f"{label}={value:.2f}"
                for label, value in zip(TRAJECTORY_METRIC_LABELS, values)
            )
            if values is not None
            else "trajectory metrics unavailable"
        )
        items.append((_annotated_plot(method, scene, title), f"{title} · {metric_caption}"))
    return items


def build_eval_tab(
    *,
    blocks: gr.Blocks,
    controller: Any,
    output_root: Path,
    eval_viser_controller_cls: Any,
    load_image_if_exists: Callable[[Path], Optional[Any]],
) -> Dict[str, Any]:
    del blocks, load_image_if_exists

    # Visualization compares the final runs in ``main_results``.  This is
    # intentionally independent from ``controller.v3_output_root``, which is
    # normally ``analysis_outputs_v3`` and supplies the analysis tabs.
    output_root = Path(output_root).resolve()
    reference_output_root = Path(
        getattr(controller, "v3_output_root", output_root)
    ).resolve()
    viser_host = getattr(controller, "viser_host", "127.0.0.1")
    viser_port = int(getattr(controller, "viser_port", 8090))
    datasets = _discover_datasets(output_root)
    requested_dataset = str(getattr(controller, "dataset", ""))
    dataset0 = requested_dataset if requested_dataset in datasets else (datasets[0] if datasets else "")
    methods0 = _methods_by_backbone(
        output_root,
        dataset0,
        reference_output_root=reference_output_root,
    )
    initial_backbone = next(
        (backbone for backbone in VISUALIZATION_BACKBONES if methods0.get(backbone)),
        VISUALIZATION_BACKBONES[0],
    )
    initial_2d_backbones = [initial_backbone]
    initial_2d_methods = {initial_backbone: methods0.get(initial_backbone, [])}
    scenes0 = _sort_scenes_by_replaceme_advantage(
        _scene_union(initial_2d_methods), initial_2d_methods
    )
    scene0 = scenes0[0] if scenes0 else None
    initial_methods = methods0.get(initial_backbone, [])
    initial_method = initial_methods[0].key if initial_methods else None
    initial_metrics = _visualization_metrics_table(initial_methods, scene0 or "")

    eval_ctrl = None
    try:
        if output_root.is_dir():
            eval_ctrl = eval_viser_controller_cls(
                eval_root=output_root, host=str(viser_host), port=viser_port
            )
    except Exception:
        eval_ctrl = None

    iframe_html = (
        f'<iframe src="http://{viser_host}:{viser_port}" '
        'style="width:100%; height:650px; border:0;"></iframe>'
    )

    with gr.Tab("Visualization") as tab_eval:
        gr.Markdown("## Visualization")
        gr.Markdown(
            "Compare each backbone at its selected budget (VGGT 10, Pi3 16, "
            "VGGT-Omega 18), with and without test-time compression, alongside "
            "the corresponding original-model results."
        )

        with gr.Row():
            eval_dataset_dd = gr.Dropdown(label="Dataset", choices=datasets, value=dataset0 or None)
            eval_scene_dd = gr.Dropdown(label="Scene", choices=scenes0, value=scene0)
            eval_2d_backbones = gr.CheckboxGroup(
                label="2D backbones",
                choices=[(backbone_display_name(bb), bb) for bb in VISUALIZATION_BACKBONES],
                value=initial_2d_backbones,
            )

        with gr.Tabs():
            with gr.Tab("2D comparison"):
                galleries: Dict[tuple[str, int], Any] = {}
                backbone_columns: Dict[str, Any] = {}
                for backbone in VISUALIZATION_BACKBONES:
                    budget = VISUALIZATION_BUDGETS[backbone]
                    methods = methods0.get(backbone, [])
                    selected = backbone in initial_2d_backbones
                    with gr.Column(visible=selected) as backbone_columns[backbone]:
                        gr.Markdown(f"### {backbone_display_name(backbone)} · budget {budget}")
                        first_row_count = sum(
                            method.show_in_2d
                            and _method_2d_group(method)[0] == 1
                            for method in methods
                        )
                        second_row_count = sum(
                            method.show_in_2d
                            and _method_2d_group(method)[0] == 2
                            for method in methods
                        )
                        galleries[(backbone, 1)] = gr.Gallery(
                            value=_gallery_for_scene(methods, scene0 or "", row=1) if selected else [],
                            label="FastVGGT / QuantVGGT / Ours / Original",
                            columns=max(1, first_row_count),
                            rows=1,
                            object_fit="contain",
                            height="auto",
                        )
                        galleries[(backbone, 2)] = gr.Gallery(
                            value=_gallery_for_scene(methods, scene0 or "", row=2) if selected else [],
                            label="Ours with test-time compression",
                            columns=max(1, second_row_count),
                            rows=1,
                            object_fit="contain",
                            height="auto",
                        )

            with gr.Tab("3D interactive") as tab_3d:
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["control-column"]):
                        eval_backbone = gr.Dropdown(
                            label="Backbone",
                            choices=[(backbone_display_name(bb), bb) for bb in VISUALIZATION_BACKBONES],
                            value=initial_backbone,
                        )
                        eval_method = gr.Radio(
                            label="Method",
                            choices=[(method.label, method.key) for method in initial_methods],
                            value=initial_method,
                        )
                        eval_metrics = gr.HTML(
                            value=initial_metrics,
                        )
                        eval_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=5):
                        eval_3d_html = gr.HTML(
                            value="Select this tab to start the 3D viewer."
                        )

        def _load_3d(dataset: str, scene: str, backbone: str, method_key: str) -> str:
            methods = _discover_visualization_methods(
                output_root,
                str(backbone),
                str(dataset),
                reference_output_root=reference_output_root,
            )
            method = next((item for item in methods if item.key == str(method_key)), None)
            if method is None:
                return "No visualization result for the selected method."
            if not scene:
                return "Select a scene."
            asset_root = _method_asset_root(method)
            if str(scene) not in _discover_scenes(asset_root):
                return f"{method.label}: scene {scene!r} is unavailable; scene selection was preserved."
            if eval_ctrl is None:
                return "Viser is unavailable."
            eval_ctrl.set_eval_root(asset_root)
            # Reusing the server and its handles replaces geometry without
            # resetting the connected client's current camera.
            return f"{method.label} | {eval_ctrl.load_scene(str(scene), show_gt=True)}"

        def _metrics_for(dataset: str, scene: str, backbone: str) -> str:
            methods = _discover_visualization_methods(
                output_root,
                str(backbone),
                str(dataset),
                reference_output_root=reference_output_root,
            )
            return _visualization_metrics_table(methods, str(scene or ""))

        def _refresh_dataset(
            dataset: str,
            scene: str,
            selected_backbones: List[str],
            backbone: str,
            method_key: str,
        ):
            methods_by_backbone = _methods_by_backbone(
                output_root,
                str(dataset),
                reference_output_root=reference_output_root,
            )
            selected_set = set(selected_backbones or [])
            selected_methods = {
                bb: methods
                for bb, methods in methods_by_backbone.items()
                if bb in selected_set
            }
            scenes = _sort_scenes_by_replaceme_advantage(
                _scene_union(selected_methods or methods_by_backbone),
                selected_methods or methods_by_backbone,
            )
            selected_scene = scenes[0] if scenes else ""
            selected_backbone = str(backbone)
            methods = methods_by_backbone.get(selected_backbone, [])
            selected_method = str(method_key or "")
            if not any(item.key == selected_method for item in methods):
                selected_method = methods[0].key if methods else ""
            gallery_updates = []
            for bb in VISUALIZATION_BACKBONES:
                bb_methods = methods_by_backbone.get(bb, [])
                for row in (1, 2):
                    count = sum(
                        method.show_in_2d
                        and _method_2d_group(method)[0] == row
                        for method in bb_methods
                    )
                    gallery_updates.append(
                        gr.update(
                            value=(
                                _gallery_for_scene(bb_methods, selected_scene, row=row)
                                if bb in selected_set
                                else []
                            ),
                            columns=max(1, count),
                        )
                    )
            return (
                gr.update(choices=scenes, value=selected_scene or None),
                *gallery_updates,
                gr.update(
                    choices=[(item.label, item.key) for item in methods],
                    value=selected_method or None,
                ),
                _load_3d(str(dataset), selected_scene, selected_backbone, selected_method),
                _metrics_for(str(dataset), selected_scene, selected_backbone),
            )

        def _refresh_scene(
            dataset: str,
            scene: str,
            selected_backbones: List[str],
            backbone: str,
            method_key: str,
        ):
            methods_by_backbone = _methods_by_backbone(
                output_root,
                str(dataset),
                reference_output_root=reference_output_root,
            )
            selected_set = set(selected_backbones or [])
            return (
                *[
                    gr.update(
                        value=(
                            _gallery_for_scene(
                                methods_by_backbone.get(bb, []), str(scene or ""), row=row
                            )
                            if bb in selected_set
                            else []
                        )
                    )
                    for bb in VISUALIZATION_BACKBONES
                    for row in (1, 2)
                ],
                _load_3d(str(dataset), str(scene or ""), str(backbone), str(method_key or "")),
                _metrics_for(str(dataset), str(scene or ""), str(backbone)),
            )

        def _refresh_2d_backbones(
            dataset: str,
            scene: str,
            selected_backbones: List[str],
            backbone: str,
        ):
            methods_by_backbone = _methods_by_backbone(
                output_root,
                str(dataset),
                reference_output_root=reference_output_root,
            )
            selected_set = set(selected_backbones or [])
            selected_methods = {
                bb: methods
                for bb, methods in methods_by_backbone.items()
                if bb in selected_set
            }
            scenes = _sort_scenes_by_replaceme_advantage(
                _scene_union(selected_methods or methods_by_backbone),
                selected_methods or methods_by_backbone,
            )
            selected_scene = str(scene or "")
            if selected_scene not in scenes:
                selected_scene = scenes[0] if scenes else ""
            column_updates = [
                gr.update(visible=bb in selected_set) for bb in VISUALIZATION_BACKBONES
            ]
            gallery_updates = []
            for bb in VISUALIZATION_BACKBONES:
                methods = methods_by_backbone.get(bb, [])
                for row in (1, 2):
                    gallery_updates.append(
                        gr.update(
                            value=(
                                _gallery_for_scene(methods, selected_scene, row=row)
                                if bb in selected_set
                                else []
                            )
                        )
                    )
            return (
                gr.update(choices=scenes, value=selected_scene or None),
                *column_updates,
                *gallery_updates,
                _metrics_for(str(dataset), selected_scene, str(backbone)),
            )

        def _refresh_backbone(dataset: str, scene: str, backbone: str, method_key: str):
            methods = _discover_visualization_methods(
                output_root,
                str(backbone),
                str(dataset),
                reference_output_root=reference_output_root,
            )
            selected_method = str(method_key or "")
            if not any(item.key == selected_method for item in methods):
                selected_method = methods[0].key if methods else ""
            return (
                gr.update(
                    choices=[(item.label, item.key) for item in methods],
                    value=selected_method or None,
                ),
                _load_3d(str(dataset), str(scene or ""), str(backbone), selected_method),
                _metrics_for(str(dataset), str(scene or ""), str(backbone)),
            )

        common_inputs = [
            eval_dataset_dd,
            eval_scene_dd,
            eval_2d_backbones,
            eval_backbone,
            eval_method,
        ]
        gallery_outputs = [
            galleries[(bb, row)]
            for bb in VISUALIZATION_BACKBONES
            for row in (1, 2)
        ]
        eval_dataset_dd.change(
            _refresh_dataset,
            inputs=common_inputs,
            outputs=[eval_scene_dd, *gallery_outputs, eval_method, eval_status, eval_metrics],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        eval_scene_dd.change(
            _refresh_scene,
            inputs=common_inputs,
            outputs=[*gallery_outputs, eval_status, eval_metrics],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        eval_2d_backbones.change(
            _refresh_2d_backbones,
            inputs=[eval_dataset_dd, eval_scene_dd, eval_2d_backbones, eval_backbone],
            outputs=[
                eval_scene_dd,
                *[backbone_columns[bb] for bb in VISUALIZATION_BACKBONES],
                *gallery_outputs,
                eval_metrics,
            ],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        eval_backbone.change(
            _refresh_backbone,
            inputs=[eval_dataset_dd, eval_scene_dd, eval_backbone, eval_method],
            outputs=[eval_method, eval_status, eval_metrics],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        # Method changes deliberately do not update the scene dropdown or iframe.
        eval_method.change(
            _load_3d,
            inputs=[eval_dataset_dd, eval_scene_dd, eval_backbone, eval_method],
            outputs=eval_status,
            show_progress="minimal",
            trigger_mode="always_last",
        )

        def _open_3d(dataset: str, scene: str, backbone: str, method_key: str):
            # Start/load Viser first, then create the iframe.  Creating the
            # iframe before the server exists leaves browsers on a permanent
            # connection-error page even after Viser starts.
            status = _load_3d(dataset, scene, backbone, method_key)
            metrics = _metrics_for(dataset, scene, backbone)
            return status, metrics, iframe_html

        tab_3d.select(
            _open_3d,
            inputs=[eval_dataset_dd, eval_scene_dd, eval_backbone, eval_method],
            outputs=[eval_status, eval_metrics, eval_3d_html],
            show_progress="minimal",
        )
        tab_eval.select(
            _refresh_scene,
            inputs=common_inputs,
            outputs=[*gallery_outputs, eval_status, eval_metrics],
            show_progress="minimal",
        )

    return {"tab": tab_eval}
