from __future__ import annotations

import copy
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis.prune_plan import load_prune_plan_bundle
from analysis.prune_plan.eval_plots import (
    PRUNING_METRIC_GRID,
    PRUNING_MEAN_METRIC_GRID,
    collect_pruning_eval_payload,
    pruning_average_strategy_keys,
    render_strategy_metric_figure,
    strategy_is_single_interval,
)
from analysis.vis.dataset_average_warning import (
    DATASET_AVERAGE_GROUPS,
    format_incomplete_average_warning,
)
from analysis.vis.display_names import (
    backbone_display_name,
    backbone_publication_sort_key,
)


GalleryItems = List[Any]

_BACKBONE_ORDER = ("mapanything_v1", "mapanything", "vggt_omega", "pi3", "vggt")
_AVERAGE_SCOPES = ("training", "testing")
_MEAN_BACKBONE = "mean"
_DEFAULT_DROP_STEP = 2
_DETAIL_PLOT_GROUPS = (("ours", "ours"), ("others", "others"))
_OVERVIEW_PLOT_GROUPS = (
    ("single", "ours"),
    ("double", "ours"),
    ("others", "others"),
)
_OVERVIEW_SINGLE_MUTED_STYLE = {
    "linewidth": 1.6,
    "alpha": 0.55,
    "zorder": 1.5,
    "markeredgewidth": 1.5,
    "markersize": 49.0,
    "marker_alpha": 0.55,
    "marker_zorder": 1.7,
}
_POINT_METRIC_FIELDS = (
    "metrics_mean",
    "relative_to_upper",
    "normalized_relative_to_upper",
    "raw_metric",
    "normalized_metric",
)


def _backbone_sort_key(name: str) -> Tuple[int, str]:
    key = str(name)
    try:
        return (_BACKBONE_ORDER.index(key), key)
    except ValueError:
        return (len(_BACKBONE_ORDER), key)


def _scope_label(scope: str) -> str:
    if scope == "training":
        return "Training average"
    if scope == "testing":
        return "Testing average"
    return str(scope)


def _bundle_training_datasets(backbone_root: Path) -> List[str]:
    try:
        bundle = load_prune_plan_bundle(Path(backbone_root))
    except Exception:
        return []
    raw = bundle.get("dataset", "")
    if isinstance(raw, str):
        return [name.strip() for name in raw.split(",") if name.strip()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(name).strip() for name in raw if str(name).strip()]
    return []


def _result_datasets(backbone_root: Path) -> List[str]:
    return sorted(
        {
            path.parent.parent.name
            for path in Path(backbone_root).glob("*/eval/pruning")
            if path.is_dir() and any(path.glob("*/summary.json"))
        }
    )


def discover_pruning_plan_backbones(output_root: Path) -> List[str]:
    root = Path(output_root)
    if not root.is_dir():
        return []
    found = [
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and (path / "prune_plan_all.json").is_file()
        and bool(_result_datasets(path))
    ]
    return sorted(set(found), key=_backbone_sort_key)


def discover_pruning_plan_scopes(
    output_root: Path,
    backbone: str,
) -> List[str]:
    datasets = _result_datasets(Path(output_root) / str(backbone))
    return [*_AVERAGE_SCOPES, *datasets]


def resolve_pruning_scope_datasets(
    output_root: Path,
    backbone: str,
    scope: str,
) -> List[str]:
    backbone_root = Path(output_root) / str(backbone)
    available = _result_datasets(backbone_root)
    training = set(_bundle_training_datasets(backbone_root))
    if scope == "training":
        return [dataset for dataset in available if dataset in training]
    if scope == "testing":
        expected = _expected_pruning_scope_datasets(backbone_root, scope)
        return [dataset for dataset in expected if dataset in set(available)]
    return [scope] if scope in available else []


def _expected_pruning_scope_datasets(
    backbone_root: Path,
    scope: str,
) -> List[str]:
    """Return the complete dataset set an average is expected to contain."""
    training = _bundle_training_datasets(backbone_root)
    if scope == "training":
        if set(training).intersection(DATASET_AVERAGE_GROUPS["training average"]):
            return list(DATASET_AVERAGE_GROUPS["training average"])
        return training
    if scope == "testing":
        # Production plans use the canonical two training datasets and four
        # held-out datasets. Keep synthetic/custom plans useful by falling back
        # to their discovered non-training datasets.
        known_names = {
            *DATASET_AVERAGE_GROUPS["training average"],
            *DATASET_AVERAGE_GROUPS["testing average"],
        }
        available = _result_datasets(backbone_root)
        if set(training).intersection(known_names) or set(available).intersection(known_names):
            return list(DATASET_AVERAGE_GROUPS["testing average"])
        return [dataset for dataset in available if dataset not in set(training)]
    return [scope]


def _figure_to_rgb(fig: Any) -> np.ndarray:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    with Image.open(buffer) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _scope_plot_payload(payload: Dict[str, Any], scope: str) -> Dict[str, Any]:
    if scope in _AVERAGE_SCOPES:
        average = payload.get("dataset_average", {})
        return average if isinstance(average, dict) else {}
    datasets = payload.get("datasets", {})
    if not isinstance(datasets, dict):
        return {}
    dataset_payload = datasets.get(scope, {})
    return dataset_payload if isinstance(dataset_payload, dict) else {}


def _minimum_metric_map(*maps: Any) -> Dict[str, float]:
    values: Dict[str, List[float]] = {}
    for metric_map in maps:
        if not isinstance(metric_map, dict):
            continue
        for key, value in metric_map.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.setdefault(str(key), []).append(float(value))
    return {key: min(items) for key, items in values.items() if items}


def _merge_replaceme_strategies(plot_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Replace Single/Double ReplaceMe curves with their pointwise minimum."""
    strategies = plot_payload.get("strategies", {})
    if not isinstance(strategies, dict):
        return plot_payload
    source_strategies = [
        strategies.get("replaceme_single"),
        strategies.get("replaceme_double"),
    ]
    source_strategies = [item for item in source_strategies if isinstance(item, dict)]
    if not source_strategies:
        return plot_payload

    drop_keys = {
        str(drop_key)
        for strategy in source_strategies
        for drop_key in (strategy.get("points", {}) or {})
        if str(drop_key).isdigit()
    }
    merged_points: Dict[str, Any] = {}
    for drop_key in sorted(drop_keys, key=int):
        candidates = [
            points[drop_key]
            for strategy in source_strategies
            for points in [strategy.get("points", {})]
            if isinstance(points, dict)
            and isinstance(points.get(drop_key), dict)
            and bool(points[drop_key].get("available", False))
        ]
        if not candidates:
            continue
        merged_point = copy.deepcopy(candidates[0])
        merged_point["selector"] = "pointwise_min(replaceme_single,replaceme_double)"
        for field in _POINT_METRIC_FIELDS:
            merged_point[field] = _minimum_metric_map(
                *(candidate.get(field, {}) for candidate in candidates)
            )
        merged_points[drop_key] = merged_point

    merged_payload = dict(plot_payload)
    merged_strategies = dict(strategies)
    merged_strategies.pop("replaceme_single", None)
    merged_strategies.pop("replaceme_double", None)
    merged_strategies["replaceme"] = {
        "display_name": "ReplaceMe (min Single/Double)",
        "points": merged_points,
    }
    merged_payload["strategies"] = merged_strategies
    return merged_payload


def _average_metric_maps(maps: Sequence[Any]) -> Dict[str, float]:
    valid = [item for item in maps if isinstance(item, dict)]
    if not valid:
        return {}
    common_keys = set(valid[0])
    for item in valid[1:]:
        common_keys.intersection_update(item)
    averaged: Dict[str, float] = {}
    for key in common_keys:
        values = [item.get(key) for item in valid]
        if all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values
        ):
            averaged[str(key)] = float(np.mean(values))
    return averaged


def _mean_backbone_plot_payloads(
    plot_payloads: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Average points shared by every backbone payload."""
    if not plot_payloads:
        return {}
    strategy_maps = [payload.get("strategies", {}) for payload in plot_payloads]
    if not all(isinstance(item, dict) for item in strategy_maps):
        return {}
    strategy_keys = set(strategy_maps[0])
    for strategies in strategy_maps[1:]:
        strategy_keys.intersection_update(strategies)

    mean_strategies: Dict[str, Any] = {}
    for strategy_key in sorted(strategy_keys):
        strategy_payloads = [strategies[strategy_key] for strategies in strategy_maps]
        point_maps = [item.get("points", {}) for item in strategy_payloads]
        if not all(isinstance(item, dict) for item in point_maps):
            continue
        drop_keys = set(point_maps[0])
        for points in point_maps[1:]:
            drop_keys.intersection_update(points)
        mean_points: Dict[str, Any] = {}
        for drop_key in sorted(drop_keys, key=lambda value: int(value)):
            points = [point_map[drop_key] for point_map in point_maps]
            if not all(
                isinstance(point, dict) and bool(point.get("available", False))
                for point in points
            ):
                continue
            mean_point: Dict[str, Any] = {
                "available": True,
                "drop_count": int(drop_key),
                "num_backbones": len(points),
            }
            for field in _POINT_METRIC_FIELDS:
                mean_point[field] = _average_metric_maps(
                    [point.get(field, {}) for point in points]
                )
            mean_points[str(drop_key)] = mean_point
        mean_strategies[str(strategy_key)] = {
            "display_name": str(
                strategy_payloads[0].get("display_name", strategy_key)
            ),
            "points": mean_points,
        }
    return {
        "upper_average": _average_metric_maps(
            [payload.get("upper_average", {}) for payload in plot_payloads]
        ),
        "normalized_upper_average": _average_metric_maps(
            [
                payload.get("normalized_upper_average", {})
                for payload in plot_payloads
            ]
        ),
        "strategies": mean_strategies,
        "num_backbones": len(plot_payloads),
    }


def render_pruning_plan_gallery(
    output_root: Path,
    backbone: str | Sequence[str],
    selected_scopes: Sequence[str] | None,
    merge_replaceme: bool = True,
    view: str = "overview",
    normalize_individual_metrics: bool = True,
) -> Tuple[GalleryItems, str]:
    """Reload current summaries and render the pruning views in memory."""
    view_key = str(view).strip().lower()
    if view_key not in {"overview", "detail"}:
        raise ValueError(f"Unsupported pruning-plan view: {view!r}")
    detail_metrics = [metric for row in PRUNING_METRIC_GRID for metric in row]
    metric_keys = detail_metrics if view_key == "detail" else ["gt_mean"]
    plot_groups = (
        _DETAIL_PLOT_GROUPS if view_key == "detail" else _OVERVIEW_PLOT_GROUPS
    )
    expected_plots = len(plot_groups) * len(metric_keys)
    scopes = list(dict.fromkeys(str(scope) for scope in (selected_scopes or [])))
    selected_backbones = (
        [str(backbone)]
        if isinstance(backbone, str) and backbone
        else list(dict.fromkeys(str(item) for item in (backbone or []) if str(item)))
    )
    if not selected_backbones:
        return [], "Select at least one backbone."
    selected_backbones = sorted(
        selected_backbones,
        key=lambda name: (
            (-1, "")
            if name == _MEAN_BACKBONE
            else backbone_publication_sort_key(name)
        ),
    )
    if not scopes:
        return [], "Select at least one dataset or average scope."

    available_backbones = discover_pruning_plan_backbones(Path(output_root))
    selected_backbones = [
        item
        for item in selected_backbones
        if item == _MEAN_BACKBONE or item in available_backbones
    ]
    gallery: GalleryItems = []
    warnings: List[str] = []
    messages: List[str] = []
    for backbone_index, selected_backbone in enumerate(selected_backbones):
        contributors = (
            available_backbones
            if selected_backbone == _MEAN_BACKBONE
            else [selected_backbone]
        )
        for scope_index, scope in enumerate(scopes):
            try:
                contributor_plots: List[Dict[str, Any]] = []
                dataset_labels: List[str] = []
                for contributor in contributors:
                    backbone_root = Path(output_root) / contributor
                    datasets = resolve_pruning_scope_datasets(
                        Path(output_root), contributor, scope
                    )
                    if scope in _AVERAGE_SCOPES:
                        expected = _expected_pruning_scope_datasets(
                            backbone_root, scope
                        )
                        warning = format_incomplete_average_warning(
                            owner=contributor,
                            scope=_scope_label(scope),
                            expected=expected,
                            available=datasets,
                        )
                        if warning:
                            warnings.append(warning)
                    if not datasets:
                        continue
                    payload = collect_pruning_eval_payload(
                        experiment_root=backbone_root,
                        prune_plan_root=backbone_root,
                        datasets=datasets,
                        drop_step=_DEFAULT_DROP_STEP,
                    )
                    contributor_plots.append(_scope_plot_payload(payload, scope))
                    dataset_labels.extend(f"{contributor}:{item}" for item in datasets)
                if len(contributor_plots) != len(contributors):
                    messages.append(
                        f"{selected_backbone}/{scope}: missing one or more backbone summaries"
                    )
                    continue
                plot_payload = (
                    _mean_backbone_plot_payloads(contributor_plots)
                    if selected_backbone == _MEAN_BACKBONE
                    else contributor_plots[0]
                )
                if merge_replaceme:
                    plot_payload = _merge_replaceme_strategies(plot_payload)
                strategies = plot_payload.get("strategies", {})
                strategy_keys = tuple(strategies) if isinstance(strategies, dict) else ()
                rendered = 0
                scope_start = len(gallery)
                for group_index, (plot_group, strategy_group) in enumerate(plot_groups):
                    selected = pruning_average_strategy_keys(
                        strategy_keys, strategy_group
                    )
                    style_overrides: Dict[str, Dict[str, Any]] = {}
                    if view_key == "overview" and plot_group in {"single", "double"}:
                        single_keys = tuple(
                            key
                            for key in selected
                            if strategy_is_single_interval(key)
                        )
                        if plot_group == "single":
                            selected = single_keys
                            style_overrides = {
                                key: {"linestyle": "-", "dashes": None}
                                for key in single_keys
                            }
                        else:
                            style_overrides = {
                                key: dict(_OVERVIEW_SINGLE_MUTED_STYLE)
                                for key in single_keys
                            }
                    if plot_group == "others":
                        selected = tuple(
                            key
                            for key in selected
                            if key not in {"ours_single", "gt_single"}
                        )
                    for metric_index, metric_key in enumerate(metric_keys):
                        metric_grid = (
                            PRUNING_MEAN_METRIC_GRID
                            if view_key == "overview"
                            else ((metric_key,),)
                        )
                        fig = render_strategy_metric_figure(
                            payload=plot_payload,
                            title=(
                                f"{_scope_label(scope)} · "
                                f"{plot_group.title()}"
                            ),
                            metric_grid=metric_grid,
                            strategy_keys=selected,
                            compact_pruning_legend=True,
                            legend_fontsize=12,
                            normalize_metrics=(
                                True
                                if metric_key == "gt_mean"
                                else bool(normalize_individual_metrics)
                            ),
                            strategy_style_overrides=style_overrides,
                            show_figure_title=(
                                backbone_index == 0
                                and scope_index == 0
                                if view_key == "overview"
                                else metric_index == 0
                            ),
                            show_metric_titles=(
                                backbone_index == 0 and scope_index == 0
                            ),
                            shared_xlabel=(
                                "Dropped units"
                                if backbone_index == len(selected_backbones) - 1
                                and scope_index == len(scopes) - 1
                                and (
                                    view_key == "overview"
                                    or (
                                        group_index == len(plot_groups) - 1
                                        and metric_index >= len(metric_keys) - 4
                                    )
                                )
                                else None
                            ),
                            x_tick_step=2,
                            tick_label_fontsize=13,
                            axis_label_fontsize=15,
                            legend_markerscale=1.5,
                            legend_handlelength=2.8,
                        )
                        if fig is None:
                            continue
                        try:
                            gallery.append(_figure_to_rgb(fig))
                            rendered += 1
                        finally:
                            plt.close(fig)
                if rendered:
                    separator_index = scope_start + (rendered - 1) // 2
                    gallery[separator_index] = (
                        gallery[separator_index],
                        f"{backbone_display_name(selected_backbone)} · {_scope_label(scope)}",
                    )
                messages.append(
                    f"{selected_backbone}/{scope}: datasets={','.join(dataset_labels)} "
                    f"| view={view_key} | plots={rendered}/{expected_plots} "
                    f"| individual_normalized={bool(normalize_individual_metrics)} "
                    "| mean_normalized=True "
                    f"| merge_replaceme={bool(merge_replaceme)} "
                    "| normalization_scope=single_intervals_drop_lte_24"
                )
            except Exception as exc:
                messages.append(
                    f"{selected_backbone}/{scope}: {type(exc).__name__}: {exc}"
                )
    return gallery, "\n".join(
        [*warnings, f"backbones={','.join(selected_backbones)}", *messages]
    )


def build_pruning_plan_tab(
    *,
    blocks: gr.Blocks,
    output_root: Path,
    tab_id: str = "pruning_plan",
) -> Dict[str, Any]:
    root = Path(output_root)
    backbones = discover_pruning_plan_backbones(root)
    initial_backbone = backbones[0] if backbones else ""
    initial_scopes = discover_pruning_plan_scopes(root, initial_backbone)
    initial_selection = (
        ["training"] if "training" in initial_scopes else initial_scopes[:1]
    )

    def _selected_real_backbones(selected: Sequence[str] | str | None) -> List[str]:
        values = [selected] if isinstance(selected, str) else list(selected or [])
        if _MEAN_BACKBONE in values:
            return list(backbones)
        return [str(value) for value in values if value in backbones]

    def _scope_choices(selected: Sequence[str] | str | None) -> List[Tuple[str, str]]:
        selected_scopes = {
            scope
            for backbone in _selected_real_backbones(selected)
            for scope in discover_pruning_plan_scopes(root, backbone)
        }
        ordered = [
            *[scope for scope in _AVERAGE_SCOPES if scope in selected_scopes],
            *sorted(selected_scopes.difference(_AVERAGE_SCOPES)),
        ]
        return [
            (_scope_label(scope), scope)
            for scope in ordered
        ]

    def _render(
        backbone: Sequence[str],
        selected_scopes: Sequence[str],
        merge_replaceme: bool,
        *,
        view: str,
        normalize_individual_metrics: bool = True,
    ) -> Tuple[GalleryItems, str]:
        return render_pruning_plan_gallery(
            root,
            backbone,
            selected_scopes,
            merge_replaceme=merge_replaceme,
            view=view,
            normalize_individual_metrics=normalize_individual_metrics,
        )

    def _selected_scopes_after_backbone_change(
        backbone: Sequence[str],
        selected_scopes: Sequence[str],
    ) -> List[str]:
        scopes = [value for _, value in _scope_choices(backbone)]
        retained = [scope for scope in (selected_scopes or []) if scope in scopes]
        return retained or (["training"] if "training" in scopes else scopes[:1])

    def _backbone_changed(
        backbone: Sequence[str],
        selected_scopes: Sequence[str],
        merge_replaceme: bool,
        *,
        view: str,
        normalize_individual_metrics: bool = True,
    ) -> Tuple[Any, GalleryItems, str]:
        selected = _selected_scopes_after_backbone_change(
            backbone, selected_scopes
        )
        gallery, status = _render(
            backbone,
            selected,
            merge_replaceme,
            view=view,
            normalize_individual_metrics=normalize_individual_metrics,
        )
        return (
            gr.update(choices=_scope_choices(backbone), value=selected),
            gallery,
            status,
        )

    def _refresh(
        backbone: Sequence[str],
        selected_scopes: Sequence[str],
        merge_replaceme: bool,
        *,
        view: str,
        normalize_individual_metrics: bool = True,
    ) -> Tuple[Any, Any, GalleryItems, str]:
        latest_backbones = discover_pruning_plan_backbones(root)
        backbone_value = [
            value
            for value in (backbone or [])
            if value == _MEAN_BACKBONE or value in latest_backbones
        ]
        backbone_value = backbone_value or latest_backbones[:1]
        scopes = [value for _, value in _scope_choices(backbone_value)]
        selected = [scope for scope in (selected_scopes or []) if scope in scopes]
        selected = selected or (["training"] if "training" in scopes else scopes[:1])
        gallery, status = _render(
            backbone_value,
            selected,
            merge_replaceme,
            view=view,
            normalize_individual_metrics=normalize_individual_metrics,
        )
        return (
            gr.update(
                choices=[
                    (backbone_display_name(item), item)
                    for item in [_MEAN_BACKBONE, *latest_backbones]
                ],
                value=backbone_value,
            ),
            gr.update(choices=_scope_choices(backbone_value), value=selected),
            gallery,
            status,
        )

    def _controls(*, detail: bool) -> Dict[str, Any]:
        view = "detail" if detail else "overview"
        title = "Pruning Plan Detail" if detail else "Pruning Plan Overview"
        tab_suffix = "detail" if detail else "overview"
        with gr.Tab(title, id=f"{tab_id}_{tab_suffix}") as tab:
            gr.Markdown(f"## {title}")
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["control-column"]):
                    backbone_dd = gr.CheckboxGroup(
                        label="Backbones",
                        choices=[
                            (backbone_display_name(item), item)
                            for item in [_MEAN_BACKBONE, *backbones]
                        ],
                        value=[_MEAN_BACKBONE, *backbones],
                    )
                    scope_cb = gr.CheckboxGroup(
                        label="Datasets / averages (multi-select)",
                        choices=_scope_choices(backbones),
                        value=initial_selection,
                        type="value",
                        interactive=True,
                        info="Check one or more individual datasets, Training, or Testing.",
                    )
                    normalize_metrics = (
                        gr.Checkbox(
                            label="Normalize individual metrics to [0,1]",
                            value=True,
                            info="The 7-metric mean is always normalized per metric.",
                        )
                        if detail
                        else None
                    )
                    merge_replaceme = gr.Checkbox(
                        label="Merge ReplaceMe Single/Double (pointwise min)",
                        value=True,
                    )
                    refresh_btn = gr.Button("Reload summaries", variant="primary")
                    status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=5,
                    )
                with gr.Column(scale=5):
                    gallery = gr.Gallery(
                        label=f"Live pruning-plan {view} plots",
                        show_label=False,
                        columns=4 if detail else 3,
                        height="auto",
                        object_fit="contain",
                        elem_classes=[
                            "results-gallery",
                            "pruning-plan-gallery",
                            (
                                "pruning-plan-detail-gallery"
                                if detail
                                else "pruning-plan-overview-gallery"
                            ),
                        ],
                    )

            if detail:
                assert normalize_metrics is not None

                def redraw(backbone, scopes, normalize, merge):
                    return _render(
                        backbone,
                        scopes,
                        merge,
                        view=view,
                        normalize_individual_metrics=bool(normalize),
                    )

                def backbone_changed(backbone, scopes, normalize, merge):
                    return _backbone_changed(
                        backbone,
                        scopes,
                        merge,
                        view=view,
                        normalize_individual_metrics=bool(normalize),
                    )

                def refresh(backbone, scopes, normalize, merge):
                    return _refresh(
                        backbone,
                        scopes,
                        merge,
                        view=view,
                        normalize_individual_metrics=bool(normalize),
                    )

                inputs = [backbone_dd, scope_cb, normalize_metrics, merge_replaceme]
                change_triggers = [
                    scope_cb.change,
                    normalize_metrics.change,
                    merge_replaceme.change,
                ]
            else:
                def redraw(backbone, scopes, merge):
                    return _render(backbone, scopes, merge, view=view)

                def backbone_changed(backbone, scopes, merge):
                    return _backbone_changed(backbone, scopes, merge, view=view)

                def refresh(backbone, scopes, merge):
                    return _refresh(backbone, scopes, merge, view=view)

                inputs = [backbone_dd, scope_cb, merge_replaceme]
                change_triggers = [scope_cb.change, merge_replaceme.change]

            for trigger in change_triggers:
                trigger(
                    redraw,
                    inputs=inputs,
                    outputs=[gallery, status],
                    show_progress="minimal",
                    trigger_mode="always_last",
                )
            backbone_dd.change(
                backbone_changed,
                inputs=inputs,
                outputs=[scope_cb, gallery, status],
                show_progress="minimal",
                trigger_mode="always_last",
            )
            refresh_btn.click(
                refresh,
                inputs=inputs,
                outputs=[backbone_dd, scope_cb, gallery, status],
                show_progress="minimal",
                trigger_mode="always_last",
            )
            blocks.load(
                redraw,
                inputs=inputs,
                outputs=[gallery, status],
                show_progress="minimal",
            )

        controls = {
            "tab": tab,
            "backbone_dd": backbone_dd,
            "scope_cb": scope_cb,
            "merge_replaceme": merge_replaceme,
            "gallery": gallery,
            "status": status,
            "refresh_btn": refresh_btn,
        }
        if normalize_metrics is not None:
            controls["normalize_metrics"] = normalize_metrics
        return controls

    overview = _controls(detail=False)
    detail = _controls(detail=True)

    return {
        **overview,
        "overview_tab": overview["tab"],
        "overview": overview,
        "detail_tab": detail["tab"],
        "detail": detail,
    }
