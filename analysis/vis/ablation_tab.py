from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr
import matplotlib.pyplot as plt

from analysis.prune_plan.eval_plots import (
    PRUNING_MEAN_METRIC_GRID,
    collect_pruning_eval_payload,
    render_strategy_metric_figure,
)
from analysis.vis.dataset_average_warning import (
    format_incomplete_average_warning,
)
from analysis.vis.display_names import (
    backbone_display_name,
    backbone_publication_sort_key,
)
from analysis.vis.pruning_tab import (
    PRUNING_PLAN_OVERVIEW_LEGEND_FONT_SIZE,
    _DEFAULT_DROP_STEP,
    _OTHER_METHOD_LINEWIDTH,
    _OVERVIEW_SINGLE_STYLE,
    _TRAINING_LEGEND_CAPTION,
    _compose_pruning_overview,
    _expected_pruning_scope_datasets,
    _figure_to_rgb,
    _pruning_overview_figure_kwargs,
    _scope_label,
    _scope_plot_payload,
    _style_pruning_overview_figure,
    resolve_pruning_scope_datasets,
)
from analysis.vis.shared_legend import (
    LegendEntry,
    collect_and_remove_legends,
    render_shared_legend_banner,
)


TOKEN_ABLATION_SCOPE = "training"
TOKEN_ABLATION_Y_AXIS_UPPER = 0.5
MAIN_PRUNE_PLAN_FILENAME = "prune_plan_all.json"
TOKEN_ABLATION_VARIANTS = (
    (
        "all",
        "all",
        "prune_plan_all_token_all.json",
        "#1f77b4",
    ),
    (
        "special",
        "special",
        "prune_plan_all_token_special.json",
        "#9467bd",
    ),
    (
        "patch",
        "patch",
        "prune_plan_all_token_patch.json",
        "#ff7f0e",
    ),
    (
        "special_patch_mean",
        "special_patch_mean",
        "prune_plan_all_token_special_patch_mean.json",
        "#d62728",
    ),
)

_TOKEN_VARIANT_ORDER = tuple(
    variant for variant, _label, _filename, _color in TOKEN_ABLATION_VARIANTS
)
_RETENTION_METHOD = "retention"
_METHOD_ORDER = (_RETENTION_METHOD, *_TOKEN_VARIANT_ORDER)
_TOKEN_METHOD_LABELS = {
    _RETENTION_METHOD: "retention",
    **{
        f"token_{variant}": label
        for variant, label, _filename, _color in TOKEN_ABLATION_VARIANTS
    },
    "token_special_patch_mean": "s-p",
}
_TOKEN_METHOD_COLORS = {
    _RETENTION_METHOD: "#111111",
    **{
        f"token_{variant}": color
        for variant, _label, _filename, color in TOKEN_ABLATION_VARIANTS
    },
}


def _load_bundle_file(
    path: Path,
    *,
    required_strategies: Sequence[str] = ("CKA_single", "CKA_double"),
) -> Dict[str, Any]:
    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Missing token-ablation prune plan: {bundle_path}")
    raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid token-ablation prune plan: {bundle_path}")
    strategies = raw.get("strategies", {})
    if not isinstance(strategies, dict):
        raise ValueError(
            f"Invalid strategies in token-ablation prune plan: {bundle_path}"
        )
    missing = [
        strategy
        for strategy in required_strategies
        if strategy not in strategies
    ]
    if missing:
        raise ValueError(
            f"Missing {missing} in token-ablation prune plan: {bundle_path}"
        )
    return raw


def discover_token_ablation_backbones(output_root: Path) -> List[str]:
    root = Path(output_root)
    if not root.is_dir():
        return []
    required_files = (
        MAIN_PRUNE_PLAN_FILENAME,
        *(
            filename
            for _variant, _label, filename, _color in TOKEN_ABLATION_VARIANTS
        ),
    )
    return sorted(
        (
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and all((path / filename).is_file() for filename in required_files)
        ),
        key=backbone_publication_sort_key,
    )


def load_token_ablation_bundles(
    output_root: Path,
    backbone: str,
    variants: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    backbone_root = Path(output_root) / str(backbone)
    selected = set(variants) if variants is not None else set(_TOKEN_VARIANT_ORDER)
    bundles = {
        variant: _load_bundle_file(backbone_root / filename)
        for variant, _label, filename, _color in TOKEN_ABLATION_VARIANTS
        if variant in selected
    }
    if _RETENTION_METHOD in selected:
        bundles[_RETENTION_METHOD] = _load_bundle_file(
            backbone_root / MAIN_PRUNE_PLAN_FILENAME,
            required_strategies=("retention_single", "retention_double"),
        )
    return bundles


def _combine_scope_payloads(
    *,
    backbone_root: Path,
    bundles: Dict[str, Dict[str, Any]],
    datasets: Sequence[str],
    scope: str,
    variants: Sequence[str],
) -> Dict[str, Any]:
    combined: Dict[str, Any] = {}
    combined_strategies: Dict[str, Any] = {}
    for variant, _label, _filename, _color in TOKEN_ABLATION_VARIANTS:
        if variant not in variants:
            continue
        bundle = bundles[variant]
        bundle_strategies = bundle.get("strategies", {})
        selected_strategies = {
            key: bundle_strategies[key]
            for key in ("CKA_single", "CKA_double")
        }
        payload = collect_pruning_eval_payload(
            experiment_root=backbone_root,
            prune_plan_root=backbone_root,
            bundle=bundle,
            strategies=selected_strategies,
            datasets=datasets,
            drop_step=_DEFAULT_DROP_STEP,
        )
        scoped = _scope_plot_payload(payload, scope)
        if not combined:
            combined = copy.deepcopy(scoped)
            combined["strategies"] = combined_strategies
        scoped_strategies = scoped.get("strategies", {})
        if not isinstance(scoped_strategies, dict):
            continue
        for structure, source_key in (
            ("single", "CKA_single"),
            ("double", "CKA_double"),
        ):
            source = scoped_strategies.get(source_key)
            if not isinstance(source, dict):
                continue
            renamed = copy.deepcopy(source)
            renamed["display_name"] = f"{variant} · {structure.title()}"
            combined_strategies[f"token_{variant}_{structure}"] = renamed

    if _RETENTION_METHOD in variants:
        bundle = bundles[_RETENTION_METHOD]
        bundle_strategies = bundle.get("strategies", {})
        selected_strategies = {
            key: bundle_strategies[key]
            for key in ("retention_single", "retention_double")
        }
        payload = collect_pruning_eval_payload(
            experiment_root=backbone_root,
            prune_plan_root=backbone_root,
            bundle=bundle,
            strategies=selected_strategies,
            datasets=datasets,
            drop_step=_DEFAULT_DROP_STEP,
        )
        scoped = _scope_plot_payload(payload, scope)
        if not combined:
            combined = copy.deepcopy(scoped)
            combined["strategies"] = combined_strategies
        scoped_strategies = scoped.get("strategies", {})
        if isinstance(scoped_strategies, dict):
            for key in ("retention_single", "retention_double"):
                source = scoped_strategies.get(key)
                if isinstance(source, dict):
                    combined_strategies[key] = copy.deepcopy(source)
    return combined


def _token_strategy_styles() -> Dict[str, Dict[str, Any]]:
    styles: Dict[str, Dict[str, Any]] = {}
    for family, color in _TOKEN_METHOD_COLORS.items():
        styles[f"{family}_single"] = {
            **_OVERVIEW_SINGLE_STYLE,
            "color": color,
        }
        styles[f"{family}_double"] = {
            "color": color,
            "linewidth": _OTHER_METHOD_LINEWIDTH,
        }
    return styles


def render_token_ablation_gallery(
    output_root: Path,
    selected_variants: Sequence[str] | None = None,
) -> Tuple[List[Any], str]:
    root = Path(output_root)
    requested_variants = (
        _METHOD_ORDER
        if selected_variants is None
        else selected_variants
    )
    requested_set = {str(variant) for variant in requested_variants}
    variants = [
        method for method in _METHOD_ORDER if method in requested_set
    ]
    if selected_variants is not None and not variants:
        return [], "Select at least one comparison method."
    backbones = discover_token_ablation_backbones(root)
    if not backbones:
        return [], "No backbones with complete token-ablation plans were found."

    styles = _token_strategy_styles()
    gallery: List[Any] = []
    legends: List[LegendEntry] = []
    warnings: List[str] = []
    messages: List[str] = []
    for backbone_index, backbone in enumerate(backbones):
        backbone_root = root / backbone
        try:
            bundles = load_token_ablation_bundles(root, backbone, variants)
            datasets = resolve_pruning_scope_datasets(
                root, backbone, TOKEN_ABLATION_SCOPE
            )
            warning = format_incomplete_average_warning(
                owner=backbone,
                scope=_scope_label(TOKEN_ABLATION_SCOPE),
                expected=_expected_pruning_scope_datasets(
                    backbone_root, TOKEN_ABLATION_SCOPE
                ),
                available=datasets,
            )
            if warning:
                warnings.append(warning)
            if not datasets:
                messages.append(f"{backbone}: no completed training datasets")
                continue

            plot_payload = _combine_scope_payloads(
                backbone_root=backbone_root,
                bundles=bundles,
                datasets=datasets,
                scope=TOKEN_ABLATION_SCOPE,
                variants=variants,
            )
            strategy_keys = tuple(
                (
                    f"{method}_{structure}"
                    if method == _RETENTION_METHOD
                    else f"token_{method}_{structure}"
                )
                for method in variants
                for structure in ("single", "double")
            )
            compact_method_order = [
                method
                if method == _RETENTION_METHOD
                else f"token_{method}"
                for method in variants
            ]
            fig = render_strategy_metric_figure(
                payload=plot_payload,
                title=_scope_label(TOKEN_ABLATION_SCOPE),
                metric_grid=PRUNING_MEAN_METRIC_GRID,
                strategy_keys=strategy_keys,
                compact_pruning_legend=True,
                compact_method_labels=_TOKEN_METHOD_LABELS,
                compact_method_order=compact_method_order,
                combine_compact_legend=True,
                normalize_metrics=True,
                strategy_style_overrides=styles,
                default_linewidth=_OTHER_METHOD_LINEWIDTH,
                upper_linewidth=_OTHER_METHOD_LINEWIDTH,
                draw_solid_before_dashed=True,
                dodge_structure_markers=True,
                fixed_y_axis_upper=TOKEN_ABLATION_Y_AXIS_UPPER,
                **_pruning_overview_figure_kwargs(),
            )
            if fig is None:
                messages.append(f"{backbone}: no complete ablation points")
                continue
            collect_and_remove_legends(fig, legends)
            _style_pruning_overview_figure(
                fig,
                backbone_index=backbone_index,
            )
            try:
                image = _figure_to_rgb(fig, compact=True)
            finally:
                plt.close(fig)
            caption = (
                f"{backbone_display_name(backbone)} · "
                f"{_scope_label(TOKEN_ABLATION_SCOPE)} · training"
            )
            gallery.append((image, caption))
            available_counts = {
                key: sum(
                    bool(point.get("available", False))
                    for point in strategy.get("points", {}).values()
                    if isinstance(point, dict)
                )
                for key, strategy in plot_payload.get("strategies", {}).items()
                if isinstance(strategy, dict)
            }
            messages.append(
                f"{backbone}: datasets={','.join(datasets)} | "
                f"available_points={available_counts}"
            )
        except Exception as exc:
            messages.append(f"{backbone}: {type(exc).__name__}: {exc}")

    legend = render_shared_legend_banner(
        legends,
        font_size_pixels=PRUNING_PLAN_OVERVIEW_LEGEND_FONT_SIZE,
        max_columns=1,
        trailing_labels=("Single", "Multi"),
        frameon=False,
    )
    if legend is not None:
        gallery.append((
            legend,
            f"{_TRAINING_LEGEND_CAPTION}|entries={len(legends)}",
        ))

    return gallery, "\n".join(
        [
            *warnings,
            f"backbones={','.join(backbones)}",
            "scope=training",
            "methods=" + ",".join(variants),
            *messages,
        ]
    )


def build_ablation_tab(
    *,
    blocks: gr.Blocks,
    output_root: Path,
    tab_id: str = "token_ablation_overview",
) -> Dict[str, Any]:
    root = Path(output_root)

    method_choices = [
        ("retention", _RETENTION_METHOD),
        *[
            (label, variant)
            for variant, label, _filename, _color in TOKEN_ABLATION_VARIANTS
        ],
    ]

    def redraw(
        selected_variants: Sequence[str],
        hide_dataset_names: bool,
    ) -> Tuple[Any, str]:
        images, status_value = render_token_ablation_gallery(
            root, selected_variants
        )
        image = (
            _compose_pruning_overview(
                images,
                hide_dataset_names=bool(hide_dataset_names),
            )
            if images
            else None
        )
        return image, status_value

    with gr.Tab("Ablation Token", id=tab_id) as tab:
        gr.Markdown(
            "## Ablation Token Overview\n"
            "All-backbone retention/CKA comparison on training datasets: "
            "main-plan `retention`, plus token variants `all`, "
            "`special`, `patch`, and `special_patch_mean`."
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["control-column"]):
                method_cb = gr.CheckboxGroup(
                    label="Comparison methods (multi-select)",
                    choices=method_choices,
                    value=list(_TOKEN_VARIANT_ORDER),
                    type="value",
                    interactive=True,
                )
                hide_dataset_names = gr.Checkbox(
                    label="Hide dataset names",
                    value=True,
                )
                refresh_btn = gr.Button("Reload summaries", variant="primary")
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=7,
                )
            with gr.Column(scale=5):
                gallery = gr.Image(
                    label="Live token-ablation overview plots",
                    show_label=False,
                    type="numpy",
                    interactive=False,
                    elem_classes=["results-media", "composite-plot-image"],
                )

        method_cb.change(
            redraw,
            inputs=[method_cb, hide_dataset_names],
            outputs=[gallery, status],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        hide_dataset_names.change(
            redraw,
            inputs=[method_cb, hide_dataset_names],
            outputs=[gallery, status],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        refresh_btn.click(
            redraw,
            inputs=[method_cb, hide_dataset_names],
            outputs=[gallery, status],
            show_progress="minimal",
            trigger_mode="always_last",
        )
        blocks.load(
            redraw,
            inputs=[method_cb, hide_dataset_names],
            outputs=[gallery, status],
            show_progress="minimal",
        )

    return {
        "tab": tab,
        "method_cb": method_cb,
        "hide_dataset_names": hide_dataset_names,
        "gallery": gallery,
        "status": status,
        "refresh_btn": refresh_btn,
    }
