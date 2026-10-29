"""Export the seven results-panel tabs for a serverless GitHub Pages dashboard.

Scientific transformations stay in the local Gradio callbacks and renderers.
The browser receives final heatmap cells, plotted coordinates and table rows.
No model, dataset images, activation caches or point clouds are exported.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


class Assets:
    """Capture the actual inputs to heatmap drawing and outputs of curve drawing."""

    def __init__(self, root):
        self.root = root
        self.heatmaps = {}
        self.figures = {}
        self.tokens = {}

    def token(self, kind, key):
        index = len(self.tokens) + 1
        color = [index >> 16 & 255, index >> 8 & 255, index & 255]
        self.tokens[index] = {"kind": kind, "key": key}
        return np.broadcast_to(np.array(color, dtype=np.uint8), (16, 16, 3)).copy()

    def reference(self, image):
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return None
        rgb = arr[arr.shape[0] // 2, arr.shape[1] // 2, :3].astype(int)
        return self.tokens.get(int(rgb[0] * 65536 + rgb[1] * 256 + rgb[2]))

    def gallery(self, value):
        if isinstance(value, dict):
            value = value.get("value", [])
        output = []
        for entry in value or []:
            image, caption = entry if isinstance(entry, tuple) else (entry, "")
            ref = self.reference(image)
            if ref:
                output.append({**ref, "caption": str(caption)})
            elif image is not None:
                output.append({"kind": "empty", "caption": str(caption) or "No result for this selection."})
        return output

    def heatmap(self, packed, num_layers, title, **kwargs):
        from analysis.redundancy.metrics import upper_packed_to_symmetric_matrix
        grid = np.asarray(upper_packed_to_symmetric_matrix(packed, num_layers), dtype=np.float64)
        item = clean({"n": num_layers, "packed": grid[np.triu_indices(num_layers)], "title": title, **kwargs})
        key = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:20]
        self.heatmaps[key] = item
        return self.token("heatmap", key)

    def figure(self, fig, **_):
        if fig is None:
            return None
        axes = []
        names = getattr(fig, "_dashboard_methods", {})
        for ax in fig.axes:
            if not ax.axison:
                continue
            lines = []
            for line in ax.lines:
                x, y = line.get_data()
                if not len(x):
                    continue
                label = str(line.get_label())
                key = names.get(label, "")
                identifier = f"dashboard-line-{len(axes)}-{len(lines)}"
                line.set_gid(identifier)
                lines.append(clean({
                    "id": identifier, "label": label, "method": key,
                    "x": np.asarray(x), "y": np.asarray(y),
                    "color": to_hex(line.get_color()), "style": line.get_linestyle(),
                }))
            if lines:
                axes.append({"title": ax.get_title(), "ylabel": ax.get_ylabel(), "lines": lines})
        stream = io.StringIO()
        with matplotlib.rc_context({"svg.fonttype": "none", "svg.hashsalt": "dashboard"}):
            fig.savefig(stream, format="svg", bbox_inches="tight", metadata={"Date": None})
        svg = stream.getvalue()
        key = hashlib.sha256(svg.encode()).hexdigest()[:20]
        filename = f"plots/{key}.svg"
        (self.root / "plots").mkdir(exist_ok=True)
        (self.root / filename).write_text(svg)
        self.figures[key] = {"file": filename, "axes": axes}
        return self.token("figure", key)

    def patches(self, stack):
        from analysis import view
        from analysis.vis import pruning_tab, recovery_tab
        collected = {}
        for module, names in ((pruning_tab, ("collect_pruning_eval_payload",)),
                              (recovery_tab, ("collect_pruning_eval_payload", "collect_recovery_eval_payload"))):
            for name in names:
                original_collector = getattr(module, name)

                def collect(*args, _original=original_collector, **kwargs):
                    key = (_original, json.dumps([args, kwargs], default=str, sort_keys=True))
                    if key not in collected:
                        collected[key] = _original(*args, **kwargs)
                    # Renderers may transform their payloads in place.
                    return copy.deepcopy(collected[key])
                stack.enter_context(patch.object(module, name, collect))
        stack.enter_context(patch.object(view, "_heatmap_rgb_from_packed", self.heatmap))
        for module, name in ((pruning_tab, "_figure_to_rgb"), (recovery_tab, "_figure_to_rgb"),
                             (recovery_tab, "_compact_pruning_figure")):
            stack.enter_context(patch.object(module, name, self.figure))
        for module in (pruning_tab, recovery_tab):
            original = module.render_strategy_metric_figure

            def render(*args, _original=original, **kwargs):
                fig = _original(*args, **kwargs)
                if fig is not None:
                    fig._dashboard_methods = {
                        str(v.get("display_name", k)): k
                        for k, v in kwargs.get("payload", {}).get("strategies", {}).items()
                    }
                return fig
            stack.enter_context(patch.object(module, "render_strategy_metric_figure", render))


def export_heatmaps(root, assets, analysis_root, backbones):
    from analysis import view
    controller = view.PairAttentionController(analysis_root, "scannet_dev50", {}, "cpu")
    callbacks = view._build_ui(controller, export_only=True)
    order = [bb for bb in callbacks["backbones"] if bb in backbones]
    datasets = callbacks["single_datasets"]
    output = {"backbones": order, "single_datasets": datasets, "single_overview": [], "remaining": [], "similarity": []}
    for count in range(1, len(datasets) + 1):
        for subset in itertools.combinations(datasets, count):
            print("Single-interval overview", subset, flush=True)
            raw = callbacks["single_overview"](order, list(subset), split_pose_metrics=False)
            output["single_overview"].append({"datasets": subset, "rows": [assets.gallery(raw[callbacks["backbones"].index(bb)]) for bb in order], "status": raw[-1]})
    for ds in callbacks["datasets"]:
        if ds == "min":
            continue
        print("Remaining", ds, flush=True)
        raw = callbacks["remaining"](ds)
        output["remaining"].append({"dataset": ds, "rows": [assets.gallery(raw[callbacks["backbones"].index(bb)])[:2] for bb in order], "status": raw[-1]})
    for bb in order:
        for ds in callbacks["datasets"]:
            scoped = callbacks["heatmap_scope"](bb, ds)
            scenes = ["mean"]
            if not scenes:
                continue
            print("Similarity", bb, ds, len(scenes), "scenes", flush=True)
            rows = []
            for scene in scenes:
                raw = callbacks["similarity_bundle"](
                    scoped, scene, "absolute", 100, None, view.STAGE2_TOKEN_ROW_ORDER, 1, 0,
                    request_seq=None, force_gallery_refresh=True, show_column_titles=False,
                )
                regions = {}
                for start in range(0, 8 * len(callbacks["modes"]), 8):
                    combined = assets.gallery(raw[start + 2])
                    if not combined:
                        continue
                    for offset, region in enumerate(view.STAGE2_TOKEN_ROW_ORDER, 3):
                        metric_cells = assets.gallery(raw[start + offset])
                        if metric_cells:
                            regions[region] = [combined[0], *metric_cells]
                if regions:
                    rows.append({"scene": scene, "regions": regions, "status": str(raw[-7])})
            filename = f"similarity/{bb}_{ds}.json"
            write(root / filename, rows)
            output["similarity"].append({"backbone": bb, "dataset": ds, "scenes": [r["scene"] for r in rows], "file": filename})
    # Keep heatmap payloads next to their consumers, loading only selected cells.
    for key, item in assets.heatmaps.items():
        write(root / f"heatmaps/{key}.json", item)
    write(root / "heatmaps.json", output)
    return output


def export_pruning(root, assets, analysis_root, backbones):
    from analysis.vis import pruning_tab as p
    scopes = set()
    for bb in backbones:
        scopes.update(p.discover_pruning_plan_scopes(analysis_root, bb))
    if "mean" in scopes:
        scopes.remove("mean")
        scopes.update(p._AVERAGE_SCOPES)
    entries = []
    for bb, ds, merge, method in itertools.product(["mean", *backbones], sorted(scopes), (False, True), ("baseline", "similarity")):
        print("Pruning", bb, ds, "overview", merge, method, flush=True)
        gallery, status = p.render_pruning_plan_gallery(analysis_root, [bb], [ds], merge_replaceme=merge, view="overview", overview_method=method)
        entries.append({"backbone": bb, "dataset": ds, "detail": False, "merge": merge, "method": method, "cells": [c for c in assets.gallery(gallery) if c["kind"] == "figure"], "status": status})
    write(root / "pruning.json", {"scopes": sorted(scopes), "entries": entries})


def export_recovery(root, assets, analysis_root, main_root, backbones):
    from analysis.vis import recovery_tab as r, final_results_tab as f
    records = []
    methods_all = set()
    methods_by_view = {}
    for final in (False, True):
        source = main_root if final else analysis_root
        _, methods = r.discover_recovery_overview_options(source)
        methods_by_view["final" if final else "recovery"] = methods
        methods_all.update(methods)
        strategies = ["Retention_min"] if final else list(r._RECOVERY_OVERVIEW_PRUNE_STRATEGIES)
        groups = f.FINAL_OVERVIEW_DATASET_AVERAGE_GROUPS if final else r._OVERVIEW_DATASET_AVERAGE_GROUPS
        datasets = r._discover_recovery_generalization_datasets(source, dataset_average_groups=groups, allowed_datasets=f.FINAL_DATASETS if final else None)
        if final:
            datasets = [ds for ds in datasets if ds in (*f.FINAL_OVERVIEW_DATASETS, f.FINAL_OVERVIEW_AVERAGE)]
        for bb, ds, strategy in itertools.product(["mean", *backbones], datasets, strategies):
            print("Final" if final else "Recovery", "overview", bb, ds, strategy, flush=True)
            gallery, status = r.render_recovery_overview(
                source, [bb], methods, ds, prune_strategy=strategy, detail=False,
                dataset_average_groups=groups,
                comparison_dataset_average_groups=groups,
                calibration_dataset_average_groups=f.FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS if final else None,
                comparison_calibration_dataset_average_groups=f.FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS if final else None,
                comparison_output_root=analysis_root if final else None,
            )
            cells = assets.gallery(gallery)
            # Drop separate raster legend tiles; the browser filters SVG legends.
            cells = [c for c in cells if c["kind"] == "figure" and assets.figures[c["key"]]["axes"]]
            records.append({"final": final, "detail": False, "backbone": bb, "dataset": ds, "strategy": strategy, "cells": cells, "status": status})
            plt.close("all")
    write(root / "recovery.json", {"methods": sorted(methods_all), "methods_by_view": methods_by_view, "entries": records})


def export_tables(root, analysis_root, main_root):
    from analysis.vis import summary_tab as s, final_results_tab as f
    from analysis.vis.dataset_average_warning import GENERALIZATION_DATASET_AVERAGE_GROUPS
    for final, source in ((True, main_root),):
        groups = f.FINAL_TABLE_DATASET_AVERAGE_GROUPS if final else GENERALIZATION_DATASET_AVERAGE_GROUPS
        kwargs = dict(prune_strategies=("retention", "replaceme") if final else s.DEFAULT_SUMMARY_PRUNE_STRATEGIES,
                      include_compression=final, include_efficiency_metrics=final)
        header, live, _ = s._load_live_eval_rows(source, **kwargs)
        training = s._load_training_best_rows(output_root=source, metric_headers=header[3:])
        variants = {}
        for include_training in (False, True):
            rows = s._add_dataset_average_rows([*live, *(training if include_training else [])], header[3:], groups)
            rows = s._select_best_recovery_rows(rows, include_lower=not final, share_recovery_winner_across_variants=final)
            variants[str(include_training).lower()] = [
                {**asdict(row), "group": s._mode_group_for_row(row.mode), "method": s._recovery_method_for_row(row.mode),
                 "compression": s._compression_for_row(row.mode) or "none", "drop": s._span_value_for_row(row.mode),
                 "families": s._pruning_families_for_sources(row.pruning_sources), "structures": s._pruning_structures_for_sources(row.pruning_sources),
                 "numeric": {h: s._numeric_cell_value(row.metrics.get(h, "")) for h in header[3:]}}
                for row in rows
            ]
        write(root / ("final_table.json" if final else "recovery_table.json"), {
            "header": header, "rows": variants, "one_decimal_headers": sorted(s._ONE_DECIMAL_HEADERS),
        })


def finalize(output):
    """Keep only assets reachable from the current dashboard, then checksum them."""
    refs = {"heatmap": set(), "figure": set()}

    def walk(value):
        if isinstance(value, dict):
            if value.get("kind") in refs:
                refs[value["kind"]].add(value["key"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    sources = [p for p in output.glob("*.json") if p.name not in ("manifest.json", "figures.json")]
    sources.extend((output / "similarity").glob("*.json"))
    for path in sources:
        walk(json.loads(path.read_text()))
    for path in (output / "heatmaps").glob("*.json"):
        if path.stem not in refs["heatmap"]:
            path.unlink()
    raw_meta = json.loads((output / "figures.json").read_text())
    meta = {key: json.loads((output / value["metadata"]).read_text()) if "metadata" in value else value
            for key, value in raw_meta.items() if key in refs["figure"]}
    pointers = {}
    for key, value in meta.items():
        filename = f"plots/{key}.json"
        write(output / filename, value)
        pointers[key] = {"file": value["file"], "metadata": filename}
    write(output / "figures.json", pointers)
    svg_files = {item["file"] for item in meta.values()}
    for path in (output / "plots").glob("*.svg"):
        if path.relative_to(output).as_posix() not in svg_files:
            path.unlink()
    for path in (output / "plots").glob("*.json"):
        if path.stem not in meta:
            path.unlink()
    files = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    write(output / "manifest.json", {"schema_version": "dashboard_v2", "files": files, "total_bytes": sum(x["bytes"] for x in files)})
    print("Exported", len(files), "assets,", round(sum(x["bytes"] for x in files) / 1024**2, 2), "MiB", flush=True)


def export_dashboard(analysis_root, main_root, output, backbones, sections=None):
    from analysis.vis.display_names import backbone_display_name
    output.mkdir(parents=True, exist_ok=True)
    assets = Assets(output)
    # A partial refresh reuses assets from an earlier export in the same output.
    index_path = output / "index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    if (output / "figures.json").is_file():
        metadata = json.loads((output / "figures.json").read_text())
        assets.figures.update({key: json.loads((output / value["metadata"]).read_text()) if "metadata" in value else value for key, value in metadata.items()})
    index.pop("radars", None)
    index.pop("radar_status", None)
    sections = set(sections or ("heatmaps", "pruning", "recovery", "tables"))
    with ExitStack() as stack:
        assets.patches(stack)
        if "heatmaps" in sections:
            export_heatmaps(output, assets, analysis_root, backbones)
        if "pruning" in sections:
            export_pruning(output, assets, analysis_root, backbones)
        if "recovery" in sections:
            export_recovery(output, assets, analysis_root, main_root, backbones)
        if "tables" in sections:
            export_tables(output, analysis_root, main_root)
    write(output / "figures.json", assets.figures)
    palettes = {name: np.round(matplotlib.colormaps[name](np.arange(256))[:, :3] * 255).astype(int) for name in ("coolwarm", "coolwarm_r", "viridis", "magma", "magma_r")}
    index.update({"schema_version": "dashboard_v2", "exported_at": datetime.now(timezone.utc).isoformat(),
                  "backbones": backbones, "backbone_labels": {bb: backbone_display_name(bb) for bb in ["mean", *backbones]},
                  "palettes": palettes, "selection_policy": "Original dashboard callbacks; calibration selection, normalization and dataset averages are computed in Python."})
    write(index_path, index)
    finalize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis_outputs_v3"))
    parser.add_argument("--main-root", type=Path, default=Path("main_results"))
    parser.add_argument("--output", type=Path, default=Path("web/data"))
    parser.add_argument("--backbones", nargs="+", default=["vggt_omega", "pi3", "vggt"])
    parser.add_argument("--sections", nargs="+", choices=["heatmaps", "pruning", "recovery", "tables"])
    args = parser.parse_args()
    if args.sections:
        export_dashboard(args.analysis_root, args.main_root, args.output, args.backbones, args.sections)
    else:
        # Never leave a published tree containing a mixture of old and new data.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".dashboard-", dir=args.output.parent) as temporary:
            stage = Path(temporary) / "data"
            export_dashboard(args.analysis_root, args.main_root, stage, args.backbones)
            previous = Path(temporary) / "previous"
            if args.output.exists():
                args.output.rename(previous)
            try:
                stage.rename(args.output)
            except Exception:
                if previous.exists():
                    previous.rename(args.output)
                raise


if __name__ == "__main__":
    main()
