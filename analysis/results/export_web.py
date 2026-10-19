"""Export the data used by the static website, without archiving experiment trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis.metric_labels import SEVEN_METRIC_KEYS, SEVEN_METRIC_LABELS
from analysis.prune_plan.eval_plots import collect_pruning_eval_payload, render_strategy_metric_figure
from analysis.triangular_heatmap import draw_symmetric_heatmap_panel
from analysis.vis.dataset_average_warning import GENERALIZATION_DATASET_AVERAGE_GROUPS
from analysis.vis.final_results_tab import (
    FINAL_TABLE_DATASET_AVERAGE_GROUPS,
    FINAL_DATASET_AVERAGE_GROUPS,
    FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS,
)
from analysis.vis.recovery_tab import render_recovery_overview
from analysis.vis.summary_tab import (
    collect_summary_rows,
    _compression_for_row,
    _mode_group_for_row,
    _recovery_method_for_row,
    _span_value_for_row,
)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items() if not str(k).endswith(("_path", "_root"))}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.name
    return value


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def export_tables(analysis_root: Path, main_root: Path, data: Path) -> dict:
    views = {}
    for name, root, groups, strategies in (
        ("final", main_root, FINAL_TABLE_DATASET_AVERAGE_GROUPS, ("retention", "replaceme")),
        ("analysis", analysis_root, GENERALIZATION_DATASET_AVERAGE_GROUPS, ("CKA", "retention")),
    ):
        final = name == "final"
        header, rows = collect_summary_rows(
            root, prune_strategies=strategies, dataset_average_groups=groups,
            include_compression=final, include_efficiency_metrics=final,
            separate_pruning_structures=final, winner_only_recovery=final,
        )
        if not rows:
            raise ValueError(f"No {name} result rows in {root}")
        by_backbone = {}
        for row in rows:
            raw = row.raw_metrics or {}
            item = {
                "dataset": row.dataset, "backbone": row.backbone, "mode": row.mode,
                "mode_group": _mode_group_for_row(row.mode),
                "recovery_method": _recovery_method_for_row(row.mode),
                "compression": _compression_for_row(row.mode) or "none",
                "drop_count": _span_value_for_row(row.mode),
                "pruning_sources": row.pruning_sources,
                "display_metrics": row.metrics,
                "raw_metrics": {k: raw.get(SEVEN_METRIC_LABELS[k]) for k in SEVEN_METRIC_KEYS},
            }
            by_backbone.setdefault(row.backbone, []).append(item)
        files = {}
        for backbone, selected in by_backbone.items():
            filename = f"tables/{name}_{backbone}.json"
            write_json(data / filename, {"columns": header[3:], "rows": selected})
            files[backbone] = filename
        views[name] = {"files": files, "dataset_groups": groups, "row_count": len(rows)}
    return views


def export_figures(analysis_root: Path, main_root: Path, data: Path, backbones: list[str]) -> list[dict]:
    entries = []
    figures = data / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for backbone in backbones:
        root = analysis_root / backbone
        payload = collect_pruning_eval_payload(experiment_root=root, prune_plan_root=root)
        write_json(data / f"curves/pruning_{backbone}.json", payload)
        for dataset in ("scannet_dev50", "nuscenes_train"):
            for region in ("all", "special", "patch"):
                source = root / dataset / "redundancy" / f"redundancy_graph_{region}.npz"
                if not source.is_file():
                    continue
                # Research NPZ files store unit_ids as object strings. These are
                # locally generated artifacts, never browser/user uploads.
                with np.load(source, allow_pickle=True) as z:
                    matrices = {k: z[f"{k}_mean"] for k in ("cka", "cosine", "cca", "inv_mse") if f"{k}_mean" in z}
                    write_json(data / f"matrices/{backbone}_{dataset}_{region}.json", {"unit_ids": z["unit_ids"], "metrics": matrices})
                    for metric, values in matrices.items():
                        fig, ax = plt.subplots(figsize=(5.2, 4.5))
                        draw_symmetric_heatmap_panel(ax, values, title=f"{backbone} · {metric.upper()}", colorbar_label=f"Raw {metric} similarity")
                        filename = f"figures/similarity_{backbone}_{dataset}_{region}_{metric}.png"
                        fig.savefig(data / filename, dpi=135, bbox_inches="tight")
                        plt.close(fig)
                        entries.append({"kind": "similarity", "backbone": backbone, "dataset": dataset, "region": region, "metric": metric, "file": filename})
        # Render from the exact Python plotting paths used by the research UI.
        for dataset in ("scannet_dev50", "nuscenes_train", "scannet_test50", "nuscenes_val"):
            if dataset not in payload.get("datasets", {}):
                continue
            fig = render_strategy_metric_figure(
                payload=payload["datasets"][dataset],
                title=f"{backbone} · {dataset}",
                metric_grid=(SEVEN_METRIC_KEYS[:4], SEVEN_METRIC_KEYS[4:]),
                strategy_keys=("retention_single", "retention_double", "CKA_single", "CKA_double"),
                normalize_metrics=True,
            )
            if fig is None:
                continue
            filename = f"figures/pruning_{backbone}_{dataset}.png"
            fig.savefig(data / filename, dpi=120, bbox_inches="tight")
            plt.close(fig)
            entries.append({"kind": "pruning", "backbone": backbone, "dataset": dataset, "file": filename})
        for dataset in FINAL_DATASET_AVERAGE_GROUPS:
            images, status = render_recovery_overview(
                main_root, [backbone], ["recovery_ghost_interval_sequential_autogroup"], dataset,
                prune_strategy="Retention_min", comparison_output_root=analysis_root,
                dataset_average_groups=FINAL_DATASET_AVERAGE_GROUPS,
                comparison_dataset_average_groups=FINAL_DATASET_AVERAGE_GROUPS,
                calibration_dataset_average_groups=FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS,
                comparison_calibration_dataset_average_groups=FINAL_CALIBRATION_DATASET_AVERAGE_GROUPS,
            )
            if not images:
                raise ValueError(f"Cannot export recovery figure for {backbone}/{dataset}: {status}")
            for index, (pixels, caption) in enumerate(images):
                filename = f"figures/recovery_{backbone}_{dataset.replace(' ', '_')}_{index}.png"
                Image.fromarray(np.asarray(pixels)).save(data / filename)
                entries.append({"kind": "recovery", "backbone": backbone, "dataset": dataset, "file": filename, "caption": caption, "status": status})
    return entries


def export_scenes(main_root: Path, checkpoints: Path, data: Path, max_points: int) -> list[dict]:
    manifest = json.loads((checkpoints / "manifest.json").read_text())
    scenes = []
    for backbone, info in manifest["models"].items():
        dataset = "scannet_test50"
        eval_root = main_root / backbone / dataset / "eval"
        recovered = eval_root / "recovery" / info["selector"] / info["recovery_method"]
        candidates = sorted(recovered.glob("*/vis/pred/viser_points.npz"))
        for candidate in candidates:
            scene_id = candidate.parents[2].name
            original = eval_root / "upper" / scene_id / "vis/pred/viser_points.npz"
            if not original.is_file():
                continue
            variants = []
            for label, source in (("Original", original), ("LaRe", candidate)):
                with np.load(source, allow_pickle=False) as z:
                    points = z["points"].reshape(-1, 3)
                    colors = z["colors"].reshape(-1, 3)
                    finite = np.isfinite(points).all(axis=1)
                    valid = np.flatnonzero(finite)
                    if "confidence" in z:
                        confidence = z["confidence"].reshape(-1)
                        valid = valid[np.isfinite(confidence[valid]) & (confidence[valid] >= np.nanquantile(confidence[valid], 0.5))]
                    if len(valid) > max_points:
                        valid = np.random.default_rng(33).choice(valid, max_points, replace=False)
                    cloud = np.column_stack((points[valid], colors[valid] / 255.0)).astype("<f4")
                filename = f"scenes/{backbone}_{label.lower()}.bin"
                target = data / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                cloud.tofile(target)
                variants.append({"label": label, "file": filename, "points": len(cloud), "bytes": target.stat().st_size})
            scenes.append({"backbone": backbone, "dataset": dataset, "scene_id": scene_id, "variants": variants, "sampling": "finite points; upper confidence half; seeded sample", "max_points": max_points})
            break
    return scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis_outputs_v3"))
    parser.add_argument("--main-root", type=Path, default=Path("main_results"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("web/data"))
    parser.add_argument("--max-points", type=int, default=30000)
    args = parser.parse_args()
    if args.max_points <= 0:
        parser.error("--max-points must be positive")
    views = export_tables(args.analysis_root, args.main_root, args.output)
    backbones = list(views["final"]["files"])
    figures = export_figures(args.analysis_root, args.main_root, args.output, backbones)
    scenes = export_scenes(args.main_root, args.checkpoints, args.output, args.max_points)
    models = json.loads((args.checkpoints / "manifest.json").read_text())
    write_json(args.output / "index.json", {
        "schema_version": "lare_web_v1", "views": views, "figures": figures,
        "scenes": scenes, "models": models,
        "metric_labels": SEVEN_METRIC_LABELS,
        "selection_policy": "Calibration raw seven-error mean; Double must improve by >0.1; compression follows uncompressed winner.",
    })
    files = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({"file": path.relative_to(args.output).as_posix(), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(args.output / "manifest.json", {"schema_version": 1, "files": files, "total_bytes": sum(item["bytes"] for item in files)})
    print(f"Exported {len(files)} web assets ({sum(item['bytes'] for item in files) / 1024**2:.2f} MiB) to {args.output}")


if __name__ == "__main__":
    main()
