"""Export only the three selected calibrated checkpoints and their provenance."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from analysis.recovery.shared_results import best_recovery_result

OPERATING_POINTS = {"vggt": 10, "pi3": 16, "vggt_omega": 18}
CALIBRATION_DATASETS = ("scannet_dev50", "7scenes_train", "nuscenes_train", "waymo_training")
RECOVERY = "recovery_ghost_interval_sequential_autogroup"


def export_checkpoints(results_root: Path, output: Path, *, standalone: bool = False, device: str = "cpu") -> dict:
    # Resolve every candidate before creating any release files.
    selected = {}
    for backbone, drop in OPERATING_POINTS.items():
        best = best_recovery_result(
            results_root / backbone,
            strategies=["retention_single", "retention_double"],
            drop_count=drop,
            recovery_plan_name=RECOVERY,
            datasets=list(CALIBRATION_DATASETS),
        )
        selector = best["selector"]
        source = results_root / backbone / "recovery" / selector / RECOVERY / "checkpoints/latest.pt"
        if not source.is_file():
            raise FileNotFoundError(source)
        selected[backbone] = (source, best)
    entries = {}
    for backbone, (source, best) in selected.items():
        destination = output / backbone / ("model.pt" if standalone else "recovery.pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if standalone:
            import gc
            from lare.model import load_calibrated
            from analysis.recovery.train.checkpoint_v3 import atomic_save_checkpoint, load_train_checkpoint_payload

            calibrated = load_calibrated(backbone, checkpoint=source, device=device)
            payload = load_train_checkpoint_payload(source, map_location="cpu")
            payload["variant"] = "lare_standalone_v1"
            payload["model_state_dict"] = {key: value.detach().cpu() for key, value in calibrated.model.state_dict().items()}
            atomic_save_checkpoint(payload, destination)
            del payload, calibrated
            gc.collect()
        elif source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        with destination.open("rb") as handle:
            hasher = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
            digest = hasher.hexdigest()
        entries[backbone] = {
            "checkpoint": destination.relative_to(output).as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": digest,
            "drop_count": OPERATING_POINTS[backbone],
            "selector": best["selector"],
            "strategy": best["strategy"],
            "calibration_score": best["score"],
            "recovery_method": RECOVERY,
        }
    manifest = {
        "schema_version": "lare_calibrated_v1",
        "format": "standalone_model" if standalone else "base_weights_plus_recovery_delta",
        "calibration_datasets": list(CALIBRATION_DATASETS),
        "selection_policy": "calibration_raw_seven_metric_mean; double_improvement > 0.1",
        "models": entries,
    }
    name = "standalone_manifest.json" if standalone else "manifest.json"
    (output / name).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("main_results"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--standalone", action="store_true", help="Export complete pruned model weights; no base-weight download is needed at inference")
    parser.add_argument("--device", default="cpu", help="Device used to assemble standalone models")
    args = parser.parse_args()
    manifest = export_checkpoints(args.results_root, args.output, standalone=args.standalone, device=args.device)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
