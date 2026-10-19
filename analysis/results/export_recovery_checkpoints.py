"""Stage recovery checkpoints referenced by current public pruning/recovery plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from analysis.prune_plan import (
    bundle_plan_records,
    bundle_strategy_records,
    load_prune_plan_bundle,
    physical_pruning_selector_from_record,
)
from analysis.prune_plan.recovery_plan import list_recovery_plan_names


def export_recovery_checkpoints(roots: list[Path], output: Path) -> dict:
    labels = [root.name for root in roots]
    if len(set(labels)) != len(labels):
        raise ValueError("Source roots must have distinct directory names")
    files = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        if output.resolve() == root.resolve() or root.resolve() in output.resolve().parents:
            raise ValueError("Export destination must not be inside an experiment result tree")
        for plan_path in sorted(root.glob("*/prune_plan_all.json")):
            backbone_root = plan_path.parent
            bundle = load_prune_plan_bundle(backbone_root)
            records = bundle_plan_records(bundle)
            selectors = set()
            for strategy in bundle_strategy_records(bundle).values():
                for alias in strategy.get("points", {}).values():
                    record = records.get(str(alias))
                    if record is not None:
                        selectors.add(physical_pruning_selector_from_record(record))
            methods = list_recovery_plan_names(backbone_root)
            for selector in sorted(selectors):
                for method in methods:
                    source = backbone_root / "recovery" / selector / method / "checkpoints/latest.pt"
                    if not source.is_file():
                        continue
                    relative = Path(root.name) / source.relative_to(root)
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    hasher = hashlib.sha256()
                    with target.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            hasher.update(block)
                    files.append({"file": relative.as_posix(), "bytes": target.stat().st_size, "sha256": hasher.hexdigest()})
            for name in ("prune_plan_all.json", "recovery_plan_all.json"):
                source = backbone_root / name
                if source.is_file():
                    destination = output / root.name / backbone_root.name / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "lare_recovery_release_v1", "scope": "existing checkpoints referenced by current public plan strategies and recovery methods", "files": files, "total_bytes": sum(item["bytes"] for item in files)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, default=[Path("analysis_outputs_v3"), Path("main_results")])
    parser.add_argument("--output", type=Path, default=Path("release_artifacts/recovery_checkpoints"))
    args = parser.parse_args()
    manifest = export_recovery_checkpoints(args.roots, args.output)
    print(f"Staged {len(manifest['files'])} recovery checkpoint files ({manifest['total_bytes'] / 1024**3:.2f} GiB) in {args.output}; no archive or upload was created")


if __name__ == "__main__":
    main()
