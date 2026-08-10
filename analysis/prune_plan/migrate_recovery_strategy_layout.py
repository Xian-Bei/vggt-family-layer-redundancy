"""Migrate recovery artifacts into the shared selector-first layout."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from analysis.recovery.shared_results import canonical_selector


@dataclass(frozen=True)
class Migration:
    source: Path
    destination: Path


@dataclass(frozen=True)
class SymlinkRepair:
    path: Path
    target: str


def _linked_recovery_method(path: Path) -> str | None:
    if not path.is_symlink():
        return None
    try:
        parts = Path(os.readlink(path)).parts
    except OSError:
        return None
    return next((part for part in parts if part.startswith("recovery_")), None)


def _shared_symlink_repairs(output_root: Path) -> list[SymlinkRepair]:
    repairs: list[SymlinkRepair] = []
    roots = [
        recovery_root
        for backbone_root in output_root.iterdir()
        if backbone_root.is_dir()
        for recovery_root in [
            backbone_root / "recovery",
            *backbone_root.glob("*/eval/recovery"),
        ]
        if recovery_root.is_dir()
    ]
    for recovery_root in roots:
        for selector_root in recovery_root.iterdir():
            if not selector_root.is_dir() or selector_root.name == "_plots":
                continue
            for method_path in selector_root.iterdir():
                linked_method = _linked_recovery_method(method_path)
                if linked_method is None:
                    continue
                expected = method_path.parent / linked_method
                try:
                    already_correct = method_path.resolve(strict=False) == expected.resolve(strict=False)
                except OSError:
                    already_correct = False
                if expected.exists() and not already_correct:
                    repairs.append(SymlinkRepair(method_path, linked_method))
    return repairs


def _method_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("recovery_")
    )


def _build_migrations(
    output_root: Path,
) -> tuple[list[Migration], list[Path], list[Path], list[SymlinkRepair]]:
    moves: list[Migration] = []
    derived_files: list[Path] = []
    cleanup_dirs: list[Path] = []
    for backbone_root in sorted(path for path in output_root.iterdir() if path.is_dir()):
        if not (backbone_root / "prune_plan_all.json").is_file():
            continue

        # eval/recovery/<strategy>/<method>/<selector> ->
        # eval/recovery/<selector>/<method>
        for recovery_root in sorted(backbone_root.glob("*/eval/recovery")):
            if not recovery_root.is_dir():
                continue
            for strategy_root in sorted(path for path in recovery_root.iterdir() if path.is_dir()):
                strategy_methods = [
                    method_root
                    for method_root in _method_dirs(strategy_root)
                    if not (method_root / "summary.json").is_file()
                    and any(
                        (selector_root / "summary.json").is_file()
                        for selector_root in method_root.iterdir()
                        if selector_root.is_dir()
                    )
                ]
                if not strategy_methods:
                    continue
                for method_root in strategy_methods:
                    for selector_root in sorted(path for path in method_root.iterdir() if path.is_dir()):
                        selector = canonical_selector(backbone_root, selector_root.name)
                        moves.append(
                            Migration(
                                selector_root,
                                recovery_root / selector / method_root.name,
                            )
                        )
                    derived_files.extend(
                        path
                        for path in method_root.iterdir()
                        if path.is_file() and path.name.endswith("_index_v3.json")
                    )
                    cleanup_dirs.append(method_root)
                for plot in sorted(strategy_root.glob("*.png")):
                    moves.append(
                        Migration(plot, recovery_root / "_plots" / strategy_root.name / plot.name)
                    )
                cleanup_dirs.append(strategy_root)

        # recovery/<strategy>/<selector>/<method> -> recovery/<selector>/<method>
        recovery_root = backbone_root / "recovery"
        if recovery_root.is_dir():
            for strategy_root in sorted(path for path in recovery_root.iterdir() if path.is_dir()):
                selector_roots = [
                    path
                    for path in strategy_root.iterdir()
                    if path.is_dir() and _method_dirs(path)
                ]
                for selector_root in selector_roots:
                    selector = canonical_selector(backbone_root, selector_root.name)
                    for method_root in _method_dirs(selector_root):
                        moves.append(
                            Migration(
                                method_root,
                                recovery_root / selector / method_root.name,
                            )
                        )
                    cleanup_dirs.append(selector_root)
                summary = strategy_root / "summary_comparison.md"
                if summary.is_file():
                    moves.append(
                        Migration(
                            summary,
                            backbone_root / f"summary_comparison_{strategy_root.name}.md",
                        )
                    )
                if selector_roots or summary.is_file():
                    cleanup_dirs.append(strategy_root)
                if strategy_root.name != "_plots":
                    for plot in sorted(strategy_root.glob("*.png")):
                        moves.append(
                            Migration(
                                plot,
                                recovery_root / "_plots" / strategy_root.name / plot.name,
                            )
                        )
                        cleanup_dirs.append(strategy_root)

    destinations: set[Path] = set()
    for migration in moves:
        if migration.destination in destinations:
            raise RuntimeError(f"duplicate migration destination: {migration.destination}")
        destinations.add(migration.destination)
        if migration.destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing destination: {migration.destination}"
            )
    return moves, derived_files, cleanup_dirs, _shared_symlink_repairs(output_root)


def migrate(output_root: Path, *, apply: bool) -> int:
    moves, derived_files, cleanup_dirs, repairs = _build_migrations(Path(output_root))
    for migration in moves:
        print(f"{'MOVE' if apply else 'WOULD MOVE'} {migration.source} -> {migration.destination}")
    for path in derived_files:
        print(f"{'REMOVE' if apply else 'WOULD REMOVE'} derived index {path}")
    for repair in repairs:
        print(f"{'RELINK' if apply else 'WOULD RELINK'} {repair.path} -> {repair.target}")
    if not apply:
        print(
            f"Dry run: {len(moves)} directories/files and "
            f"{len(derived_files)} derived indexes are ready to migrate; "
            f"{len(repairs)} symlinks need repair."
        )
        return len(moves) + len(repairs)

    for migration in moves:
        linked_method = _linked_recovery_method(migration.source)
        migration.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(migration.source), str(migration.destination))
        if linked_method is not None:
            migration.destination.unlink()
            migration.destination.symlink_to(linked_method, target_is_directory=True)
    for path in derived_files:
        path.unlink(missing_ok=True)
    for repair in repairs:
        repair.path.unlink()
        repair.path.symlink_to(repair.target, target_is_directory=True)
    for directory in sorted(
        set(cleanup_dirs), key=lambda path: len(path.parts), reverse=True
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    print(
        f"Migrated {len(moves)} directories/files into the shared selector layout "
        f"and repaired {len(repairs)} symlinks."
    )
    return len(moves) + len(repairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("analysis_outputs_v3"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    migrate(args.output_root, apply=bool(args.apply))


if __name__ == "__main__":
    main()
