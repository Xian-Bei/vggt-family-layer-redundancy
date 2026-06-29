#!/usr/bin/env python3
"""Install pinned backbone and inference-compression repositories for LaRe.

Examples:
  python backbones/install_backbone.py vggt pi3
  python backbones/install_backbone.py all --download-ckpt
  python backbones/install_backbone.py all --source-root /tmp/backbones_backup
  python backbones/install_backbone.py pi3 --no-build-extensions
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOCK_PATH = SCRIPT_DIR / "backbones.lock.json"


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=env)


def load_lock() -> dict[str, Any]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def registry(lock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, spec in lock["backbones"].items():
        out[name] = spec
        for alias in spec.get("aliases", []):
            out[str(alias)] = spec
    return out


def canonical_names(lock: dict[str, Any], requested: list[str]) -> list[str]:
    reg = registry(lock)
    if not requested or requested == ["all"]:
        return list(lock["backbones"].keys())
    names: list[str] = []
    for raw in requested:
        key = raw.strip().lower()
        if key not in reg:
            raise SystemExit(f"Unknown backbone {raw!r}. Available: {', '.join(lock['backbones'])}, all")
        for canonical, spec in lock["backbones"].items():
            if spec is reg[key]:
                if canonical not in names:
                    names.append(canonical)
                break
    return names


def source_url(spec: dict[str, Any], source_root: Path | None) -> str:
    if source_root is None:
        return str(spec["url"])
    src = source_root / Path(spec["path"]).name
    if not src.is_dir():
        raise SystemExit(f"Missing local source for {spec['path']}: {src}")
    return str(src)


def clone_or_update(spec: dict[str, Any], *, force: bool, source_root: Path | None) -> None:
    path = REPO_ROOT / spec["path"]
    commit = str(spec["commit"])
    url = source_url(spec, source_root)

    if path.exists():
        if not force:
            if (path / ".git").exists():
                run(["git", "fetch", "--all", "--tags"], cwd=path)
                run(["git", "checkout", "--detach", commit], cwd=path)
                run(["git", "reset", "--hard", commit], cwd=path)
                run(["git", "clean", "-fd"], cwd=path)
                run(["git", "remote", "set-url", "origin", str(spec["url"])], cwd=path)
                return
            raise SystemExit(f"Path exists and is not a git checkout: {path}. Use --force to replace it.")
        shutil.rmtree(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", url, str(path)])
    run(["git", "checkout", "--detach", commit], cwd=path)
    run(["git", "reset", "--hard", commit], cwd=path)
    run(["git", "clean", "-fd"], cwd=path)
    run(["git", "remote", "set-url", "origin", str(spec["url"])], cwd=path)


def download_checkpoint(spec: dict[str, Any]) -> None:
    weight = spec.get("default_weight") or {}
    kind = weight.get("kind")
    if kind in ("huggingface_model_id", None):
        print(f"[ckpt] {spec['path']}: no local checkpoint download required ({weight.get('value', 'n/a')}).")
        return
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise SystemExit("Install huggingface_hub or use huggingface-cli before --download-ckpt.") from exc

    if kind == "huggingface_file":
        local_path = REPO_ROOT / str(weight["local_path"])
        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=str(weight["repo_id"]),
            filename=str(weight["filename"]),
            local_dir=str(local_path.parent),
            local_dir_use_symlinks=False,
        )
        got = Path(downloaded)
        if got.resolve() != local_path.resolve() and got.is_file():
            shutil.copy2(got, local_path)
        print(f"[ckpt] wrote {local_path}")
        return

    if kind == "huggingface_files":
        local_dir = REPO_ROOT / str(weight["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        for filename in weight.get("filenames", []):
            hf_hub_download(
                repo_id=str(weight["repo_id"]),
                filename=str(filename),
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[ckpt] wrote {local_dir / filename}")
        return

    raise SystemExit(f"Unsupported checkpoint kind for {spec['path']}: {kind}")


def copy_checkpoint_from_source(spec: dict[str, Any], source_root: Path | None) -> None:
    if source_root is None:
        return
    weight = spec.get("default_weight") or {}
    kind = weight.get("kind")
    if kind == "huggingface_file":
        rel = Path(str(weight["local_path"]))
        src = source_root / Path(spec["path"]).name / rel.name
        dst = REPO_ROOT / rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[ckpt] copied {src} -> {dst}")
        return
    if kind == "huggingface_files":
        src_dir = source_root / Path(spec["path"]).name / Path(str(weight["local_dir"])).name
        dst_dir = REPO_ROOT / str(weight["local_dir"])
        for filename in weight.get("filenames", []):
            src = src_dir / str(filename)
            if src.is_file():
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst_dir / str(filename))
                print(f"[ckpt] copied {src} -> {dst_dir / str(filename)}")


def build_optional_extensions(name: str, *, cuda_arch_list: str | None) -> None:
    if name != "pi3":
        return
    ext_dir = REPO_ROOT / "analysis" / "backbones" / "vendor_extensions" / "pi3_curope" / "models" / "curope"
    setup_py = ext_dir / "setup.py"
    if not setup_py.is_file():
        print(f"[ext] Pi3 cuRoPE setup.py not found, skipping: {setup_py}")
        return
    print("[ext] building optional Pi3 cuRoPE CUDA extension")
    env = os.environ.copy()
    if cuda_arch_list:
        env["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list
    elif "TORCH_CUDA_ARCH_LIST" not in env:
        env["TORCH_CUDA_ARCH_LIST"] = "8.6"
    try:
        run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=ext_dir, env=env)
    except subprocess.CalledProcessError:
        print(
            "[ext] warning: Pi3 cuRoPE build failed; Pi3 will fall back to its slower PyTorch RoPE implementation.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backbones", nargs="*", help="Vendored component names or 'all'.")
    parser.add_argument("--force", action="store_true", help="Replace existing checkout directories.")
    parser.add_argument("--download-ckpt", action="store_true", help="Download local checkpoints when required.")
    parser.add_argument(
        "--no-build-extensions",
        action="store_true",
        help="Skip optional local CUDA extension builds such as Pi3 cuRoPE.",
    )
    parser.add_argument(
        "--cuda-arch-list",
        default=None,
        help=(
            "TORCH_CUDA_ARCH_LIST value for optional extension builds. "
            "Defaults to the environment value, or 8.6 when unset."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Clone from a local backup directory containing the backbone folders, for offline tests.",
    )
    args = parser.parse_args()

    lock = load_lock()
    for name in canonical_names(lock, args.backbones):
        spec = lock["backbones"][name]
        print(f"== installing {name} -> {spec['path']} @ {spec['commit'][:12]} ==")
        clone_or_update(spec, force=bool(args.force), source_root=args.source_root)
        copy_checkpoint_from_source(spec, args.source_root)
        if not args.no_build_extensions:
            build_optional_extensions(name, cuda_arch_list=args.cuda_arch_list)
        if args.download_ckpt:
            download_checkpoint(spec)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
