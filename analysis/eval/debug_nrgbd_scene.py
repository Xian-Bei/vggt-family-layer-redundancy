#!/usr/bin/env python3
"""Diagnose NRGBD / 7andn memory and subsampling for one scene (e.g. thin_geometry).

NRGBD uses ``len(os.listdir(.../images))`` and subsamples with::

    img_idxs = img_idxs[:: min(kf_every, len(img_idxs) // 2)]

so the *effective* view count is not obvious from "how big the room looks".
Run from repo root::

    conda activate vggt
    PYTHONPATH=. python analysis/eval/debug_nrgbd_scene.py \\
        --root data/neural_rgbd_data --scene thin_geometry --kf 10

Optional GPU smoke test (loads VGGT like eval_7andN.py)::

    PYTHONPATH=. python analysis/eval/debug_nrgbd_scene.py ... --forward
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import torch

from analysis.amp_policy import amp_dtype_for_device, autocast_device_dtype


def _nrgbd_img_indices(*, root: str, scene: str, kf_every: int) -> tuple[list[str], int, int]:
    """Mirror ``NRGBD._get_views`` index logic (no image loading)."""
    img_dir = osp.join(root, scene, "images")
    names = os.listdir(img_dir)
    num_files = len(names)
    img_idxs = [f"{i}" for i in range(num_files)]
    denom = len(img_idxs) // 2
    step_raw = min(int(kf_every), denom) if denom > 0 else int(kf_every)
    if step_raw == 0:
        raise ValueError(
            f"{scene}: subsample step would be 0 (len(img_idxs)={len(img_idxs)}, kf_every={kf_every}). "
            "This is a bug-prone edge case in data.py (min(kf, len//2) with len//2==0)."
        )
    subsampled = img_idxs[::step_raw]
    return subsampled, num_files, step_raw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=str, default="data/neural_rgbd_data", help="NRGBD dataset root")
    p.add_argument("--scene", type=str, default="thin_geometry", help="Scene folder name under root")
    p.add_argument("--kf", type=int, default=10, help="kf_every (same as eval_7andN --kf)")
    p.add_argument("--size", type=int, default=518, help="518 -> (518,392) like eval_7andN")
    p.add_argument(
        "--compare-all",
        action="store_true",
        help="Print per-scene listdir count and effective view count for every scene under root.",
    )
    p.add_argument(
        "--forward",
        action="store_true",
        help="Run one VGGT Baseline forward on GPU (needs HF cache / weights like eval).",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--hf-model", type=str, default="facebook/VGGT-1B")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    scene = args.scene
    img_dir = osp.join(root, scene, "images")
    if not osp.isdir(img_dir):
        raise SystemExit(f"Missing images dir: {img_dir}")

    png = sorted(Path(img_dir).glob("img*.png"))
    listed = os.listdir(img_dir)
    subs, num_files, step = _nrgbd_img_indices(root=root, scene=scene, kf_every=int(args.kf))

    print("== NRGBD index / subsample (mirrors data.py) ==")
    print(f"  root:     {root}")
    print(f"  scene:    {scene}")
    print(f"  kf_every: {args.kf}")
    print(f"  listdir(images) count: {len(listed)}")
    print(f"  glob img*.png count:   {len(png)}")
    print(f"  range(num_files) len:  {num_files}")
    print(f"  subsample step:        min(kf, len//2) = min({args.kf}, {num_files // 2}) = {step}")
    print(f"  effective view count:  {len(subs)} (indices {subs[0]} … {subs[-1]})")

    if len(png) != num_files:
        print(
            "\n[warn] listdir count != img*.png count — NRGBD uses listdir length for range(), "
            "which can desync from real frames if extra files exist."
        )

    if args.compare_all:
        print("\n== All scenes under root (listdir count, effective views @ this kf) ==")
        rows = []
        for d in sorted(os.listdir(root)):
            pth = osp.join(root, d)
            if not osp.isdir(pth) or not osp.isdir(osp.join(pth, "images")):
                continue
            try:
                sub_i, nf, st = _nrgbd_img_indices(root=root, scene=d, kf_every=int(args.kf))
            except ValueError as e:
                rows.append((d, nf, -1, -1, str(e)))
                continue
            rows.append((d, nf, len(sub_i), st, ""))
        rows.sort(key=lambda r: r[1])
        for d, nf, nv, st, err in rows:
            if err:
                print(f"  {d:22s}  files={nf:5d}  views=?  step=?  {err}")
            else:
                print(f"  {d:22s}  files={nf:5d}  views={nv:5d}  step={st}")

    if args.size != 518:
        raise SystemExit("Only --size 518 is wired here (matches eval default).")
    res = (518, 392)

    # Dataset batch (CPU) — same path as eval_7andn_utils.
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    eval_dir = os.path.join(repo, "analysis", "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    from data import NRGBD  # noqa: E402
    from torch.utils.data._utils.collate import default_collate  # noqa: E402

    ds = NRGBD(
        split="test",
        ROOT=root,
        resolution=res,
        num_seq=1,
        full_video=True,
        kf_every=int(args.kf),
        seed=33,
        test_id=scene,
    )
    batch = default_collate([ds[0]])

    imgs_list = [v["img"] for v in batch]
    stacked = torch.cat(imgs_list, dim=0)
    print("\n== default_collate([dataset[0]]) ==")
    print(f"  len(batch) (views): {len(batch)}")
    print(f"  stacked img shape:   {tuple(stacked.shape)}  (~{stacked.numel() * stacked.element_size() / 1e6:.1f} MB)")

    if not args.forward:
        print("\n(Done. Pass --forward for a one-shot GPU forward + peak VRAM.)")
        return

    if not torch.cuda.is_available():
        raise SystemExit("--forward requires CUDA")

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    from analysis.recovery.train.model import build_model  # noqa: E402

    model = build_model(
        model_name="Baseline",
        backend="vggt",
        hf_model=str(args.hf_model),
        device="cpu",
        unit_plan=None,
        ckpt_path=None,
        train_dir=None,
        epoch=None,
        eval_mode=True,
    ).to(dev).eval()
    amp_dt = amp_dtype_for_device(dev)
    if amp_dt != torch.float32:
        model = model.to(amp_dt)

    x = stacked.to(dev, non_blocking=True)
    if x.dim() == 4:
        x = x.unsqueeze(0)
    if amp_dt != torch.float32:
        x = x.to(dtype=amp_dt)
    with torch.no_grad():
        with torch.autocast(**autocast_device_dtype(dev)):
            _ = model(x)
    torch.cuda.synchronize(dev)
    peak_gb = torch.cuda.max_memory_allocated(dev) / (1024.0**3)
    print(f"\n== GPU forward (Baseline vggt, amp_dtype={amp_dt}) ==")
    print(f"  peak VRAM (max_memory_allocated): {peak_gb:.3f} GiB")


if __name__ == "__main__":
    main()
