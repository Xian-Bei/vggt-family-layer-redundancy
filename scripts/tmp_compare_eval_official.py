#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONES = ("vggt", "fastvggt", "vggt_omega", "pi3")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _json_print(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def _run_subprocess(payload: Dict[str, Any], conda_env: str) -> Dict[str, Any]:
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        str(Path(__file__).resolve()),
        "--worker",
        json.dumps(payload, ensure_ascii=False),
    ]
    env = dict(os.environ)
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    env.setdefault("MPLBACKEND", "Agg")
    p = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    parsed: Dict[str, Any] | None = None
    for ln in reversed(lines):
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict) and obj.get("kind") == "worker_result":
                parsed = obj
                break
        except Exception:
            continue
    if parsed is None:
        raise RuntimeError(
            f"Worker failed to produce JSON.\nexit={p.returncode}\noutput:\n{p.stdout}"
        )
    if p.returncode != 0:
        raise RuntimeError(f"Worker failed (exit {p.returncode}): {parsed}")
    return parsed


def _discover_default_scenes(data_root: Path, max_scenes: int) -> List[str]:
    from analysis.eval.scannet_eval import discover_scannet_scenes

    scenes = discover_scannet_scenes(data_root)
    return [s.scene_id for s in scenes[: max(1, int(max_scenes))]]


def _worker_load_scene_paths(scene_ids: Sequence[str], data_root: Path, max_frames: int) -> Dict[str, List[str]]:
    from analysis.eval.scannet_eval import discover_scannet_scenes, select_scene_image_paths_fastvggt

    all_scenes = {s.scene_id: s for s in discover_scannet_scenes(data_root)}
    out: Dict[str, List[str]] = {}
    for sid in scene_ids:
        if sid not in all_scenes:
            raise FileNotFoundError(f"Scene not found: {sid}")
        sel = select_scene_image_paths_fastvggt(all_scenes[sid], max_frames=int(max_frames))
        out[sid] = [str(p) for p in sel.image_paths]
    return out


def _bench(fn, *, warmup: int, runs: int):
    import numpy as np
    import torch

    for _ in range(max(0, int(warmup))):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    vals: List[float] = []
    for _ in range(max(1, int(runs))):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(vals)), vals


def _device_dtype_for_our_eval():
    import torch
    from analysis.amp_policy import amp_dtype_for_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, amp_dtype_for_device(device)


@dataclass
class ModeRunResult:
    mode: str
    backbone: str
    module_file: str
    model_class: str
    device: str
    autocast_dtype: str
    scene_ms: Dict[str, float]
    input_shapes: Dict[str, List[int]]


def _worker_run_our(backbone: str, scene_to_paths: Dict[str, List[str]], warmup: int, runs: int) -> ModeRunResult:
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms as TF
    from analysis.amp_policy import autocast_device_dtype
    from analysis.backbones.bootstrap import apply_pythonpath
    from analysis.backbones.registry import get_backbone

    def load_and_preprocess_images_crop_local(image_path_list: Sequence[str]) -> torch.Tensor:
        # Mirror analysis.eval.scannet_eval -> vggt.utils.load_fn(load_and_preprocess_images, mode="crop")
        to_tensor = TF.ToTensor()
        target_size = 518
        imgs: List[torch.Tensor] = []
        for image_path in image_path_list:
            img = Image.open(image_path)
            if img.mode == "RGBA":
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img)
            img = img.convert("RGB")
            width, height = img.size
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            t = to_tensor(img)
            if new_height > target_size:
                start_y = (new_height - target_size) // 2
                t = t[:, start_y : start_y + target_size, :]
            imgs.append(t)
        shapes = {(int(t.shape[1]), int(t.shape[2])) for t in imgs}
        if len(shapes) > 1:
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)
            padded: List[torch.Tensor] = []
            for t in imgs:
                hp = max_h - int(t.shape[1])
                wp = max_w - int(t.shape[2])
                if hp > 0 or wp > 0:
                    top = hp // 2
                    bottom = hp - top
                    left = wp // 2
                    right = wp - left
                    t = torch.nn.functional.pad(
                        t,
                        (left, right, top, bottom),
                        mode="constant",
                        value=1.0,
                    )
                padded.append(t)
            imgs = padded
        return torch.stack(imgs)

    apply_pythonpath(backbone)
    adapter = get_backbone(backbone)
    device, amp_dtype = _device_dtype_for_our_eval()
    model = adapter.load_pretrained(
        hf_model=str(adapter.default_hf_model()),
        device=str(device),
    )
    model.eval()
    module_file = str(Path(sys.modules[type(model).__module__].__file__).resolve())
    scene_ms: Dict[str, float] = {}
    input_shapes: Dict[str, List[int]] = {}
    ac_kw = autocast_device_dtype(device)

    for sid in scene_to_paths:
        # Keep exactly same input construction as ScanNet plugin (crop mode + frame selection).
        image_tensor = load_and_preprocess_images_crop_local(scene_to_paths[sid])
        images_b = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
        input_shapes[sid] = list(images_b.shape)

        def fn():
            with torch.no_grad(), torch.autocast(**ac_kw):
                _ = adapter.forward_predictions(model, images_b)

        mean_ms, _ = _bench(fn, warmup=warmup, runs=runs)
        scene_ms[sid] = float(mean_ms)

    return ModeRunResult(
        mode="our_eval",
        backbone=backbone,
        module_file=module_file,
        model_class=type(model).__name__,
        device=str(device),
        autocast_dtype=str(amp_dtype).split(".")[-1],
        scene_ms=scene_ms,
        input_shapes=input_shapes,
    )


def _worker_run_official(backbone: str, scene_to_paths: Dict[str, List[str]], warmup: int, runs: int) -> ModeRunResult:
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms as TF

    if backbone == "vggt":
        sys.path.insert(0, str(REPO / "backbones" / "vggt"))
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images

        model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        module_file = str(Path(sys.modules[type(model).__module__].__file__).resolve())
        scene_ms: Dict[str, float] = {}
        input_shapes: Dict[str, List[int]] = {}
        for sid, paths in scene_to_paths.items():
            x = load_and_preprocess_images(paths, mode="crop").unsqueeze(0).cuda()
            input_shapes[sid] = list(x.shape)

            def fn():
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
                    _ = model(x)

            mean_ms, _ = _bench(fn, warmup=warmup, runs=runs)
            scene_ms[sid] = float(mean_ms)

        return ModeRunResult(
            mode="official_demo",
            backbone=backbone,
            module_file=module_file,
            model_class=type(model).__name__,
            device="cuda",
            autocast_dtype=str(amp_dtype).split(".")[-1],
            scene_ms=scene_ms,
            input_shapes=input_shapes,
        )

    if backbone == "fastvggt":
        sys.path.insert(0, str(REPO / "backbones" / "FastVGGT"))
        from vggt.models.vggt import VGGT

        def get_vgg_input_imgs_local(images: np.ndarray):
            to_tensor = TF.ToTensor()
            vgg_input_images = []
            final_width = None
            final_height = None
            for image in images:
                img = Image.fromarray(image, mode="RGB")
                width, height = img.size
                new_width = 518
                new_height = round(height * (new_width / width) / 14) * 14
                img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
                img = to_tensor(img)
                if new_height > 518:
                    start_y = (new_height - 518) // 2
                    img = img[:, start_y : start_y + 518, :]
                    final_height = 518
                else:
                    final_height = new_height
                final_width = new_width
                vgg_input_images.append(img)
            vgg_input_images = torch.stack(vgg_input_images)
            patch_width = int(final_width // 14)
            patch_height = int(final_height // 14)
            return vgg_input_images, patch_width, patch_height

        model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        module_file = str(Path(sys.modules[type(model).__module__].__file__).resolve())
        scene_ms: Dict[str, float] = {}
        input_shapes: Dict[str, List[int]] = {}
        for sid, paths in scene_to_paths.items():
            imgs = [np.array(Image.open(p).convert("RGB")) for p in paths]
            x, patch_w, patch_h = get_vgg_input_imgs_local(np.stack(imgs, axis=0))
            model.update_patch_dimensions(int(patch_w), int(patch_h))
            x = x.cuda()
            input_shapes[sid] = list(x.shape)

            def fn():
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
                    _ = model(x.to(torch.bfloat16), image_paths=paths)

            mean_ms, _ = _bench(fn, warmup=warmup, runs=runs)
            scene_ms[sid] = float(mean_ms)

        return ModeRunResult(
            mode="official_demo",
            backbone=backbone,
            module_file=module_file,
            model_class=type(model).__name__,
            device="cuda",
            autocast_dtype=str(amp_dtype).split(".")[-1],
            scene_ms=scene_ms,
            input_shapes=input_shapes,
        )

    if backbone == "vggt_omega":
        sys.path.insert(0, str(REPO / "backbones" / "vggt-omega"))
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images

        model = VGGTOmega().cuda().eval()
        ckpt = torch.load(
            REPO / "backbones" / "vggt-omega" / "vggt_omega_1b_512.pt",
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict"), dict):
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
        module_file = str(Path(sys.modules[type(model).__module__].__file__).resolve())
        scene_ms: Dict[str, float] = {}
        input_shapes: Dict[str, List[int]] = {}
        for sid, paths in scene_to_paths.items():
            x = load_and_preprocess_images(paths, mode="balanced", image_resolution=512).unsqueeze(0).cuda()
            input_shapes[sid] = list(x.shape)

            def fn():
                with torch.inference_mode():
                    _ = model(x)

            mean_ms, _ = _bench(fn, warmup=warmup, runs=runs)
            scene_ms[sid] = float(mean_ms)

        return ModeRunResult(
            mode="official_demo",
            backbone=backbone,
            module_file=module_file,
            model_class=type(model).__name__,
            device="cuda",
            autocast_dtype="internal",
            scene_ms=scene_ms,
            input_shapes=input_shapes,
        )

    if backbone == "pi3":
        sys.path.insert(0, str(REPO / "backbones" / "Pi3"))
        from pi3.models.pi3 import Pi3

        model = Pi3.from_pretrained("yyfz233/Pi3").cuda().eval()
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        module_file = str(Path(sys.modules[type(model).__module__].__file__).resolve())
        scene_ms: Dict[str, float] = {}
        input_shapes: Dict[str, List[int]] = {}
        for sid, paths in scene_to_paths.items():
            pil = [Image.open(p).convert("RGB") for p in paths]
            w, h = pil[0].size
            max_pixels = 255000
            scale = math.sqrt(max_pixels / (w * h)) if w * h > 0 else 1.0
            wt, ht = w * scale, h * scale
            k, m = round(wt / 14), round(ht / 14)
            while (k * 14) * (m * 14) > max_pixels:
                if k / max(m, 1) > wt / max(ht, 1e-6):
                    k -= 1
                else:
                    m -= 1
            tw, th = max(1, k) * 14, max(1, m) * 14
            x = torch.stack(
                [
                    torch.from_numpy(np.array(im.resize((tw, th), Image.Resampling.LANCZOS)))
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                    for im in pil
                ],
                dim=0,
            ).cuda()
            input_shapes[sid] = list(x.shape)

            def fn():
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                    _ = model(x[None])

            mean_ms, _ = _bench(fn, warmup=warmup, runs=runs)
            scene_ms[sid] = float(mean_ms)

        return ModeRunResult(
            mode="official_demo",
            backbone=backbone,
            module_file=module_file,
            model_class=type(model).__name__,
            device="cuda",
            autocast_dtype=str(amp_dtype).split(".")[-1],
            scene_ms=scene_ms,
            input_shapes=input_shapes,
        )

    raise ValueError(f"Unsupported backbone: {backbone}")


def _worker_main(payload_json: str) -> None:
    payload = json.loads(payload_json)
    mode = str(payload["mode"])
    backbone = str(payload["backbone"])
    scene_ids = [str(x) for x in payload["scene_ids"]]
    data_root = Path(payload["data_root"])
    max_frames = int(payload["max_frames"])
    warmup = int(payload["warmup"])
    runs = int(payload["runs"])

    scene_to_paths = payload.get("scene_to_paths")
    if not isinstance(scene_to_paths, dict):
        # Fallback for direct worker invocation.
        scene_to_paths = _worker_load_scene_paths(scene_ids, data_root, max_frames)
    scene_to_paths = {str(k): [str(x) for x in v] for k, v in scene_to_paths.items()}
    try:
        if mode == "our_eval":
            res = _worker_run_our(backbone, scene_to_paths, warmup, runs)
        elif mode == "official_demo":
            res = _worker_run_official(backbone, scene_to_paths, warmup, runs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        _json_print(
            {
                "kind": "worker_result",
                "ok": True,
                "result": {
                    "mode": res.mode,
                    "backbone": res.backbone,
                    "module_file": res.module_file,
                    "model_class": res.model_class,
                    "device": res.device,
                    "autocast_dtype": res.autocast_dtype,
                    "scene_ms": res.scene_ms,
                    "input_shapes": res.input_shapes,
                },
            }
        )
    except Exception as e:
        _json_print(
            {
                "kind": "worker_result",
                "ok": False,
                "mode": mode,
                "backbone": backbone,
                "error": repr(e),
            }
        )
        raise


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Temporary A/B benchmark: our eval inference vs official demo inference."
    )
    ap.add_argument("--worker", type=str, default="")
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "scans_dev50")
    ap.add_argument("--scene-ids", type=str, default="")
    ap.add_argument("--num-scenes", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=50)
    ap.add_argument("--backbones", nargs="*", default=list(DEFAULT_BACKBONES))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--threshold-pct", type=float, default=5.0)
    ap.add_argument("--conda-env", type=str, default="vggt")
    args = ap.parse_args()

    if args.worker:
        _worker_main(args.worker)
        return

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {args.data_root}")

    os.environ["PYTHONPATH"] = f"{REPO}:{os.environ.get('PYTHONPATH', '')}".strip(":")
    if args.scene_ids.strip():
        scene_ids = [x.strip() for x in args.scene_ids.split(",") if x.strip()]
    else:
        scene_ids = _discover_default_scenes(args.data_root, args.num_scenes)
    if not scene_ids:
        raise RuntimeError("No scenes selected.")
    scene_to_paths = _worker_load_scene_paths(scene_ids, args.data_root, int(args.max_frames))

    summary: Dict[str, Any] = {
        "scenes": scene_ids,
        "max_frames": int(args.max_frames),
        "warmup": int(args.warmup),
        "runs": int(args.runs),
        "threshold_pct": float(args.threshold_pct),
        "results": {},
        "violations": [],
    }

    for bb in args.backbones:
        _json_print({"kind": "progress", "msg": f"Running {bb}: our_eval"})
        our = _run_subprocess(
            {
                "mode": "our_eval",
                "backbone": bb,
                "scene_ids": scene_ids,
                "data_root": str(args.data_root),
                "max_frames": int(args.max_frames),
                "scene_to_paths": scene_to_paths,
                "warmup": int(args.warmup),
                "runs": int(args.runs),
            },
            conda_env=args.conda_env,
        )["result"]

        _json_print({"kind": "progress", "msg": f"Running {bb}: official_demo"})
        off = _run_subprocess(
            {
                "mode": "official_demo",
                "backbone": bb,
                "scene_ids": scene_ids,
                "data_root": str(args.data_root),
                "max_frames": int(args.max_frames),
                "scene_to_paths": scene_to_paths,
                "warmup": int(args.warmup),
                "runs": int(args.runs),
            },
            conda_env=args.conda_env,
        )["result"]

        our_avg = float(sum(float(our["scene_ms"][s]) for s in scene_ids) / len(scene_ids))
        off_avg = float(sum(float(off["scene_ms"][s]) for s in scene_ids) / len(scene_ids))
        diff_pct = float((our_avg - off_avg) / max(off_avg, 1e-9) * 100.0)

        row = {
            "our_avg_ms": our_avg,
            "official_avg_ms": off_avg,
            "diff_pct_vs_official": diff_pct,
            "within_threshold": abs(diff_pct) <= float(args.threshold_pct),
            "our": our,
            "official": off,
        }
        summary["results"][bb] = row
        if not row["within_threshold"]:
            summary["violations"].append(bb)

    _json_print({"kind": "final", "summary": summary})


if __name__ == "__main__":
    main()
