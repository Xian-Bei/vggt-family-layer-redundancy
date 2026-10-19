"""Exclusive-GPU forward-only inference benchmark for test-time variants."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from analysis.amp_policy import autocast_device_dtype
from analysis.backbones.bootstrap import apply_pythonpath
from analysis.backbones.registry import get_backbone
from analysis.datasets.batch_frames import count_eval_frames
from analysis.datasets.registry import get_dataset
from analysis.inference_compression.contracts import InferenceCompressionConfig
from analysis.inference_compression.registry import get_inference_compression
from analysis.prune_plan import UnitPrunePlan, load_named_prune_plan
from analysis.prune_plan.recovery_plan import compose_recovery_unit_prune_plan
from analysis.recovery.train.checkpoint_v3 import apply_eval_model_structure

GIB = float(1024**3)
SCHEMA_VERSION = 1
FULL_VARIANTS = frozenset({"full", "full_fastvggt", "full_quantvggt"})
VARIANT_COMPRESSIONS = {
    "full": "none",
    "full_fastvggt": "fastvggt",
    "full_quantvggt": "quantvggt",
    "pruning": "none",
    "recovery": "none",
    "recovery_fastvggt": "fastvggt",
    "recovery_quantvggt": "quantvggt",
}


@dataclass(frozen=True)
class BenchmarkJob:
    dataset: str
    data_root: Path
    output_dataset: str
    split: str | None = None


@dataclass(frozen=True)
class ComputeApp:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mib: int | None


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    total_i = int(total)
    count_i = int(count)
    if count_i <= 0:
        raise ValueError("sample count must be positive")
    if total_i < count_i:
        raise ValueError(f"dataset has {total_i} scenes; benchmark requires {count_i}")
    if count_i == 1:
        return [0]
    values = np.linspace(0, total_i - 1, num=count_i)
    indices = [int(round(float(value))) for value in values]
    if len(set(indices)) != count_i:
        raise RuntimeError(f"even sampling produced duplicate indices: {indices}")
    return indices


def summarize_samples(values: Sequence[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("cannot summarize an empty sample list")
    return {
        "mean": float(statistics.fmean(samples)),
        "median": float(statistics.median(samples)),
        "std": float(statistics.pstdev(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "p95": float(np.percentile(np.asarray(samples, dtype=np.float64), 95.0)),
    }


def _safe_tag(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return normalized.strip("-") or "none"


def benchmark_config_id(
    *,
    variant: str,
    selector: str,
    recovery_plan_name: str,
    compression_config: InferenceCompressionConfig,
) -> str:
    parts = [_safe_tag(variant), _safe_tag(selector)]
    if recovery_plan_name:
        parts.append(_safe_tag(recovery_plan_name))
    if compression_config.name == "fastvggt":
        ratio = f"{float(compression_config.fastvggt_merge_ratio):g}".replace(".", "p")
        parts.append(
            f"fastvggt_m{int(compression_config.fastvggt_merge_start)}_r{ratio}"
        )
    elif compression_config.name == "quantvggt":
        parts.append(f"quantvggt_{_safe_tag(compression_config.quantvggt_dtype)}")
    return "__".join(parts)


def _run_checked(command: Sequence[str], *, timeout: float = 10.0) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise RuntimeError(
            f"GPU exclusivity query failed: {' '.join(command)}: {exc}"
        ) from exc
    return completed.stdout


def query_compute_apps(
    runner: Callable[[Sequence[str]], str] | None = None,
) -> list[ComputeApp]:
    run = runner or _run_checked
    output = run(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    )
    apps: list[ComputeApp] = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != 4:
            raise RuntimeError(f"unexpected nvidia-smi compute-app row: {row!r}")
        try:
            pid = int(row[1].strip())
        except ValueError as exc:
            raise RuntimeError(f"invalid compute-app PID in row: {row!r}") from exc
        memory_text = row[3].strip()
        try:
            memory = None if memory_text in {"", "N/A", "[N/A]"} else int(memory_text)
        except ValueError as exc:
            raise RuntimeError(f"invalid compute-app memory in row: {row!r}") from exc
        apps.append(
            ComputeApp(
                gpu_uuid=row[0].strip(),
                pid=pid,
                process_name=row[2].strip(),
                used_memory_mib=memory,
            )
        )
    return apps


def _gpu_metadata(gpu_uuid: str) -> dict[str, Any]:
    output = _run_checked(
        (
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total,temperature.gpu,power.limit,clocks.current.sm",
            "--format=csv,noheader,nounits",
        )
    )
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if row and row[0].strip() == gpu_uuid:
            if len(row) != 7:
                raise RuntimeError(f"unexpected nvidia-smi GPU row: {row!r}")
            return {
                "uuid": row[0].strip(),
                "name": row[1].strip(),
                "driver_version": row[2].strip(),
                "memory_total_mib": float(row[3].strip()),
                "temperature_c": float(row[4].strip()),
                "power_limit_w": float(row[5].strip()),
                "sm_clock_mhz": float(row[6].strip()),
            }
    raise RuntimeError(f"target GPU UUID {gpu_uuid!r} is missing from nvidia-smi")


class ExclusiveGpuGuard:
    """Fail closed when another compute process uses the benchmark GPU."""

    def __init__(self, device: torch.device, *, poll_interval_ms: int = 200) -> None:
        if device.type != "cuda":
            raise ValueError(f"inference benchmark requires CUDA, got {device}")
        if int(poll_interval_ms) <= 0:
            raise ValueError("GPU monitor poll interval must be positive")
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device.index is None
            else device
        )
        self.poll_interval_s = float(poll_interval_ms) / 1000.0
        self.pid = os.getpid()
        self.gpu_uuid = ""
        self.metadata_start: dict[str, Any] = {}
        self._lock_path: Path | None = None
        self._lock_handle: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._violation: BaseException | None = None
        self._violation_lock = threading.Lock()

    def _resolve_gpu_uuid(self) -> str:
        marker = torch.empty(1, device=self.device)
        torch.cuda.synchronize(self.device)
        apps = query_compute_apps()
        del marker
        own_uuids = {app.gpu_uuid for app in apps if app.pid == self.pid}
        if len(own_uuids) != 1:
            raise RuntimeError(
                "could not resolve exactly one physical GPU UUID from the current CUDA PID; "
                f"pid={self.pid} uuids={sorted(own_uuids)}"
            )
        return next(iter(own_uuids))

    def _poll_once(self) -> None:
        apps = query_compute_apps()
        own_present = any(
            app.gpu_uuid == self.gpu_uuid and app.pid == self.pid for app in apps
        )
        if not own_present:
            raise RuntimeError(
                f"current benchmark PID {self.pid} disappeared from GPU {self.gpu_uuid}"
            )
        foreign = [
            app for app in apps if app.gpu_uuid == self.gpu_uuid and app.pid != self.pid
        ]
        if foreign:
            details = ", ".join(
                f"pid={app.pid} process={app.process_name} memory={app.used_memory_mib}MiB"
                for app in foreign
            )
            raise RuntimeError(
                f"exclusive GPU requirement violated on {self.gpu_uuid}: {details}"
            )

    def _monitor(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            try:
                self._poll_once()
            except BaseException as exc:  # Preserve query failures for the main thread.
                with self._violation_lock:
                    if self._violation is None:
                        self._violation = exc
                self._stop.set()
                return

    def check(self) -> None:
        with self._violation_lock:
            violation = self._violation
        if violation is not None:
            raise RuntimeError(
                f"GPU exclusivity monitor failed: {violation}"
            ) from violation
        self._poll_once()

    def __enter__(self) -> "ExclusiveGpuGuard":
        torch.cuda.set_device(self.device)
        self.gpu_uuid = self._resolve_gpu_uuid()
        self._lock_path = (
            Path("/tmp") / f"lare_inference_benchmark_{_safe_tag(self.gpu_uuid)}.lock"
        )
        self._lock_handle = self._lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.check()
            self.metadata_start = _gpu_metadata(self.gpu_uuid)
            self._thread = threading.Thread(
                target=self._monitor,
                name="gpu-exclusivity-monitor",
                daemon=True,
            )
            self._thread.start()
            return self
        except BlockingIOError as exc:
            self._lock_handle.close()
            self._lock_handle = None
            raise RuntimeError(
                f"another LaRe benchmark holds {self._lock_path}"
            ) from exc
        except BaseException:
            self._lock_handle.close()
            self._lock_handle = None
            try:
                self._lock_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_interval_s * 2.0))
        if self._lock_handle is not None:
            try:
                if self._lock_path is not None:
                    try:
                        self._lock_path.unlink()
                    except FileNotFoundError:
                        pass
            finally:
                try:
                    fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                finally:
                    self._lock_handle.close()


def _selected_batches(
    dataset_plugin: Any,
    job: BenchmarkJob,
    *,
    sample_count: int,
    max_scenes: int,
    max_frames: int,
) -> Iterable[tuple[int, Any]]:
    total = dataset_plugin.count_scenes(
        job.data_root,
        max_scenes=int(max_scenes),
        split=job.split,
    )
    if total is None:
        raise RuntimeError(f"{job.dataset} does not provide a scene count")
    indices = evenly_spaced_indices(int(total), int(sample_count))
    wanted = set(indices)
    found = 0
    for index, batch in enumerate(
        dataset_plugin.iter_scenes(
            job.data_root,
            max_scenes=int(max_scenes),
            max_frames=int(max_frames),
            split=job.split,
        )
    ):
        if index not in wanted:
            continue
        yield index, batch
        found += 1
        if found == len(indices):
            break
    if found != len(indices):
        raise RuntimeError(
            f"{job.output_dataset} yielded {found}/{len(indices)} selected scenes"
        )


def _forward_once(
    adapter: Any, model: Any, images: torch.Tensor, device: torch.device
) -> Any:
    ac_kw = autocast_device_dtype(device)
    with torch.inference_mode(), torch.autocast(**ac_kw):
        return adapter.forward_predictions(model, images)


def benchmark_dataset(
    *,
    adapter: Any,
    model: Any,
    dataset_plugin: Any,
    job: BenchmarkJob,
    device: torch.device,
    guard: Any,
    sample_count: int = 10,
    max_scenes: int = 0,
    max_frames: int = 30,
    allow_variable_frames: bool = False,
) -> list[dict[str, Any]]:
    expected_signature: tuple[tuple[int, ...], str, int] | None = None
    expected_frame_independent_signature: tuple[int, tuple[int, ...], str] | None = None
    warmed_signatures: set[tuple[tuple[int, ...], str, int]] = set()
    rows: list[dict[str, Any]] = []
    for sample_index, (dataset_index, batch) in enumerate(
        _selected_batches(
            dataset_plugin,
            job,
            sample_count=sample_count,
            max_scenes=max_scenes,
            max_frames=max_frames,
        )
    ):
        prepared = dataset_plugin.prepare_inference_scene_cpu(adapter, batch)
        if not isinstance(prepared.images, torch.Tensor):
            raise TypeError(
                f"{job.output_dataset}/{batch.scene_id} inference images are not a tensor"
            )
        if prepared.images.is_cuda:
            raise ValueError("prepare_inference_scene_cpu must return CPU images")
        images = prepared.images.to(device=device, non_blocking=False)
        signature = (
            tuple(int(value) for value in images.shape),
            str(images.dtype),
            int(count_eval_frames(batch)),
        )
        if images.ndim != 5 or int(images.shape[1]) != signature[2]:
            raise ValueError(
                f"invalid benchmark input layout for {job.output_dataset}/{batch.scene_id}: "
                f"shape={signature[0]} num_frames={signature[2]}"
            )
        if expected_signature is None:
            expected_signature = signature
        elif not allow_variable_frames and signature != expected_signature:
            raise ValueError(
                f"benchmark input signature mismatch for {job.output_dataset}/{batch.scene_id}: "
                f"expected={expected_signature}, observed={signature}"
            )
        frame_independent_signature = (
            int(signature[0][0]),
            tuple(signature[0][2:]),
            signature[1],
        )
        if expected_frame_independent_signature is None:
            expected_frame_independent_signature = frame_independent_signature
        elif frame_independent_signature != expected_frame_independent_signature:
            raise ValueError(
                f"benchmark non-frame input signature mismatch for "
                f"{job.output_dataset}/{batch.scene_id}: "
                f"expected={expected_frame_independent_signature}, "
                f"observed={frame_independent_signature}"
            )

        if signature not in warmed_signatures:
            guard.check()
            warmup_prediction = _forward_once(adapter, model, images, device)
            torch.cuda.synchronize(device)
            del warmup_prediction
            gc.collect()
            guard.check()
            warmed_signatures.add(signature)

        guard.check()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device.index)
        baseline_allocated = float(torch.cuda.memory_allocated(device.index)) / GIB
        baseline_reserved = float(torch.cuda.memory_reserved(device.index)) / GIB
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device))
        prediction = _forward_once(adapter, model, images, device)
        end.record(torch.cuda.current_stream(device))
        end.synchronize()
        latency_ms = float(start.elapsed_time(end))
        peak_allocated = float(torch.cuda.max_memory_allocated(device.index)) / GIB
        peak_reserved = float(torch.cuda.max_memory_reserved(device.index)) / GIB
        guard.check()

        rows.append(
            {
                "scene_id": str(batch.scene_id),
                "dataset_index": int(dataset_index),
                "num_frames": int(signature[2]),
                "input_shape": list(signature[0]),
                "input_dtype": signature[1],
                "latency_ms": latency_ms,
                "memory_gib": {
                    "baseline_allocated": baseline_allocated,
                    "baseline_reserved": baseline_reserved,
                    "peak_allocated": peak_allocated,
                    "peak_reserved": peak_reserved,
                },
            }
        )
        del prediction, images, prepared
        gc.collect()

    if len(rows) != int(sample_count):
        raise RuntimeError(f"expected {sample_count} benchmark rows, got {len(rows)}")
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "num_frames": summarize_samples([float(row["num_frames"]) for row in rows]),
        "latency_ms": summarize_samples([float(row["latency_ms"]) for row in rows]),
        "baseline_allocated_gib": summarize_samples(
            [float(row["memory_gib"]["baseline_allocated"]) for row in rows]
        ),
        "baseline_reserved_gib": summarize_samples(
            [float(row["memory_gib"]["baseline_reserved"]) for row in rows]
        ),
        "peak_allocated_gib": summarize_samples(
            [float(row["memory_gib"]["peak_allocated"]) for row in rows]
        ),
        "peak_reserved_gib": summarize_samples(
            [float(row["memory_gib"]["peak_reserved"]) for row in rows]
        ),
    }


def benchmark_protocol(
    *,
    sample_count: int,
    max_scenes: int,
    max_frames: int,
    gpu_poll_interval_ms: int,
    allow_variable_frames: bool = False,
) -> dict[str, Any]:
    protocol = {
        "scene_selection": "evenly_spaced_over_ordered_candidate_set",
        "sample_count": int(sample_count),
        "warmup_runs_on_first_scene": 1,
        "measured_runs_per_scene": 1,
        "max_scenes_candidate_cap": int(max_scenes),
        "max_frames": int(max_frames),
        "timing_scope": "adapter.forward_predictions",
        "includes_model_load": False,
        "includes_cpu_preprocessing": False,
        "includes_h2d": False,
        "includes_d2h_or_scoring": False,
        "gpu_exclusivity_poll_interval_ms": int(gpu_poll_interval_ms),
    }
    if allow_variable_frames:
        protocol["allow_variable_frames"] = True
        protocol["warmup_runs_per_input_signature"] = 1
    return protocol


def benchmark_result_complete(
    path: Path,
    *,
    backbone: str,
    dataset: str,
    config_id: str,
    variant: str,
    selector: str,
    selection: Mapping[str, Any] | None,
    recovery_plan_name: str | None,
    compression: Mapping[str, Any],
    hf_model: str,
    checkpoint: Mapping[str, Any] | None,
    protocol: Mapping[str, Any],
) -> bool:
    """Return true only for a complete result matching the current run inputs."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            int(payload.get("schema_version", -1)) != SCHEMA_VERSION
            or payload.get("backbone") != backbone
            or payload.get("dataset") != dataset
        ):
            return False
        results = payload.get("results")
        result = results.get(config_id) if isinstance(results, dict) else None
        if not isinstance(result, dict):
            return False
        expected = {
            "config_id": config_id,
            "variant": variant,
            "selector": selector,
            "selection": dict(selection) if selection is not None else None,
            "recovery_plan_name": recovery_plan_name,
            "compression": dict(compression),
            "hf_model": hf_model,
            "checkpoint": dict(checkpoint) if checkpoint is not None else None,
            "protocol": dict(protocol),
        }
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        scenes = result.get("scenes")
        if not isinstance(scenes, list) or len(scenes) != int(protocol["sample_count"]):
            return False
        for scene in scenes:
            if not isinstance(scene, dict):
                return False
            latency = scene.get("latency_ms")
            memory = scene.get("memory_gib")
            if (
                not isinstance(latency, (int, float))
                or not math.isfinite(float(latency))
                or float(latency) <= 0.0
                or not isinstance(memory, dict)
                or not all(
                    isinstance(memory.get(key), (int, float))
                    and math.isfinite(float(memory[key]))
                    for key in (
                        "baseline_allocated",
                        "baseline_reserved",
                        "peak_allocated",
                        "peak_reserved",
                    )
                )
            ):
                return False
        summary = result.get("summary")
        return isinstance(summary, dict) and all(
            isinstance(summary.get(key), dict)
            for key in (
                "latency_ms",
                "baseline_allocated_gib",
                "baseline_reserved_gib",
                "peak_allocated_gib",
                "peak_reserved_gib",
            )
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


def atomic_merge_result(
    path: Path,
    *,
    backbone: str,
    dataset: str,
    config_id: str,
    result: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path_digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:20]
    lock_path = Path("/tmp") / f"lare_benchmark_json_{path_digest}.lock"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"benchmark JSON is being updated by another process: {path}"
            ) from exc
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError(f"unsupported benchmark schema in {path}")
            if payload.get("backbone") != backbone or payload.get("dataset") != dataset:
                raise ValueError(f"benchmark identity mismatch in {path}")
        else:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "backbone": backbone,
                "dataset": dataset,
                "results": {},
            }
        results = payload.setdefault("results", {})
        if not isinstance(results, dict):
            raise ValueError(f"benchmark results must be an object in {path}")
        results[config_id] = dict(result)
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def load_selection_aliases(path: Path, selector: str) -> list[dict[str, Any]]:
    aliases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if len(row) != 5:
                raise ValueError(f"invalid selection-map row: {row!r}")
            physical, drop_count, family, selected_strategy, score = row
            if physical != selector:
                continue
            aliases.append(
                {
                    "drop_count": int(drop_count),
                    "family": family,
                    "selected_strategy": selected_strategy,
                    "selection_score": float(score),
                }
            )
    if not aliases:
        raise ValueError(f"selection map has no aliases for selector {selector!r}")
    return aliases


def _file_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _git_record(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _parse_job(value: str) -> BenchmarkJob:
    fields = value.split("|", 3)
    if len(fields) != 4 or not all(fields[:3]):
        raise argparse.ArgumentTypeError(
            "--job must be DATASET|DATA_ROOT|OUTPUT_DATASET|SPLIT (SPLIT may be empty)"
        )
    return BenchmarkJob(
        dataset=fields[0],
        data_root=Path(fields[1]),
        output_dataset=fields[2],
        split=fields[3] or None,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--variant",
        choices=(
            "full",
            "full_fastvggt",
            "full_quantvggt",
            "pruning",
            "recovery",
            "recovery_fastvggt",
            "recovery_quantvggt",
        ),
        required=True,
    )
    parser.add_argument("--prune-plan-root", type=Path)
    parser.add_argument("--prune-plan-name", default="")
    parser.add_argument("--recovery-plan-root", type=Path)
    parser.add_argument("--recovery-plan-name", default="")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--inference-compression",
        choices=("none", "fastvggt", "quantvggt"),
        default="none",
    )
    parser.add_argument("--fastvggt-merge-start", type=int, default=0)
    parser.add_argument("--fastvggt-merge-ratio", type=float, default=0.9)
    parser.add_argument(
        "--quantvggt-dtype", choices=("w4a4", "w6a6", "w8a8"), default="w4a4"
    )
    parser.add_argument("--quantvggt-params-dir", type=Path)
    parser.add_argument("--selection-map", type=Path)
    parser.add_argument("--selection-datasets", default="")
    parser.add_argument("--output-root", type=Path, default=Path("analysis_outputs_v3"))
    parser.add_argument("--job", type=_parse_job, action="append", required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument(
        "--allow-variable-frames",
        action="store_true",
        help=(
            "Allow selected scenes to contain fewer than --max-frames while "
            "requiring batch/channel/spatial/dtype consistency; each distinct "
            "sequence length receives its own warm-up."
        ),
    )
    parser.add_argument("--gpu-poll-interval-ms", type=int, default=200)
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Recompute matching dataset results instead of resuming completed entries.",
    )
    return parser


def _compression_config(args: argparse.Namespace) -> InferenceCompressionConfig:
    return InferenceCompressionConfig(
        name=args.inference_compression,
        fastvggt_merge_start=args.fastvggt_merge_start,
        fastvggt_merge_ratio=args.fastvggt_merge_ratio,
        quantvggt_dtype=args.quantvggt_dtype,
        quantvggt_params_dir=args.quantvggt_params_dir,
    )


def _validate_variant_compression(
    variant: str, compression: InferenceCompressionConfig
) -> None:
    expected = VARIANT_COMPRESSIONS[variant]
    if compression.name != expected:
        raise ValueError(
            f"benchmark variant {variant!r} requires "
            f"--inference-compression={expected}, got {compression.name!r}"
        )


def _load_model(
    args: argparse.Namespace,
    device: torch.device,
    compression: InferenceCompressionConfig,
) -> tuple[Any, Any]:
    adapter = get_backbone(args.backbone)
    apply_pythonpath(adapter.name)
    model = adapter.load_pretrained(
        hf_model=adapter.default_hf_model(), device=str(device)
    )
    if args.variant in FULL_VARIANTS:
        prune_plan = None
        eval_mode = "upper"
        checkpoint = None
    elif args.variant == "pruning":
        prune_plan: UnitPrunePlan = load_named_prune_plan(
            args.prune_plan_root, args.prune_plan_name
        )
        eval_mode = "lower"
        checkpoint = None
    else:
        if (
            not args.recovery_plan_name
            or args.recovery_plan_root is None
            or args.checkpoint is None
        ):
            raise ValueError(
                "recovery benchmark variants require recovery plan root/name and checkpoint"
            )
        prune_plan = compose_recovery_unit_prune_plan(
            prune_plan_root=args.prune_plan_root,
            recovery_plan_root=args.recovery_plan_root,
            prune_selector=args.prune_plan_name,
            recovery_plan=args.recovery_plan_name,
        )
        eval_mode = "model"
        checkpoint = args.checkpoint
    apply_eval_model_structure(
        model,
        adapter,
        eval_mode=eval_mode,
        prune_plan=prune_plan,
        checkpoint=checkpoint,
        map_location=device,
    )
    if eval_mode == "model":
        from analysis.inference_compression.lora import merge_lora_linears

        merge_lora_linears(model)
    if compression.name != "none":
        compression_adapter = get_inference_compression(compression.name)
        compression_adapter.validate_backbone(adapter.name)
        compression_adapter.validate_config(compression)
        model = compression_adapter.apply(
            model,
            backbone_name=adapter.name,
            config=compression,
        )
    model.eval()
    return adapter, model


def _benchmark_output_path(
    output_root: Path, backbone: str, output_dataset: str
) -> Path:
    return (
        output_root
        / backbone
        / output_dataset
        / "eval"
        / "testtime"
        / "inference_benchmark_v3.json"
    )


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"inference benchmark requires CUDA, got {device}")
    if int(args.sample_count) != 10:
        raise ValueError(
            "test-time inference benchmark protocol requires exactly 10 scenes"
        )
    repo_root = Path(__file__).resolve().parents[2]
    compression = _compression_config(args)
    _validate_variant_compression(args.variant, compression)
    if args.variant in FULL_VARIANTS:
        selector = "full"
        selection = None
        recovery_plan_name = None
        checkpoint = None
    else:
        if (
            args.prune_plan_root is None
            or not args.prune_plan_name
            or args.selection_map is None
            or not args.selection_datasets
        ):
            raise ValueError(
                "pruning/recovery benchmark variants require prune plan and "
                "selection inputs"
            )
        selector = args.prune_plan_name
        aliases = load_selection_aliases(args.selection_map, selector)
        selection = {"datasets": args.selection_datasets, "aliases": aliases}
        recovery_plan_name = args.recovery_plan_name or None
        checkpoint = _file_record(args.checkpoint)
    config_id = benchmark_config_id(
        variant=args.variant,
        selector=selector,
        recovery_plan_name=recovery_plan_name or "",
        compression_config=compression,
    )
    adapter_metadata = get_backbone(args.backbone)
    hf_model = adapter_metadata.default_hf_model()
    protocol = benchmark_protocol(
        sample_count=args.sample_count,
        max_scenes=args.max_scenes,
        max_frames=args.max_frames,
        gpu_poll_interval_ms=args.gpu_poll_interval_ms,
        allow_variable_frames=args.allow_variable_frames,
    )
    pending_jobs: list[BenchmarkJob] = []
    for job in args.job:
        output_path = _benchmark_output_path(
            args.output_root, args.backbone, job.output_dataset
        )
        if not args.force_rerun and benchmark_result_complete(
            output_path,
            backbone=args.backbone,
            dataset=job.output_dataset,
            config_id=config_id,
            variant=args.variant,
            selector=selector,
            selection=selection,
            recovery_plan_name=recovery_plan_name,
            compression=compression.as_record(),
            hf_model=hf_model,
            checkpoint=checkpoint,
            protocol=protocol,
        ):
            print(
                f"[benchmark] reuse {output_path} config={config_id}",
                flush=True,
            )
        else:
            pending_jobs.append(job)
    if not pending_jobs:
        print(
            f"[benchmark] complete config={config_id}; no GPU work required", flush=True
        )
        return

    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with ExclusiveGpuGuard(
        device,
        poll_interval_ms=args.gpu_poll_interval_ms,
    ) as guard:
        adapter, model = _load_model(args, device, compression)
        guard.check()
        process_metadata = {
            "pid": os.getpid(),
            "gpu_start": guard.metadata_start,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "git": _git_record(repo_root),
        }
        for job in pending_jobs:
            plugin = get_dataset(job.dataset)
            split = plugin.validate_split(job.split)
            normalized_job = BenchmarkJob(
                dataset=job.dataset,
                data_root=job.data_root,
                output_dataset=job.output_dataset,
                split=split,
            )
            rows = benchmark_dataset(
                adapter=adapter,
                model=model,
                dataset_plugin=plugin,
                job=normalized_job,
                device=device,
                guard=guard,
                sample_count=args.sample_count,
                max_scenes=args.max_scenes,
                max_frames=args.max_frames,
                allow_variable_frames=args.allow_variable_frames,
            )
            guard.check()
            result = {
                "config_id": config_id,
                "variant": args.variant,
                "selector": selector,
                "selection": dict(selection) if selection is not None else None,
                "recovery_plan_name": recovery_plan_name,
                "compression": compression.as_record(),
                "hf_model": hf_model,
                "checkpoint": checkpoint,
                "protocol": protocol,
                "scenes": rows,
                "summary": summarize_rows(rows),
                "execution": {
                    **process_metadata,
                    "gpu_end": _gpu_metadata(guard.gpu_uuid),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            }
            output_path = _benchmark_output_path(
                args.output_root, args.backbone, job.output_dataset
            )
            atomic_merge_result(
                output_path,
                backbone=args.backbone,
                dataset=job.output_dataset,
                config_id=config_id,
                result=result,
            )
            print(
                f"[benchmark] wrote {output_path} config={config_id} "
                f"latency_mean={result['summary']['latency_ms']['mean']:.3f}ms",
                flush=True,
            )


if __name__ == "__main__":
    main()
