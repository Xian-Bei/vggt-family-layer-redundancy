"""Shared bounded preload and CUDA-to-CPU scoring pipelines for evaluation."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch

from analysis.backbones.contracts import PredictionBundle

DEFAULT_ASYNC_SCORE_WORKERS = 4
DEFAULT_ASYNC_SCORE_MAX_PENDING = 12
ASYNC_SCORE_FIELDS: Tuple[str, ...] = ("points", "camera_poses")
ASYNC_SCORE_DEFERRED_FIELDS: Tuple[str, ...] = (
    "depth",
    "deferred_unprojection_extrinsic",
    "deferred_unprojection_intrinsic",
    "camera_poses",
)


def async_score_enabled(requested: bool, device: torch.device) -> bool:
    return bool(requested) and device.type == "cuda"


def scene_preload_enabled(
    requested: bool,
    device: torch.device,
    *,
    export_vis_2d: bool,
    export_vis_3d: bool,
) -> bool:
    return (
        bool(requested)
        and device.type == "cuda"
        and not bool(export_vis_2d)
        and not bool(export_vis_3d)
    )


@dataclass
class PreloadedScene:
    batch: Any = None
    prepared: Any = None
    error: Optional[Exception] = None
    exhausted: bool = False


def _preload_next_scene(
    *,
    scene_iterator: Iterator[Any],
    prepare_fn: Callable[[Any], Any],
) -> PreloadedScene:
    try:
        batch = next(scene_iterator)
    except StopIteration:
        return PreloadedScene(exhausted=True)
    try:
        prepared = prepare_fn(batch)
        images = getattr(prepared, "images", None)
        if not isinstance(images, torch.Tensor) or images.device.type != "cpu":
            raise TypeError("CPU scene preparation must return CPU tensor images")
        images = images.detach().float().contiguous()
        if not images.is_pinned():
            images = images.pin_memory()
        prepared.images = images
        return PreloadedScene(
            batch=batch,
            prepared=prepared,
        )
    except Exception as exc:
        return PreloadedScene(
            batch=batch,
            error=exc,
        )


class AsyncScenePreloader:
    """Single-worker, one-scene lookahead for CPU-only evaluation preparation."""

    def __init__(
        self,
        *,
        scene_source: Iterator[Any],
        prepare_fn: Callable[[Any], Any],
    ) -> None:
        self.scene_iterator = iter(scene_source)
        self.prepare_fn = prepare_fn
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="eval-preload"
        )
        self.future: Optional[Future[PreloadedScene]] = self._submit()
        self.max_lookahead_seen = 1
        self.closed = False

    def _submit(self) -> Future[PreloadedScene]:
        return self.executor.submit(
            _preload_next_scene,
            scene_iterator=self.scene_iterator,
            prepare_fn=self.prepare_fn,
        )

    def next(self) -> Optional[PreloadedScene]:
        if self.closed or self.future is None:
            return None
        item = self.future.result()
        if item.exhausted:
            self.future = None
            return None
        # Start preparing N+1 before the main thread stages or infers N.
        self.future = self._submit()
        return item

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.future is not None:
                self.future.result()
        finally:
            self.future = None
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.closed = True


def stage_preloaded_scene_to_cuda(
    prepared: Any,
    *,
    device: torch.device,
    copy_stream: torch.cuda.Stream,
    timing: bool,
) -> tuple[torch.cuda.Event, Optional[torch.cuda.Event], torch.Tensor]:
    """Copy pinned scene images on a dedicated stream and gate the inference stream."""
    images = getattr(prepared, "images", None)
    if not isinstance(images, torch.Tensor) or images.device.type != "cpu":
        raise TypeError("Preloaded scene images must be a CPU tensor")
    if not images.is_pinned():
        raise ValueError("Preloaded scene images must use pinned memory")
    h2d_start = torch.cuda.Event(enable_timing=True) if timing else None
    ready = torch.cuda.Event(enable_timing=timing)
    with torch.cuda.stream(copy_stream):
        if h2d_start is not None:
            h2d_start.record(copy_stream)
        staged = images.to(device=device, non_blocking=True)
        ready.record(copy_stream)
    torch.cuda.current_stream(device).wait_event(ready)
    prepared.images = staged
    return ready, h2d_start, images


@dataclass
class _PinnedScoreSlot:
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class _ScoreResult:
    metrics: Dict[str, float]
    score_s: float


@dataclass
class _PendingScore:
    key: Any
    slot: _PinnedScoreSlot
    future: Future[_ScoreResult]
    ready_event: torch.cuda.Event
    copy_start_event: Optional[torch.cuda.Event] = None


def stage_prediction_for_cpu(
    prediction: PredictionBundle,
    *,
    slot: _PinnedScoreSlot,
    device: torch.device,
    copy_stream: torch.cuda.Stream,
    timing: bool,
) -> tuple[PredictionBundle, torch.cuda.Event, Optional[torch.cuda.Event]]:
    """Stage only the tensor fields consumed by current evaluation scorers."""
    deferred_backbone = prediction.raw.get("deferred_unprojection_backbone")
    deferred = bool(deferred_backbone)
    field_names = ASYNC_SCORE_DEFERRED_FIELDS if deferred else ASYNC_SCORE_FIELDS
    sources: Dict[str, Optional[torch.Tensor]] = {}
    originals: Dict[str, Optional[torch.Tensor]] = {}
    for name in field_names:
        value = (
            prediction.raw.get(name)
            if name.startswith("deferred_unprojection_")
            else getattr(prediction, name)
        )
        if value is None:
            sources[name] = None
            originals[name] = None
            continue
        if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
            raise TypeError(f"Async scoring requires CUDA tensor {name}, got {type(value)!r}")
        sources[name] = value.detach().float()
        originals[name] = value

    current_stream = torch.cuda.current_stream(device)
    copy_stream.wait_stream(current_stream)
    copy_start = torch.cuda.Event(enable_timing=True) if timing else None
    ready = torch.cuda.Event(enable_timing=timing)
    staged: Dict[str, Optional[torch.Tensor]] = {}
    with torch.cuda.stream(copy_stream):
        if copy_start is not None:
            copy_start.record(copy_stream)
        for name, source in sources.items():
            if source is None:
                staged[name] = None
                continue
            host = slot.tensors.get(name)
            if (
                host is None
                or tuple(host.shape) != tuple(source.shape)
                or host.dtype != source.dtype
            ):
                host = torch.empty(
                    tuple(source.shape),
                    dtype=source.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                slot.tensors[name] = host
            host.copy_(source, non_blocking=True)
            source.record_stream(copy_stream)
            original = originals[name]
            if isinstance(original, torch.Tensor) and original is not source:
                original.record_stream(copy_stream)
            staged[name] = host
        ready.record(copy_stream)
    staged_prediction = PredictionBundle(
        points=staged.get("points"),
        depth=staged.get("depth"),
        camera_poses=staged.get("camera_poses"),
    )
    if deferred:
        staged_prediction.raw = {
            "deferred_unprojection_backbone": str(deferred_backbone),
            "deferred_unprojection_extrinsic": staged.get(
                "deferred_unprojection_extrinsic"
            ),
            "deferred_unprojection_intrinsic": staged.get(
                "deferred_unprojection_intrinsic"
            ),
        }
    return (
        staged_prediction,
        ready,
        copy_start,
    )


def materialize_deferred_cpu_geometry(prediction: PredictionBundle) -> None:
    """Finish VGGT-style depth unprojection after asynchronous D2H completes."""
    backbone = prediction.raw.get("deferred_unprojection_backbone")
    if not backbone:
        return
    depth = prediction.depth
    extrinsic = prediction.raw.get("deferred_unprojection_extrinsic")
    intrinsic = prediction.raw.get("deferred_unprojection_intrinsic")
    for name, value in (
        ("depth", depth),
        ("extrinsic", extrinsic),
        ("intrinsic", intrinsic),
    ):
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise TypeError(
                f"Deferred CPU unprojection requires CPU tensor {name}, got {type(value)!r}"
            )
    from analysis.backbones.pose_geometry import unproject_depth_payload

    world_np = unproject_depth_payload(
        depth,
        extrinsic,
        intrinsic,
        backbone=str(backbone),
    )
    points = torch.from_numpy(world_np).to(dtype=torch.float32)
    if points.dim() == 4:
        points = points.unsqueeze(0)
    prediction.points = points


def _score_staged_prediction(
    *,
    ready_event: torch.cuda.Event,
    score_fn: Callable[[Any, PredictionBundle], Dict[str, float]],
    key: Any,
    prediction: PredictionBundle,
    timing: bool,
) -> _ScoreResult:
    ready_event.synchronize()
    score_t0 = time.perf_counter() if timing else 0.0
    materialize_deferred_cpu_geometry(prediction)
    metrics = score_fn(key, prediction)
    score_s = time.perf_counter() - score_t0 if timing else 0.0
    return _ScoreResult(metrics=metrics, score_s=score_s)


class AsyncPredictionScorePipeline:
    """Overlap CUDA inference with bounded parallel, ordered CPU scoring."""

    def __init__(
        self,
        *,
        device: torch.device,
        score_fn: Callable[[Any, PredictionBundle], Dict[str, float]],
        result_fn: Callable[[Any, Dict[str, float]], None],
        error_fn: Callable[[Any, Exception], None],
        timing: bool,
        score_workers: int = DEFAULT_ASYNC_SCORE_WORKERS,
        max_pending: int = DEFAULT_ASYNC_SCORE_MAX_PENDING,
        thread_name_prefix: str = "eval-score",
    ) -> None:
        if device.type != "cuda":
            raise ValueError("AsyncPredictionScorePipeline requires a CUDA device")
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        if score_workers < 1:
            raise ValueError("score_workers must be positive")
        self.device = device
        self.score_fn = score_fn
        self.result_fn = result_fn
        self.error_fn = error_fn
        self.timing = bool(timing)
        self.score_workers = int(score_workers)
        self.max_pending = int(max_pending)
        self.copy_stream = torch.cuda.Stream(device=device)
        self.executor = ThreadPoolExecutor(
            max_workers=self.score_workers, thread_name_prefix=thread_name_prefix
        )
        self.free_slots = deque(_PinnedScoreSlot() for _ in range(self.max_pending))
        self.pending: deque[_PendingScore] = deque()
        self.score_s = 0.0
        self.d2h_ms = 0.0
        self.producer_wait_s = 0.0
        self.max_pending_seen = 0
        self.closed = False

    def submit(self, key: Any, prediction: PredictionBundle) -> None:
        if self.closed:
            raise RuntimeError("Cannot submit to a closed async score pipeline")
        # Publish completed results even when the bounded queue is not full.
        # Keep callbacks on the producer thread and preserve submission order.
        self.drain_ready()
        if len(self.pending) >= self.max_pending:
            self._drain_one(count_producer_wait=True)
        slot = self.free_slots.popleft()
        try:
            staged, ready, copy_start = stage_prediction_for_cpu(
                prediction,
                slot=slot,
                device=self.device,
                copy_stream=self.copy_stream,
                timing=self.timing,
            )
        except Exception:
            self.free_slots.appendleft(slot)
            raise
        future = self.executor.submit(
            _score_staged_prediction,
            ready_event=ready,
            score_fn=self.score_fn,
            key=key,
            prediction=staged,
            timing=self.timing,
        )
        self.pending.append(
            _PendingScore(
                key=key,
                slot=slot,
                future=future,
                ready_event=ready,
                copy_start_event=copy_start,
            )
        )
        self.max_pending_seen = max(self.max_pending_seen, len(self.pending))

    def _drain_one(self, *, count_producer_wait: bool) -> None:
        item = self.pending.popleft()
        measure_wait = self.timing and count_producer_wait
        wait_t0 = time.perf_counter() if measure_wait else 0.0
        try:
            result = item.future.result()
            if measure_wait:
                self.producer_wait_s += time.perf_counter() - wait_t0
            self.score_s += result.score_s
            if self.timing and item.copy_start_event is not None:
                self.d2h_ms += float(
                    item.copy_start_event.elapsed_time(item.ready_event)
                )
            self.result_fn(item.key, result.metrics)
        except Exception as exc:
            if measure_wait:
                self.producer_wait_s += time.perf_counter() - wait_t0
            self.error_fn(item.key, exc)
        finally:
            self.free_slots.append(item.slot)

    def drain_ready(self) -> None:
        """Collect the completed prefix without waiting for unfinished scores."""
        while self.pending and self.pending[0].future.done():
            self._drain_one(count_producer_wait=False)

    def drain(self) -> None:
        while self.pending:
            self._drain_one(count_producer_wait=False)

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.drain()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.closed = True
