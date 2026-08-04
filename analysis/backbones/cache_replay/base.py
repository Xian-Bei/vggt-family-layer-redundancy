"""Protocol for backbone-specific cache-replay training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from analysis.backbones.cache_replay.common import ReplayResult
from analysis.backbones.contracts import TrainingHyperparams
from analysis.schema.v3 import PruneAction


class CacheReplayEngine(ABC):
    backbone_name: str

    @abstractmethod
    def validate_cache_row(self, row: Dict[str, Any], *, num_layers: int) -> None:
        ...

    @abstractmethod
    def build_train_model(
        self,
        *,
        hf_model: str,
        device: torch.device,
        drop_actions: Sequence[PruneAction],
        lora_actions: Sequence[PruneAction],
        frame_skip_intervals: Sequence[Tuple[int, int]],
        global_skip_intervals: Sequence[Tuple[int, int]],
        skipped_frame_idxs: Sequence[int],
        skipped_global_idxs: Sequence[int],
        prune: bool,
        inject_lora: bool,
    ) -> Tuple[nn.Module, List[nn.Parameter], int, int]:
        """Return (model, trainable_params, pre_prune_param_count, removed_param_count)."""

    @abstractmethod
    def replay_student_cache_first(
        self,
        model: nn.Module,
        get_layer_cpu: Callable[[int], torch.Tensor],
        num_layers: int,
        *,
        frame_skip_intervals: Sequence[Tuple[int, int]],
        global_skip_intervals: Sequence[Tuple[int, int]],
        lora_anchor_layers: Sequence[int],
        alignment_capture_layers: Sequence[int],
        seed_layer: int,
        seed_kind: str,
        initial_tokens: Optional[torch.Tensor],
        use_checkpoint: bool,
        device: torch.device,
        dummy_images: torch.Tensor,
    ) -> ReplayResult:
        ...

    @abstractmethod
    def build_target_batch_from_teacher(
        self,
        teacher_pred: Dict[str, torch.Tensor],
        *,
        images_batched: torch.Tensor,
        device: torch.device,
        dataset_name: str = "",
    ) -> Dict[str, torch.Tensor]:
        ...

    def build_optimizer(
        self,
        trainable: Sequence[nn.Parameter],
        *,
        args: Any,
    ) -> torch.optim.Optimizer:
        del trainable, args
        raise NotImplementedError(f"{self.backbone_name} does not define a training optimizer")

    def training_config(self) -> TrainingHyperparams:
        raise NotImplementedError(f"{self.backbone_name} does not define training hyperparameters")

    def prediction_loss(
        self,
        student_pred: Dict[str, torch.Tensor],
        target_batch: Dict[str, torch.Tensor],
        *,
        args: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del student_pred, target_batch, args
        raise NotImplementedError(f"{self.backbone_name} does not define prediction loss")

    def normalize_alignment_term(self, term: torch.Tensor) -> torch.Tensor:
        return term

    def regularization_loss(
        self,
        trainable: Sequence[nn.Parameter],
        *,
        args: Any,
        device: torch.device,
    ) -> torch.Tensor:
        del trainable, args
        return torch.zeros((), device=device)

    def update_optimizer_schedule(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        args: Any,
        global_step: int,
        total_steps: int,
    ) -> float:
        del optimizer, args, global_step, total_steps
        raise NotImplementedError(f"{self.backbone_name} does not define an optimizer schedule")

    def clip_gradients(
        self,
        trainable: Sequence[nn.Parameter],
        *,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        args: Any,
        device: torch.device,
    ) -> torch.Tensor:
        del trainable, optimizer, scaler, args
        return torch.tensor(0.0, device=device)

    def supports_dataset_targets(self) -> bool:
        return False

    def supports_periodic_eval(self) -> bool:
        return False

    def trainable_param_module(self, model: nn.Module) -> nn.Module:
        """Submodule used for parameter accounting logs."""
        return model

    def parameter_accounting_parameters(self, model: nn.Module) -> Iterable[nn.Parameter]:
        """Parameters counted in pruning summaries and training parameter logs."""
        return self.trainable_param_module(model).parameters()

    def train_replay_extra_kwargs(self, *, h: int, w: int) -> Dict[str, Any]:
        return {}

    def release_train_layer_cpu_cache(
        self,
        layer_cpu_cache: Dict[int, torch.Tensor],
        *,
        rules: Sequence[Any],
        prev_layers: Sequence[int],
        num_layers: int,
    ) -> None:
        del layer_cpu_cache, rules, prev_layers, num_layers
