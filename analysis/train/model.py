from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn

from vggt.models.vggt import VGGT

ModelName = Literal["Ours", "Baseline"]
BackendName = Literal["vggt", "fastvggt"]


def load_interval_rules(path: Union[str, Path]) -> List[IntervalRule]:
    """
    Load interval rules from the same json schema used by training:
    - either a list of objects, or {"intervals": [...]}.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing interval config file: {p}")
    import json

    payload = json.loads(p.read_text(encoding="utf-8"))
    rules_raw: Any = payload.get("intervals") if isinstance(payload, dict) else payload
    if not isinstance(rules_raw, list) or not rules_raw:
        raise ValueError("Interval config must be a non-empty list (or {\"intervals\": [...]}).")
    rules = interval_rules_from_payload(rules_raw)
    rules = sorted(rules, key=lambda x: (int(x.l), int(x.r)))
    for i in range(1, len(rules)):
        pl, pr = int(rules[i - 1].l), int(rules[i - 1].r)
        l, r = int(rules[i].l), int(rules[i].r)
        if l <= pr:
            raise ValueError(f"Overlapped intervals in config: [{pl},{pr}] and [{l},{r}]")
    return rules


@dataclass(frozen=True)
class IntervalRule:
    l: int
    r: int
    skip_frame_in_interval: bool

    @property
    def prev_layer(self) -> int:
        return int(self.l) - 1


def lora_injection_plan_from_rules(
    rules: Sequence[IntervalRule],
) -> Tuple[List[int], List[int]]:
    frame_idxs: set[int] = set()
    global_idxs: set[int] = set()
    for rule in rules:
        pidx = int(rule.prev_layer)
        global_idxs.add(pidx)
        frame_idxs.add(pidx)
        if not rule.skip_frame_in_interval:
            frame_idxs.update(range(int(rule.l), int(rule.r) + 1))
    return sorted(frame_idxs), sorted(global_idxs)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        in_f = int(base.in_features)
        out_f = int(base.out_features)
        self.lora_A = nn.Linear(in_f, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class SkipIdentityBlock(nn.Module):
    """Identity block that accepts the transformer `pos` kwarg."""

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        return x


def inject_lora_attn(block: nn.Module, rank: int, alpha: float, dropout: float) -> List[nn.Parameter]:
    trainable: List[nn.Parameter] = []
    attn = getattr(block, "attn", None)
    if attn is None:
        return trainable
    for name in ("qkv", "proj"):
        mod = getattr(attn, name, None)
        if isinstance(mod, nn.Linear):
            lora_mod = LoRALinear(mod, rank=rank, alpha=alpha, dropout=dropout)
            setattr(attn, name, lora_mod)
            trainable.extend([p for p in lora_mod.parameters() if p.requires_grad])
    return trainable


def replace_skipped_layers_with_identity(
    model: VGGT,
    *,
    frame_intervals: Sequence[Tuple[int, int]],
    global_intervals: Sequence[Tuple[int, int]],
) -> None:
    depth = len(model.aggregator.frame_blocks)
    for l, r in frame_intervals:
        for idx in range(int(l), int(r) + 1):
            if idx < 0 or idx >= depth:
                raise SystemExit(f"Skip layer index {idx} out of range for model depth {depth}.")
            model.aggregator.frame_blocks[idx] = SkipIdentityBlock()
    for l, r in global_intervals:
        for idx in range(int(l), int(r) + 1):
            if idx < 0 or idx >= depth:
                raise SystemExit(f"Skip layer index {idx} out of range for model depth {depth}.")
            model.aggregator.global_blocks[idx] = SkipIdentityBlock()


def build_skipped_layer_indices(intervals: Sequence[Tuple[int, int]]) -> List[int]:
    out: List[int] = []
    for l, r in intervals:
        out.extend(range(int(l), int(r) + 1))
    return sorted(set(out))


def count_unique_params(modules: Sequence[nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for m in modules:
        for p in m.parameters(recurse=True):
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            total += int(p.numel())
    return total


_ALLOWED_INTERVAL_KEYS = frozenset({"l", "r", "skip_frame_in_interval"})


def interval_rules_from_payload(rules_raw: Sequence[Dict[str, Any]]) -> List[IntervalRule]:
    parsed: List[IntervalRule] = []
    for i, item in enumerate(rules_raw):
        if not isinstance(item, dict):
            raise ValueError(f"interval_rules[{i}] must be an object.")
        unknown = set(item.keys()) - _ALLOWED_INTERVAL_KEYS
        if unknown:
            raise ValueError(
                f"interval_rules[{i}] has unsupported keys: {sorted(unknown)}; "
                f"allowed: {sorted(_ALLOWED_INTERVAL_KEYS)}."
            )
        if "l" not in item or "r" not in item:
            raise ValueError(f"interval_rules[{i}] must provide both 'l' and 'r'.")
        parsed.append(
            IntervalRule(
                l=int(item["l"]),
                r=int(item["r"]),
                skip_frame_in_interval=bool(item.get("skip_frame_in_interval", True)),
            )
        )
    return parsed


def intervals_from_rules(rules: Sequence[IntervalRule]) -> List[Tuple[int, int]]:
    return [(int(x.l), int(x.r)) for x in rules]


def global_skip_intervals_from_rules(rules: Sequence[IntervalRule]) -> List[Tuple[int, int]]:
    return intervals_from_rules(rules)


def frame_skip_intervals_from_rules(rules: Sequence[IntervalRule]) -> List[Tuple[int, int]]:
    return [(int(x.l), int(x.r)) for x in rules if bool(x.skip_frame_in_interval)]


def is_ours_train_checkpoint(payload: Dict[str, Any]) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("args"), dict)
        and isinstance(payload.get("interval_rules"), list)
        and isinstance(payload.get("model_state_dict"), dict)
    )


def build_baseline(
    *,
    hf_model: str,
    device: Union[str, torch.device],
    eval_mode: bool = True,
) -> VGGT:
    model = VGGT.from_pretrained(str(hf_model)).to(device)
    if eval_mode:
        model.eval()
    return model


def build_ours(
    *,
    hf_model: str,
    rules: Sequence[IntervalRule],
    device: Union[str, torch.device],
    enable_lora: bool,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    eval_mode: bool = True,
) -> VGGT:
    model = VGGT.from_pretrained(str(hf_model)).to(device)
    if eval_mode:
        model.eval()

    frame_skip_intervals = frame_skip_intervals_from_rules(rules)
    global_skip_intervals = global_skip_intervals_from_rules(rules)
    replace_skipped_layers_with_identity(
        model,
        frame_intervals=frame_skip_intervals,
        global_intervals=global_skip_intervals,
    )

    if enable_lora:
        frame_lora_idxs, global_lora_idxs = lora_injection_plan_from_rules(rules)
        for pidx in frame_lora_idxs:
            inject_lora_attn(
                model.aggregator.frame_blocks[pidx],
                rank=int(lora_rank),
                alpha=float(lora_alpha),
                dropout=float(lora_dropout),
            )
        for pidx in global_lora_idxs:
            inject_lora_attn(
                model.aggregator.global_blocks[pidx],
                rank=int(lora_rank),
                alpha=float(lora_alpha),
                dropout=float(lora_dropout),
            )
    return model


def load_ours_from_train_checkpoint(
    payload: Dict[str, Any],
    *,
    device: Union[str, torch.device],
    eval_mode: bool = True,
) -> VGGT:
    if not is_ours_train_checkpoint(payload):
        raise ValueError("Not an Ours training checkpoint payload.")
    args = payload["args"]
    hf_model = str(args.get("hf_model", "")).strip()
    if not hf_model:
        raise ValueError("Checkpoint args.hf_model is required to rebuild model.")
    rules = interval_rules_from_payload(payload["interval_rules"])
    model = build_ours(
        hf_model=hf_model,
        rules=rules,
        device=device,
        enable_lora=True,
        lora_rank=int(args.get("lora_rank", 8)),
        lora_alpha=float(args.get("lora_alpha", 16.0)),
        lora_dropout=float(args.get("lora_dropout", 0.0)),
        eval_mode=eval_mode,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def resolve_checkpoint_path(train_dir: Union[str, Path], epoch: Union[int, str]) -> Path:
    td = Path(train_dir)
    ep = int(epoch)
    return td / "checkpoints" / f"epoch_{ep}.pt"


def build_model(
    *,
    model_name: ModelName,
    backend: BackendName,
    hf_model: str,
    device: Union[str, torch.device],
    interval_rules: Optional[Sequence[IntervalRule]] = None,
    ckpt_path: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    epoch: Optional[int] = None,
    eval_mode: bool = True,
) -> VGGT:
    """
    Global unique model definition API.

    Notes:
    - `backend` is used for validation / logging; actual backend implementation is selected via PYTHONPATH
      (see run_eval.sh / run_lora_skip_baseline.sh set_backend_pythonpath).
    - Baseline: full model, no skip, no LoRA.
    - Ours:
      - with checkpoint: skip + LoRA + load weights
      - without checkpoint: skip only (no LoRA) = lower bound
    """
    if backend not in ("vggt", "fastvggt"):
        raise ValueError(f"Unsupported backend={backend}")
    if model_name not in ("Ours", "Baseline"):
        raise ValueError(f"Unsupported model_name={model_name}")
    hf_model = str(hf_model).strip()
    if not hf_model:
        raise ValueError("hf_model is required.")

    resolved_ckpt: Optional[Path] = None
    if ckpt_path is not None and str(ckpt_path).strip():
        resolved_ckpt = Path(ckpt_path)
    elif train_dir is not None and epoch is not None:
        resolved_ckpt = resolve_checkpoint_path(train_dir, epoch)

    if model_name == "Baseline":
        if resolved_ckpt is None:
            return build_baseline(hf_model=hf_model, device=device, eval_mode=eval_mode)
        payload = torch.load(resolved_ckpt, map_location="cpu")
        state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        model = build_baseline(hf_model=hf_model, device=device, eval_mode=eval_mode)
        _ = model.load_state_dict(state, strict=False)
        return model

    # Ours
    if resolved_ckpt is not None:
        payload = torch.load(resolved_ckpt, map_location="cpu")
        return load_ours_from_train_checkpoint(payload, device=device, eval_mode=eval_mode)

    if interval_rules is None:
        raise ValueError("Ours without checkpoint requires interval_rules (to apply skip-only lower bound).")
    return build_ours(
        hf_model=hf_model,
        rules=interval_rules,
        device=device,
        enable_lora=False,
        eval_mode=eval_mode,
    )

