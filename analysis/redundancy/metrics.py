from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

SUPPORTED_SIMILARITY_METRICS = frozenset(
    {"cosine", "angular", "cka", "inv_mse", "cca", "pwcca"}
)
DEFAULT_SIMILARITY_METRICS = ("cosine", "angular", "cka", "inv_mse")


def upper_triangle_length(num_layers: int) -> int:
    return num_layers * (num_layers + 1) // 2


def upper_triangle_pack(mat: torch.Tensor) -> List[float]:
    """Row-major upper triangle including diagonal: (0,0)…(0,L-1), (1,1)…, (L-1,L-1)."""
    L = mat.shape[0]
    if mat.shape != (L, L):
        raise ValueError(f"Expected square matrix, got {tuple(mat.shape)}")
    out: List[float] = []
    for i in range(L):
        for j in range(i, L):
            out.append(float(mat[i, j].item()))
    return out


def upper_triangle_unpack_for_heatmap(packed: List[float], L: int, fill_value: float) -> List[List[float]]:
    """Full L×L: entries with j >= i from ``packed``; entries with j < i set to ``fill_value``."""
    expected = upper_triangle_length(L)
    if len(packed) != expected:
        raise ValueError(f"packed length {len(packed)} != expected {expected} for L={L}")
    grid: List[List[float]] = [[fill_value] * L for _ in range(L)]
    k = 0
    for i in range(L):
        for j in range(i, L):
            grid[i][j] = packed[k]
            k += 1
    return grid


def upper_packed_to_symmetric_matrix(packed: List[float], L: int) -> List[List[float]]:
    """Recover full symmetric L×L from row-major upper triangle (incl. diagonal)."""
    expected = upper_triangle_length(L)
    if len(packed) != expected:
        raise ValueError(f"packed length {len(packed)} != expected {expected} for L={L}")
    grid: List[List[float]] = [[0.0] * L for _ in range(L)]
    k = 0
    for i in range(L):
        for j in range(i, L):
            v = packed[k]
            k += 1
            grid[i][j] = grid[j][i] = v
    return grid


def mean_offdiag_from_upper_packed(packed: List[float], L: int) -> float:
    """Mean over strict upper triangle (i < j); diagonal excluded."""
    s = 0.0
    c = 0
    k = 0
    for i in range(L):
        for j in range(i, L):
            v = packed[k]
            k += 1
            if i != j:
                s += v
                c += 1
    return s / max(c, 1)


def _flatten_tokens(tokens: torch.Tensor) -> torch.Tensor:
    # [B, S, P, C] -> [B*S*P, C]
    return tokens.reshape(-1, tokens.shape[-1]).float()


def _flatten_tokens_pair(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten two activations to ``[N, C]`` and require **identical** shape.

    Different layers in the same forward pass should have the same ``(B,S,P,C)``;
    if not, caching or indexing is inconsistent and should fail loudly.
    """
    x = _flatten_tokens(x)
    y = _flatten_tokens(y)
    if x.shape != y.shape:
        raise AssertionError(
            "Pairwise metrics expect identical token tensors after flatten (same N and C); "
            f"got {tuple(x.shape)} vs {tuple(y.shape)}."
        )
    return x, y


def sample_cosine_similarity_tensor(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean row-wise cosine similarity after flattening to ``[N,C]`` (differentiable)."""
    x, y = _flatten_tokens_pair(x, y)
    x = x / (x.norm(dim=-1, keepdim=True) + eps)
    y = y / (y.norm(dim=-1, keepdim=True) + eps)
    return (x * y).sum(dim=-1).mean()


def sample_cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    return float(sample_cosine_similarity_tensor(x, y, eps=eps).item())


def sample_angular_distance_tensor(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """ReplaceMe angular distance: mean row-wise ``acos(cosine) / pi``."""
    x, y = _flatten_tokens_pair(x, y)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(eps)
    cosine = (x * y).sum(dim=-1).clamp(min=-1.0, max=1.0)
    return (torch.acos(cosine) / torch.pi).mean()


def sample_angular_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    return float(sample_angular_distance_tensor(x, y, eps=eps).item())


def linear_cka_value_tensor(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Linear CKA similarity in ``(0,1]`` (differentiable).

    Denominator uses ``sqrt(||X^T X||_F^2 * ||Y^T Y||_F^2).clamp_min(eps)``, matching the
    historical training loss in ``train_lora_skip_baseline`` (``1 - CKA`` form).
    """
    x, y = _flatten_tokens_pair(x, y)

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    xty = x.t() @ y
    xtx = x.t() @ x
    yty = y.t() @ y
    num = (xty * xty).sum()
    den = torch.sqrt((xtx * xtx).sum() * (yty * yty).sum()).clamp_min(eps)
    return num / den


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    return float(linear_cka_value_tensor(x, y, eps=eps).item())


def token_mean_squared_error_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Scalar MSE after token flatten to ``[N,C]``."""
    x, y = _flatten_tokens_pair(x, y)
    return F.mse_loss(x, y)


def _normalize_similarity_metric_names(
    metric_names: Optional[Iterable[str]],
    *,
    compute_pwcca: bool = False,
) -> Tuple[str, ...]:
    if metric_names is None:
        names = list(DEFAULT_SIMILARITY_METRICS)
        if compute_pwcca:
            names.append("pwcca")
    else:
        names = [str(name).strip().lower() for name in metric_names if str(name).strip()]
        if compute_pwcca and "pwcca" not in names:
            names.append("pwcca")
    unknown = sorted(set(names).difference(SUPPORTED_SIMILARITY_METRICS))
    if unknown:
        raise ValueError(
            f"Unsupported similarity metric(s): {unknown}; "
            f"allowed: {sorted(SUPPORTED_SIMILARITY_METRICS)}"
        )
    out: List[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return tuple(out)


def _svd_reduce_for_cca(x: torch.Tensor, keep_components: int) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    if x.numel() == 0:
        return x
    _u, _s, vh = torch.linalg.svd(x, full_matrices=False)
    k = min(max(1, int(keep_components)), int(vh.shape[0]))
    # Return principal-component scores [N, k]. This is the SVCCA/PWCCA
    # pre-step and keeps the covariance eigensystems small enough for layer scans.
    return x @ vh[:k].t()


def _cca_decomposition(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    n = x.shape[0]
    cxx = (x.t() @ x) / max(n - 1, 1) + eps * torch.eye(
        x.shape[1], device=x.device, dtype=x.dtype
    )
    cyy = (y.t() @ y) / max(n - 1, 1) + eps * torch.eye(
        y.shape[1], device=y.device, dtype=y.dtype
    )
    cxy = (x.t() @ y) / max(n - 1, 1)

    ex, vx = torch.linalg.eigh(cxx)
    ey, vy = torch.linalg.eigh(cyy)
    ex = ex.clamp_min(eps)
    ey = ey.clamp_min(eps)

    inv_sqrt_x = vx @ torch.diag(ex.rsqrt()) @ vx.t()
    inv_sqrt_y = vy @ torch.diag(ey.rsqrt()) @ vy.t()
    tmat = inv_sqrt_x @ cxy @ inv_sqrt_y
    u, corr, _vh = torch.linalg.svd(tmat, full_matrices=False)
    x_dirs = inv_sqrt_x @ u
    return corr, x_dirs


def _cca_singular_values(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    corr, _x_dirs = _cca_decomposition(x, y, eps=eps)
    return corr


def inverse_mean_squared_error(x: torch.Tensor, y: torch.Tensor) -> float:
    """Map activation MSE to a higher-is-more-similar score in ``(0, 1]``."""
    mse = token_mean_squared_error_tensor(x, y)
    return float((1.0 / (1.0 + mse)).item())


def cca_similarity(x: torch.Tensor, y: torch.Tensor, keep_components: int = 64) -> float:
    """SVCCA-style mean canonical correlation after SVD dimensionality reduction."""
    x, y = _flatten_tokens_pair(x, y)
    x_proj = _svd_reduce_for_cca(x, keep_components)
    y_proj = _svd_reduce_for_cca(y, keep_components)
    corr = _cca_singular_values(x_proj, y_proj).clamp(0.0, 1.0)
    if corr.numel() == 0:
        return 0.0
    return float(corr.mean().item())


def pwcca_similarity(x: torch.Tensor, y: torch.Tensor, keep_components: int = 64) -> float:
    """Projection-weighted CCA after the same SVD reduction used by :func:`cca_similarity`."""
    x, y = _flatten_tokens_pair(x, y)

    x_proj = _svd_reduce_for_cca(x, keep_components)
    y_proj = _svd_reduce_for_cca(y, keep_components)

    corr, x_dirs = _cca_decomposition(x_proj, y_proj)
    corr = corr.clamp(0.0, 1.0)
    if corr.numel() == 0:
        return 0.0
    x_centered = x_proj - x_proj.mean(dim=0, keepdim=True)
    x_canonical = x_centered @ x_dirs
    weights = x_canonical.abs().sum(dim=0)
    weights = weights / weights.sum().clamp_min(1e-8)
    return float((weights * corr).sum().item())


def pairwise_similarity_matrices(
    layer_tokens: Tuple[torch.Tensor, ...],
    metric_seconds: Optional[Dict[str, float]] = None,
    compute_pwcca: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    *,
    metric_names: Optional[Iterable[str]] = None,
    progress: bool = False,
    progress_desc: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Pairwise similarity matrices between layer token tensors.

    If ``device`` is set (e.g. ``cuda``), layer tensors are moved to that device once for
    all pairs; metric matrices are returned on **CPU** float64.

    ``metric_names`` selects from cosine, angular, cka, inv_mse, cca, pwcca. If omitted,
    the default is cosine / angular / CKA / INV_MSE, plus PWCCA when
    ``compute_pwcca=True``.

    If ``metric_seconds`` is a dict, accumulate wall seconds per metric over all pairs.
    """
    if len(layer_tokens) == 0:
        return {}
    requested = _normalize_similarity_metric_names(
        metric_names,
        compute_pwcca=compute_pwcca,
    )

    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
        layer_tokens = tuple(x.to(dev, non_blocking=True) for x in layer_tokens)
    else:
        dev = layer_tokens[0].device

    num_layers = len(layer_tokens)
    mats = {
        name: torch.zeros(num_layers, num_layers, device=dev, dtype=torch.float64)
        for name in requested
    }

    def _acc(name: str, t0: float) -> None:
        if metric_seconds is not None:
            metric_seconds[name] = metric_seconds.get(name, 0.0) + (time.perf_counter() - t0)

    pairs_total = num_layers * (num_layers + 1) // 2
    pbar = None
    if progress and pairs_total > 0:
        import sys

        from tqdm import tqdm

        pbar = tqdm(
            total=pairs_total,
            desc=progress_desc or "pairwise similarity",
            unit="pair",
            leave=False,
            dynamic_ncols=True,
            file=sys.stderr,
        )
    try:
        for i in range(num_layers):
            for j in range(i, num_layers):
                t0 = time.perf_counter()
                if "cosine" in mats:
                    cos_v = sample_cosine_similarity(layer_tokens[i], layer_tokens[j])
                    mats["cosine"][i, j] = mats["cosine"][j, i] = cos_v
                    _acc("cosine", t0)

                t0 = time.perf_counter()
                if "angular" in mats:
                    angular_v = sample_angular_distance(layer_tokens[i], layer_tokens[j])
                    mats["angular"][i, j] = mats["angular"][j, i] = angular_v
                    _acc("angular", t0)

                t0 = time.perf_counter()
                if "cka" in mats:
                    cka_v = linear_cka(layer_tokens[i], layer_tokens[j])
                    mats["cka"][i, j] = mats["cka"][j, i] = cka_v
                    _acc("cka", t0)

                if "cca" in mats:
                    t0 = time.perf_counter()
                    cca_v = cca_similarity(layer_tokens[i], layer_tokens[j])
                    _acc("cca", t0)
                    mats["cca"][i, j] = mats["cca"][j, i] = cca_v

                if "pwcca" in mats:
                    t0 = time.perf_counter()
                    pw_v = pwcca_similarity(layer_tokens[i], layer_tokens[j])
                    _acc("pwcca", t0)
                    mats["pwcca"][i, j] = mats["pwcca"][j, i] = pw_v

                t0 = time.perf_counter()
                if "inv_mse" in mats:
                    im_v = inverse_mean_squared_error(layer_tokens[i], layer_tokens[j])
                    _acc("inv_mse", t0)
                    mats["inv_mse"][i, j] = mats["inv_mse"][j, i] = im_v

                if pbar is not None:
                    pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    return {name: mat.cpu() for name, mat in mats.items()}


def paired_layer_similarity_matrices(
    x_layers: Tuple[torch.Tensor, ...],
    y_layers: Tuple[torch.Tensor, ...],
    metric_seconds: Optional[Dict[str, float]] = None,
    compute_pwcca: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    *,
    metric_names: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Per-layer similarity between paired tensors ``(x_i, y_i)``.

    The returned matrices keep the same ``[L, L]`` shape / upper-triangle packing
    convention as the legacy inter-layer heatmaps, but only the diagonal entries
    are populated; off-diagonal entries are ``NaN`` because this mode measures
    *within-layer* change (for this project: ``frame_i`` vs ``global_i``).
    """
    if len(x_layers) == 0:
        return {}
    if len(x_layers) != len(y_layers):
        raise ValueError(f"Expected same number of x/y layers, got {len(x_layers)} vs {len(y_layers)}")
    requested = _normalize_similarity_metric_names(
        metric_names,
        compute_pwcca=compute_pwcca,
    )

    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
        x_layers = tuple(x.to(dev, non_blocking=True) for x in x_layers)
        y_layers = tuple(y.to(dev, non_blocking=True) for y in y_layers)
    else:
        dev = x_layers[0].device

    num_layers = len(x_layers)
    mats = {
        name: torch.full((num_layers, num_layers), float("nan"), device=dev, dtype=torch.float64)
        for name in requested
    }

    def _acc(name: str, t0: float) -> None:
        if metric_seconds is not None:
            metric_seconds[name] = metric_seconds.get(name, 0.0) + (time.perf_counter() - t0)

    for i in range(num_layers):
        t0 = time.perf_counter()
        if "cosine" in mats:
            cos_v = sample_cosine_similarity(x_layers[i], y_layers[i])
            _acc("cosine", t0)
            mats["cosine"][i, i] = cos_v

        t0 = time.perf_counter()
        if "cka" in mats:
            cka_v = linear_cka(x_layers[i], y_layers[i])
            _acc("cka", t0)
            mats["cka"][i, i] = cka_v

        if "cca" in mats:
            t0 = time.perf_counter()
            cca_v = cca_similarity(x_layers[i], y_layers[i])
            _acc("cca", t0)
            mats["cca"][i, i] = cca_v

        if "pwcca" in mats:
            t0 = time.perf_counter()
            pw_v = pwcca_similarity(x_layers[i], y_layers[i])
            _acc("pwcca", t0)
            mats["pwcca"][i, i] = pw_v

        t0 = time.perf_counter()
        if "inv_mse" in mats:
            im_v = inverse_mean_squared_error(x_layers[i], y_layers[i])
            _acc("inv_mse", t0)
            mats["inv_mse"][i, i] = im_v

    return {name: mat.cpu() for name, mat in mats.items()}


def stable_rank(weight: torch.Tensor, eps: float = 1e-8) -> float:
    w = weight.float()
    frob_sq = float((w * w).sum().item())
    sigma_max = float(torch.linalg.svdvals(w).max().item())
    return frob_sq / (sigma_max * sigma_max + eps)


def effective_rank(weight: torch.Tensor, eps: float = 1e-8) -> float:
    s = torch.linalg.svdvals(weight.float())
    p = s / (s.sum() + eps)
    entropy = -torch.sum(p * torch.log(p + eps))
    return float(torch.exp(entropy).item())


def spectral_stats(weight: torch.Tensor) -> Dict[str, float]:
    return {
        "stable_rank": stable_rank(weight),
        "effective_rank": effective_rank(weight),
    }
