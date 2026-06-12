#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Default backbones for all sub-scripts. Can be overridden from env.
ALL_BACKBONES="${BACKBONES:-vggt,pi3,fastvggt,vggt_omega}"
ALL_BACKBONES_SPACE_SEPARATED="${ALL_BACKBONES//,/ }"
CONDA_ENV="${CONDA_ENV:-vggt}"
DEVICE="${DEVICE:-cuda}"
MAX_SCENES="${MAX_SCENES:-0}"


if [[ "${DEVICE}" == cuda* ]]; then
  CUDA_AVAILABLE="$(
    conda run --no-capture-output -n "${CONDA_ENV}" python -c 'import torch; print("1" if torch.cuda.is_available() else "0")' 2>/dev/null || true
  )"
  if [[ "${CUDA_AVAILABLE}" != "1" ]]; then
    echo "[run.sh] DEVICE=${DEVICE} requested but CUDA is unavailable in conda env ${CONDA_ENV}; refusing to fall back to CPU." >&2
    echo "[run.sh] Check NVIDIA driver/device nodes before running this pipeline." >&2
    exit 1
  fi
fi


# run redundancy analysis, do not cache  -> 现在我们知道每个模型的冗余pattern不同且是其自身固有特性，与数据集无关
BACKBONES="${ALL_BACKBONES}" \
    RUN_SCANNET_DEV50=1 \
    RUN_7SCENES=1 \
    SPLIT_7SCENES=train \
    RUN_NRGBD=1 \
    CACHE_SCANNET_DEV50=0 \
    MAX_SCENES="${MAX_SCENES}" \
    CONDA_ENV="${CONDA_ENV}" \
    DEVICE="${DEVICE}" \
    SKIP_REDUNDANCY=0 \
    bash "run_cache_redundancy.sh"

# run eval with baseline (upper)
BACKBONES="${ALL_BACKBONES}" \
    RUN_SCANNET_DEV50=1 \
    RUN_SCANNET_TEST50=1 \
    RUN_7SCENES=1 \
    SPLIT_7SCENES=test \
    RUN_NRGBD=1 \
    EXPORT_VIS_2D=1 \
    EXPORT_VIS_3D=0 \
    PRUNE=0 \
    MAX_SCENES="${MAX_SCENES}" \
    CONDA_ENV="${CONDA_ENV}" \
    DEVICE="${DEVICE}" \
    bash "run_eval.sh"

# run skip eval on scannet dev50 using vggt -> 现在我们知道cka相似性结果与skip gt eval结果一致，最适合作为prune plan的metric
# BACKBONES=vggt \
#     SKIP_GT_EVAL_SCANNET_DEV50=1 \
#     SKIP_GT_EVAL_MAX_SPAN=0 \
#     bash "run_skip_eval.sh"


# generate prune plan based on scannet redundancy analysis
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  plan_root="analysis_outputs_v3/${bb}"
  rm -f "${plan_root}"/plan_t*_l*.json
  rm -f "${plan_root}"/plan_t*_l*_"${bb}"_triangles.png

  conda run -n "${CONDA_ENV}" python -m analysis.redundancy.build_prune_plan_v3 \
    --experiment "${bb}" \
    --dataset scannet_dev50 \
    --backbone "${bb}" \
    --metric cka \
    --similarity-threshold 0.9 \
    --max-triangles 2 \
    --max-lora-anchors 2 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --prune-plan-root "${plan_root}" \
    --viz-out-dir "${plan_root}"
done

# cache used units on scannet based on prune plan
# default: save union of units required by all plan_t*_l*.json under PRUNE_PLAN_ROOT.
# set PLAN_NAME=plan_t2_l1 to restrict to one specific plan.
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  cache_root="analysis_outputs_v3/${bb}"
  rm -rf "${cache_root}/scannet_dev50/cache" "${cache_root}/scannet/cache"

  BACKBONES="${bb}" \
      RUN_SCANNET_DEV50=1 \
      RUN_7SCENES=0 \
      RUN_NRGBD=0 \
      CACHE_SCANNET_DEV50=1 \
      SKIP_REDUNDANCY=1 \
      MAX_SCENES="${MAX_SCENES}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      PRUNE_PLAN_ROOT="${cache_root}" \
      bash "run_cache_redundancy.sh"
done


# run eval with prune plan (lower)
for bb in ${ALL_BACKBONES_SPACE_SEPARATED}; do
  BACKBONES="${bb}" \
      RUN_SCANNET_DEV50=1 \
      RUN_SCANNET_TEST50=1 \
      RUN_7SCENES=1 \
      SPLIT_7SCENES=test \
      RUN_NRGBD=1 \
      EXPORT_VIS_2D=1 \
      EXPORT_VIS_3D=0 \
      PRUNE=1 \
      PRUNE_PLAN_ROOT="analysis_outputs_v3/${bb}" \
      PLAN_NAME=plan_t1_l1 \
      MAX_SCENES="${MAX_SCENES}" \
      CONDA_ENV="${CONDA_ENV}" \
      DEVICE="${DEVICE}" \
      bash "run_eval.sh"
done
# generate merged summary across all backbones
conda run --no-capture-output -n "${CONDA_ENV}" python -m analysis.eval.summary_all_backbones_v3 \
    --root analysis_outputs_v3 \
    --backbones "${ALL_BACKBONES}"


EPOCHS="${EPOCHS:-50}"
EVAL_EPOCH="${EVAL_EPOCH:-5}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-10}"
# run lora training with prune plan
BACKBONES="${ALL_BACKBONES}" \
    DATASET=scannet_dev50 \
    PLAN_NAME=plan_t1_l1 \
    EPOCHS="${EPOCHS}" \
    MAX_SCENES="${MAX_SCENES}" \
    EVAL_EPOCH="${EVAL_EPOCH}" \
    SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH}" \
    CONDA_ENV="${CONDA_ENV}" \
    DEVICE="${DEVICE}" \
    EVAL_BOUNDS_ROOT=analysis_outputs_v3 \
    EVAL_SCANNET_DEV50=1 \
    EVAL_SCANNET_TEST50=1 \
    SPLIT_7SCENES=test \
    EVAL_7SCENES=1 \
    EVAL_NRGBD=1 \
    EVAL_AT_START=1 \
    bash "run_lora_skip_baseline.sh"


# run eval with lora finetuned checkpoint
BACKBONES="${ALL_BACKBONES}" \
    RUN_SCANNET_DEV50=1 \
    RUN_SCANNET_TEST50=1 \
    RUN_7SCENES=1 \
    SPLIT_7SCENES=test \
    RUN_NRGBD=1 \
    EXPORT_VIS_2D=1 \
    EXPORT_VIS_3D=0 \
    PRUNE=1 \
    PRUNE_PLAN_ROOT=analysis/prune_plan \
    CHECKPOINT=analysis_outputs_v3/vggt/scannet_dev50/Train_epoch_50/vggt/checkpoint_best.pt \
    MAX_SCENES=0 \
    bash "run_eval.sh"
