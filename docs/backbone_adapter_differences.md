# 骨干网络适配器差异（v3 以适配器为中心的流水线）

本文档说明各模型特有行为如何隔离在 `analysis/backbones/adapters/*` 中，以及共享的 `BackboneAdapter` 契约如何定义在 `analysis/backbones/contracts.py`。阶段脚本（`cache`、`train`、`eval`、编排器、shell 包装脚本）不得按骨干网络名称分支。

## 能力矩阵

| 能力 | VGGT | FastVGGT | VGGT-Omega | Pi3 |
|------|------|----------|------------|-----|
| 缓存（阶段 1） | frame_i + global_i | frame_i + global_i | frame_i + global_i（`inter_frame_blocks` 记作 global） | frame_i + global_i（decoder 配对） |
| 冗余 / 剪枝 | unit drop + LoRA | 同上 | 同上 | 同上 |
| 训练（阶段 3） | 缓存回放（`train_lora_skip_baseline`） | 缓存回放 | 缓存回放（`vggt_family` + 9D pose + `inter_frame_blocks`） | 缓存回放（同上，points/conf 蒸馏） |
| Eval ScanNet | VGGT 重建路径 | VGGT 重建路径 | points 适配器路径（深度+位姿反投影） | points 适配器路径 |
| Eval 7scenes/nrgbd | legacy 7andn criterion | legacy 7andn criterion | points + pose 适配器路径 | points + pose 适配器路径 |
| 默认 HF 模型 | `facebook/VGGT-1B` | `facebook/VGGT-1B` | 本地权重路径（如 `backbones/vggt-omega/vggt_omega_1b_512.pt`） | `yyfz233/Pi3` |
| PYTHONPATH | `backbones/vggt/training`、repo | `backbones/FastVGGT` + vggt training + repo | `backbones/vggt-omega`、repo | `backbones/Pi3`、repo |
| 缓存阶段 AMP | bf16 autocast，fp32 权重 | bf16 autocast，fp32 权重 | bf16 autocast，fp32 权重（模型内部按官方 head 精度） | bf16 autocast，fp32 权重 |
| 检查点恢复 | interval_rules + aggregator LoRA | 同上 | prune_plan + adapter 注入（`inter_frame_blocks`） | prune_plan + state_dict |

## 统一契约方法

| 方法 | 用途 |
|------|------|
| `default_hf_model()` | 供编排器 / shell 使用的 HF 仓库 id |
| `pythonpath_entries(repo_root)` | vendor 导入路径 |
| `forward_cache_batch(model, images, device)` | 单次缓存前向（autocast + extract + predict） |
| `cache_manifest_extra(model)` | manifest 字段（如 `num_heads`） |
| 剪枝计划 | 手填 `analysis/prune_plan/{backbone}.json`（可选 `build_prune_plan_v3` 从 redundancy 生成） |
| `train_from_cache(TrainFromCacheArgs)` | 阶段 3 训练实现 |
| `eval_scene(model, dataset_name, batch, ctx)` | 阶段 4 的逐场景指标 |
| `apply_checkpoint_structure(model, payload, prune_plan=...)` | 加载权重前恢复 skip/LoRA 拓扑 |

实现映射：

- **VGGT / FastVGGT**：`analysis/backbones/adapters/_vggt_base.py` 中的 `VGGTFamilyAdapterMixin`
- **VGGT-Omega**：`analysis/backbones/adapters/vggt_omega_adapter.py` 中的 `VGGTOmegaAdapter`
- **Pi3**：`analysis/backbones/adapters/pi3_adapter.py` 中的 `Pi3Adapter`
- **训练启动**：`analysis/backbones/train_from_cache.py`（子进程 → 共享 trainer）
- **训练引擎**：`analysis/backbones/cache_replay/{vggt_family,pi3,registry}.py`
- **评估辅助**：`analysis/backbones/eval_impl.py`
- **PYTHONPATH 引导**：`analysis/backbones/bootstrap.py`
- **Demo 对齐几何（pose/反投影）**：`analysis/backbones/pose_geometry.py` + `analysis/backbones/vendor_inference/`（从各 backbone 官方 demo 拷贝，无需 PYTHONPATH）
- **剪枝计划（阶段 3/4）**：`analysis/prune_plan/{backbone}.json`（手填模板；不再写入 `analysis_outputs_v3/.../prune/`）
- **Eval 模式（`run_eval.sh`）**：`PRUNE` 可为 `0,1`（逗号分隔多轮）；`0`→`upper`；`1` 无 `CHECKPOINT`→`lower`；`1` 有 checkpoint→`model`

## 缓存格式（破坏性变更）

旧布局对 VGGT 系列每个场景写入 **72 个文件**（24×frame + 24×global + 24×layer concat）。

新布局每个场景只写 **48 个文件**（仅 frame_i + global_i）。`[B,S,P,2C]` 的 concat 在 `load_scene_replay_layer_v3()` 加载时通过 `frame_i + global_i -> cat` 构建。旧版 `layer_{i}.pt` 文件会被拒绝，并给出明确的重新缓存提示。

## 各模型差异

### 输入 / 前向

- **VGGT 系列**：`VGGT` 类，`aggregator.frame_blocks` / `global_blocks`，在 block 输出上挂 hook；预测通过标准 VGGT head。
- **FastVGGT**：unit 语义相同；vendor 路径固定为 `backbones/FastVGGT`，不再提供备用 vendor 路径。回放拼接时 global 激活可能需要 `_reshape_global_to_match_frame()`。
- **VGGT-Omega**：24 个 `frame_blocks` + 24 个 `inter_frame_blocks`（其中部分层是 register attention）；适配器仍统一导出 `frame_i/global_i`。点云由 `depth + pose_enc->(extrinsic,intrinsic)` 反投影得到。
- **Pi3**：decoder block 映射为交替的 `frame_p` / `global_p`；预测来自 Pi3 dict（`points`、`conf`、`camera_poses`）。

### 数据类型 / autocast

所有骨干网络在前向时使用 **fp32 权重 + `autocast` 混合精度**（与官方 VGGT/Pi3 demo 一致）。阶段脚本不得调用 `model.to(bfloat16)`；`load_pretrained(..., dtype=None)` 在设备上保持 fp32 参数。需要 fp32 的 Pi3/VGGT head 在 vendor `forward()` 内以 `autocast(enabled=False)` 运行。

### 训练

所有骨干网络共用 **`analysis.train.train_lora_skip_baseline`**（`--backbone` 选择引擎）。`BackboneAdapter.train_from_cache()` 经 `analysis/backbones/train_from_cache.py` 子进程启动该脚本；interval 配置由剪枝计划推导。

骨干相关逻辑位于 `analysis/backbones/cache_replay/`：

- **vggt_family.py**：VGGT / FastVGGT / VGGT-Omega 的 decoder 回放、camera/depth 官方损失（Omega 使用 `encoding_to_camera` 解码 9D pose）
- **pi3.py**：Pi3 decoder 回放、points/conf 教师蒸馏

Pi3 训练使用 `--dtype float32`（权重保持 fp32，与官方 demo 一致）。

### 评估

- **VGGT 系列 ScanNet**：`infer_vggt_and_reconstruct` + pose/point/depth 指标。
- **VGGT-Omega ScanNet**：`forward_predictions`（depth + pose）并反投影点云后评估。
- **Pi3 ScanNet**：`forward_predictions` → 点云 vs GT mesh。
- **VGGT 系列 7scenes/nrgbd**：`_evaluate_one_scene_7andn` criterion 路径。
- **VGGT-Omega 7scenes/nrgbd**：points + pose 适配器路径。
- **Pi3 7scenes/nrgbd**：collated views，图像归一化到 [0,1]，从适配器输出计算 point + pose 指标。

## 阶段层：禁止骨干网络分支

重构后，以下入口点仅委托给 registry/adapter：

| 文件 | 调用 |
|------|------|
| `analysis/redundancy/cache_redundancy_v3.py` | `get_backbone`、`forward_cache_batch`、`cache_manifest_extra`、GPU 上算 redundancy |
| `analysis/redundancy/cache_layout_v3.py` | 仅 frame+global concat |
| `analysis/train/train_prune_lora_v3.py` | `adapter.train_from_cache` |
| `analysis/eval/eval_v3.py` | `adapter.eval_scene` |
| `analysis/train/checkpoint_v3.py` | `adapter.apply_checkpoint_structure` |
| `analysis/pipeline/run_minimal_all_backbones.py` | `default_hf_model`、`analysis/prune_plan`、bootstrap |
| `run_cache.sh`、`run_eval.sh`、`run_lora_skip_baseline.sh` | `python -m analysis.backbones.bootstrap` |

验证阶段脚本无分支：

```bash
rg 'if backbone ==|adapter\.name ==|if bb ==' analysis/ --glob '*.py'
# 预期：无匹配
```

## 最小回归命令

使用 conda 环境 `vggt`：

```bash
cd /home/fengyi/Proj/LaRe

# 完整最小流水线（所有骨干，1 场景 cache/train，每个 eval 数据集 1 场景）
# VGGT-Omega 需设置本地权重：
#   HF_MODEL=/home/fengyi/Proj/LaRe/backbones/vggt-omega/vggt_omega_1b_512.pt
conda run --no-capture-output -n vggt python -m analysis.pipeline.run_minimal_all_backbones \
  --experiment-base minimal_bb_refactor \
  --max-frames 5 \
  --max-steps 1 \
  --device cuda

# 单骨干缓存冒烟（24 层 VGGT 模型 unit 数应约 48）
conda run --no-capture-output -n vggt python -m analysis.redundancy.cache_redundancy_v3 \
  --experiment smoke_cache \
  --dataset scannet \
  --data-root data/scans_dev50 \
  --backbone vggt \
  --max-scenes 1 \
  --max-frames 5

# 统计缓存 unit 文件数（VGGT-1B 预期 48 = 24 frame + 24 global）
find analysis_outputs_v3/smoke_cache/scannet/cache -name '*.pt' | wc -l

# PYTHONPATH 引导
conda run --no-capture-output -n vggt python -m analysis.backbones.bootstrap --backbone fastvggt
```

## 迁移说明

- 含 `layer_{i}.pt` 的**旧缓存**不兼容；删除 `analysis_outputs_v3/<experiment>/` 并重新运行阶段 1。
- **旧检查点**若 payload 格式匹配，可通过 `apply_checkpoint_structure` 加载；刷新缓存后建议重新训练。
- `train_prune_lora_adapter_v3.py` 已弃用；所有骨干请使用 `train_prune_lora_v3.py`。
