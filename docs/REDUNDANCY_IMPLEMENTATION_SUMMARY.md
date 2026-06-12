# VGGT 冗余流水线 — 当前实现说明

本文件记录 **以 `run_scannet_redundancy.sh` 为唯一入口** 的当前主线实现：Stage 0 权重统计 → Stage 1 缓存与 GT 基线 → Stage 2 统一冗余分析 → Stage 3 预留。

## 1. 四阶段主线

| 阶段 | 模块 | 当前作用 |
| ------ | ------ | ----------- |
| Stage 0 | `analysis.redundancy.analyze_vggt_weights` | 仅分析 checkpoint 权重。 |
| Stage 1 | `analysis.redundancy.cache_scannet` | 全场景完整前向；保存 Stage2 所需缓存；直接完成 GT 评测与可选可视化导出。 |
| Stage 2 | `analysis.redundancy.analyze_scannet_cache` | 读取 Stage1 缓存，输出相似性分析与 skip-range GT loss。 |
| Stage 3 | 占位 | 当前不执行。 |

## 2. Stage 1 当前保存内容

Stage 1 对每个 scene 保存一个 `CACHE_DIR/<scene_id>.pt`。其中与当前主线直接相关的字段有：

- `aggregated_concat_per_layer`：
  每层完整的 `concat(frame_i, global_i)`，这是 Stage2 suffix replay 的边界状态缓存。
  Stage2 的默认相似性以及 legacy 相似性都会直接从这里恢复所需层输出。
- `patch_start_idx`：
  用于在 `aggregated_concat_per_layer` 中切分 special 与 patch token。
- `scene_scores`：
  一些轻量场景统计。

同时 Stage1 还会写：

- `manifest.json`：记录 `scene_id`、缓存文件名、`frame_ids` 等。
- `eval_scannet.json`：Stage1 GT 基线指标。
- 可选可视化目录 `CACHE_DIR/visualization/<scene_id>/...`。

Stage1 **不再** 主线写 `*_pred.pt`；GT 评测和可视化都直接基于内存 prediction payload 完成。

### 2.1 Stage 1 GT 指标口径

当前 Stage1 `eval_scannet.json` 中的主线 GT 指标包括：

- pose / geometry：
  `ate`、`are`、`rpe_rot`、`rpe_trans`、`chamfer_distance`
- dense depth：
  `depth_abs_rel`、`depth_delta1`

其中 depth 指标的实现口径为：

- prediction 直接使用当前 VGGT forward 输出的 `depth`
- GT depth **不** 反向 resize prediction，而是按与输入 RGB 完全一致的 `crop` 预处理对齐到网络输入坐标系
- GT depth 的有效范围与 FastVGGT 的 ScanNet 习惯保持一致：
  `65535 -> 0`，单位从毫米转米，并将 `<1e-3m` 或 `>10m` 的值视为无效
- depth metric 使用 **scale-only** 对齐（鲁棒迭代更新单个 scale），指标形式与 StreamVGGT 的 video-depth 评测保持兼容
- Stage1 aggregate 与 FastVGGT 保持一致：**对 scene 做直接平均**，而不是按有效像素数加权

## 3. Stage 2 相似性分析

### 3.1 默认主逻辑

当前默认 `PAIR_MAX_SPAN=1`。此时 Stage2 的相似性不再解释为“层间 `(i,j)` 相似”，而是：

- 对每一层 `i`，计算 `sim(frame_i, global_i)`；
- 用它衡量第 `i` 层 global block 对自身输入 `frame_i` 改动了多少。

实现上：

- 从 Stage1 缓存的 `aggregated_concat_per_layer` 中拆出每层的 `frame_i` 与 `global_i`；
- patch token 与 special token 分开算；
- 使用 `cosine`、`linear CKA`、可选 `PWCCA`、`neg-MSE`；
- 为了保持输出格式兼容，仍写成上三角 packed matrix，但**只有对角线有值**，非对角位置为 `NaN`。

因此默认热力图的主对角线才是有效信息。

### 3.2 Legacy 模式

如果显式设置 `PAIR_MAX_SPAN > 1`，则保留旧的 inter-layer 逻辑：

- 比较的是 Stage1 原始完整前向中不同层 global 输出之间的相似性；
- 由于相邻 global 输出之间夹着 frame block，这个量 **混合了 frame/global 的共同作用**；
- 因此代码和报告里都将它标记为 `legacy`，不能解释为 strict global-only redundancy。

## 4. Stage 2 skip-range GT loss

skip-range 分析当前是我们讨论后的实现：

- 对每个有效区间 `[i..j]`（闭区间，且 `j-i+1 <= SKIP_RANGE_MAX_SPAN`）；
- **不** 从头重跑整网；
- 而是复用 Stage1 缓存的前缀与边界状态，从第 `i` 层开始做 suffix replay。

更准确地说：

1. 直接复用 `0..i-1` 层缓存好的 `concat(frame, global)`；
2. 从第 `i` 层缓存里切出 `frame_i`；
3. 在 `[i..j]` 内把 global block 视作 `skip_global`（恒等透传）；
4. frame block 继续正常传播；
5. 从 `j+1` 起恢复正常的 frame/global 执行；
6. 最终重新构造完整 `aggregated_tokens_list`，再正常走 heads 与 GT eval。

所以 skip-range 分析真正对应的是“**只跳 global，保留 frame 正常传播**”。

此外，Stage2 replay 现在支持一个仅影响 skip 层的实验开关列表：

- `MODES=skip_global`：
  默认值之一，与当前主线 identity 行为一致；被 skip 的 global 层直接做恒等透传。
- `MODES=special_only`：
  在被 skip 的 global 层中，仅对 camera/register token 保留非对称 global attention；
  patch token 不做 patch-patch global attention，并原样透传到后续 frame block。

- `MODES=patch_special`：
  在被 skip 的 global 层中，special token 仍然对全部 token 做全局 attention；
  patch token 也会对 special token 做全局 attention，并继续经过正常的 residual 与 `norm2/mlp`；
  但 patch token 之间仍然不做 global attention。

- `MODES=special_only_nomlp`：
  与 `special_only` 相同的 attention 路径，但 special token 不再经过后续 `norm2/mlp` 分支。

- `MODES=patch_special_nomlp`：
  与 `patch_special` 相同的 attention 路径，但 special 与 patch 两个分支都不再经过后续 `norm2/mlp` 分支。

- `MODES=special_only,skip_global`：
  Stage2 会按列表中的 mode 依次各跑一次 skip-range；
  多模式时 heatmap / summary / visualization 会附加 mode 后缀，方便横向对比。

这个开关**只作用于 Stage2 skip-range replay**，不会改变 Stage1 缓存，也不会改变正常前向。

每个 GT 指标 `k` 的分数定义为：

`score_k(i,j) = -abs(pruned_metric_k - stage1_metric_k)`

其中当前主线

`k ∈ {ate, are, rpe_rot, rpe_trans, chamfer_distance, depth_abs_rel, depth_delta1}`。

解释方式保持不变：

- 越接近 `0`：和 Stage1 越接近，说明该 skip 区间影响越小、冗余越高；
- 越负：退化越大。

各 GT 指标分别出图，不做跨单位平均。Stage2 的 skip-range aggregate 也仍然按 **scene 直接平均**。

## 5. 关键默认参数

与 `run_scannet_redundancy.sh` 当前默认值一致：

| 变量 | 默认值 | 说明 |
| ------ | -------- | ------ |
| `MAX_TOKENS` | `4096` | `run_cache_redundancy.sh`：相似度计算时每层 token 子采样（0=全量）。 |
| `MAX_CONCAT_TOKENS` | `0` | Stage2 相似性前不二次下采样。 |
| `PAIR_MAX_SPAN` | `1` | 默认启用主逻辑 `sim(frame_i, global_i)`。 |
| `SKIP_RANGE_MAX_SPAN` | `6` | 最多连续跳过 6 个 global block。 |
| `MODES` | `special_only,skip_global` | Stage2 skip 层近似方式列表；支持 `skip_global`、`skip_both`、`special_only`、`special_only_nomlp`、`patch_special`、`patch_special_nomlp`（兼容别名 `patch_special_bipartite`）。单值保持旧单模式行为，多值会按 mode 分别运行并给输出加后缀。 |
| `EXPORT_VIS_STAGE1` | `0` | Stage1 可视化默认关闭。 |
| `EXPORT_VIS_STAGE2` | `0` | Stage2 可视化默认关闭。 |
| `SAVE_PER_SCENE_STAGE2` | `0` | Stage2 per-scene 报告默认关闭。 |
| `STAGE1_SANITY_CHECK` | `1` | Stage2 默认检查 frame selection 与 no-op GT 一致性。 |

## 6. 主要产物

- `CACHE_DIR/manifest.json`
- `CACHE_DIR/eval_scannet.json`
- `CACHE_DIR/<scene_id>.pt`
- `ANALYSIS_DIR/full_report.json`
- `ANALYSIS_DIR/summary_metrics.csv`
- `ANALYSIS_DIR/per_scene_scores.csv`
- `ANALYSIS_DIR/heatmap_*.png`
- 可选：
  `CACHE_DIR/visualization/<scene_id>/...`
  与
  `ANALYSIS_DIR/visualization/<scene_id>/skip_lXX_lYY/...`

## 7. 备注

- Stage2 相似性与 skip-range 是两条独立分析支线。
- 默认推荐只把 `PAIR_MAX_SPAN` 保持为 `1`；如果设大于 `1`，请把结果视为 legacy 辅助参考。
- GT 评测依赖 `evo`、`open3d`、`scipy` 以及完整有效的 ScanNet pose / mesh / depth 布局。
- 当前主线假设数据集已准备完整：若 Stage1/Stage2 评测时缺失 pose、mesh、depth 或对应 scene/eval row，
  会直接报错停止，而不是静默跳过。
- 当前 depth eval 面向 ScanNet 的实现选择是：
  **GT depth 跟随输入图像的 `crop` 预处理对齐到模型输入坐标系**，
  而不是把 prediction resize 回原始深度图分辨率。
- 可用 `analysis.redundancy.visualize_pair_attention` 基于 Stage1 cache 的 `aggregated_concat_per_layer`
  对指定 scene / layer / frame pair 做双向 attention 面板可视化（`i -> j` 与 `j -> i`）。
