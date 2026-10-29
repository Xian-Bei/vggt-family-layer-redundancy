# Redundancy Architecture and Tensor Path

本文记录 `analysis/redundancy` 当前依赖的公共执行路径。具体 vendor 模型结构由
`analysis/backbones/adapters/` 封装；redundancy 代码不应直接依赖某个模型的
`aggregator`、`decoder` 或 block 容器。

## 公共数据流

1. dataset adaptor 读取一个 scene，并返回 `EvalSampleBatch`。
2. backbone adaptor 的 `load_eval_images()` 将输入归一化为 `[S, 3, H, W]`。
3. cache/redundancy 路径调用 `forward_units_and_prediction()` 或
   `forward_cache_batch()`。
4. adaptor 将每个物理 block 输出规范化为 `unit_map[unit_id]`，形状
   `[B, S, P, C]`。
5. `meta["patch_start_idx"]` 标记 special/register token 与 patch token 的边界；
   `meta["aggregated_concat"]` 是可选的 backbone-specific 辅助结果。
6. `PredictionBundle` 保存标准化后的 pose、depth、points 和置信度字段；不支持的字段
   为 `None`。
7. redundancy graph 按 scene 计算选择的 pairwise metrics，并输出到：

```text
analysis_outputs_v3/<backbone>/<dataset>/redundancy/redundancy_graph_<region>.npz
```

默认 metrics 为 `cka,cosine,angular,inv_mse,cca`，默认 token regions 为
`special,patch,all`。单次 pruning plan 构建支持 `all`、`special`、`patch` 和
`special_patch_mean`；其中 `special_patch_mean` 在每个 dataset 内只平均 special 与
patch，再在 plan datasets 之间平均。主分析流程固定用 `all` 构建完整候选，并额外用
`special_patch_mean` 构建 CKA、Cosine、INV_MSE、CCA 的 Single/Double 候选；这四个
family 先只在 training datasets 上评测两组候选，再逐预算选择七指标归一化退化更小的
方案；held-out datasets 只评测最终 winner。Retention 不依赖
token region，ReplaceMe、ShortGPT 和 SIMPLER 固定使用 `all`。

## Unit 执行顺序

公共 unit 下标是 `layer_unit_catalog(model)` 返回列表的位置，不由名称公式推导：

- VGGT、VGGT-Omega、Pi3 当前按 `frame_0, global_0, ...` 执行；

因此缓存、selector、heatmap 和 recovery plan 必须同时保存 `unit_ids`，公共代码不得
假设偶数下标总是 frame unit。

## VGGT family 的具体映射

VGGT family 的 aggregator 会生成 camera/register/patch tokens，然后交替运行 frame
和 global attention blocks。adaptor 从 block 边界收集单独的 `[B,S,P,C]` activation；
vendor 返回的 `aggregated_tokens_list`（若使用）只是 adaptor 内部构造 prediction heads
输入的实现细节，不是公共 redundancy contract。

## 运行入口

当前没有 `analysis.redundancy.architecture_map` 命令。生成 redundancy graph 使用：

```bash
BACKBONES=vggt \
RUN=scannet_dev50 \
METRICS=cka,cosine,angular,inv_mse,cca \
SIM_TOKEN_REGIONS=special,patch,all \
bash scripts/run_cache_redundancy.sh
```

该命令会执行 GPU 模型前向；按项目约定，应在沙箱外的 `vggt` conda 环境中运行。
