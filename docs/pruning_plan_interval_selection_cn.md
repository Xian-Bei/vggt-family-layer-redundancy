# Pruning Plan 选择与 Baseline 复现说明（当前实现）

本文档说明步骤 6 如何生成 pruning plan、步骤 7 如何完成物理方案评测，并重点区分：

1. 论文或官方仓库定义的方法；
2. LaRe 当前真正执行的选择逻辑；
3. 从 LLM/2D 模型迁移到 3D backbone 时不可避免的输入适配；
4. 当前仍不属于严格官方复现的部分。

文档以当前代码为准。主要实现位于：

- `analysis/prune_plan/build_prune_plan_v3.py`
- `analysis/prune_plan/__init__.py`
- `analysis/prune_plan/stage7.py`
- `analysis/prune_plan/eval_plots.py`
- `analysis/redundancy/metrics.py`
- `analysis/redundancy/graph_io.py`

这里的“层”统一称为 **unit**。例如 VGGT 的 unit 按实际执行顺序排列：

```text
frame_0, global_0, frame_1, global_1, ...
```

删层预算 `n` 表示实际删除的 unit 总数，不表示区间跨度或区间数量。

## 1. 当前 strategy 结构

当前实现支持以下 strategy key：

```text
ours_single
ours_double
gt_single
gt_double
replaceme_single
replaceme_double
shortgpt
simpler
cosine_single
cosine_double
inv_mse_single
inv_mse_double
```

其中：

- `ours_single`、`gt_single`、`replaceme_single` 是单区间/单 block 版本；
- `ours_double`、`cosine_double`、`inv_mse_double`、`gt_double`
  是“在单区间和双区间之间自适应选择”的版本；
- `replaceme_double` 对应官方 `num_A=2`，强制先选择两个等长 block；
- `shortgpt` 使用官方无区间约束的全局 BI 排序，可能形成任意多个区间；
- `simpler` 是当前 pipeline 中的 suffix sweep 版本；
- `cosine_single/double`、`inv_mse_single/double` 是 Ours 的内部相似度
  ablation，不应描述成对应论文的严格官方方法。

`done.sh` 默认包含：

```text
gt_single,gt_double,
ours_single,ours_double,
replaceme_single,replaceme_double,
shortgpt,simpler
```

环境变量 `PRUNE_STRATEGY` 和 `PLAN_STRATEGY` 可以覆盖实际参与生成和评测的方法。`TODO.sh` 可能为当前实验临时注释步骤或缩小默认方法集合，因此论文实验应同时记录实际环境变量。

## 2. 步骤 6 的输入和输出

典型设置为：

```text
PRUNE_DATASET=scannet_dev50,nuscenes_train
PRUNE_TOKEN_REGION=mean
metric=cka
```

步骤 6 读取：

- plan-building datasets 的 redundancy graph；
- `scannet_dev50`、`nuscenes_train` 的单区间 `eval/pruning/<selector>/summary.json`，用于 GT；
- backbone 的 unit catalog 和参数统计。

输出：

```text
analysis_outputs_v3/<backbone>/prune_plan_all.json
analysis_outputs_v3/<backbone>/prune_plan_<backbone>_drop_grid.png
```

步骤 6 不要求候选方案已经完成真实 pruning eval，也不生成 pruning metric 曲线。metric plot 由步骤 7 在结果补齐后生成。

以下 legacy 图不再生成：

```text
*_gt_components.png
*_paths.png
*_gap_heatmaps.png
```

## 3. Redundancy graph 与 metric 聚合

每个 redundancy artifact 保存 unit 输出之间的 pairwise metric：

```text
M[p,q]
```

同时保存：

```text
<metric>_scenes
<metric>_mean
```

例如：

```text
cosine_scenes.shape = [num_scenes, num_units, num_units]
cosine_mean.shape   = [num_units, num_units]
```

注意：`cosine_scenes[s,p,q]` 已经是在 scene 内对 token row 求均值的结果，不是逐 token cosine 明细。

当 `token_region=mean` 时，加载顺序为：

```text
先对一个 dataset 的可用 token-region 矩阵求均值
再对 scannet_dev50、nuscenes_train 的矩阵求均值
```

即：

```text
token region → dataset
```

当前各方法读取的输入为：

| 方法 | 输入 |
|---|---|
| Ours | CKA similarity |
| Cosine | cosine similarity |
| INV_MSE | inverse-MSE similarity |
| ReplaceMe | angular distance |
| ShortGPT | cosine similarity |
| SIMPLER | 默认与 Ours 相同的 CKA similarity |
| GT | 单区间真实下游指标，不以 redundancy graph 为目标 |

### 3.1 ReplaceMe 的 angular distance

ReplaceMe 使用原生 angular metric：

```text
angular(x,y) = mean_token(acos(cosine(x,y)) / pi)
```

该非线性在 token 求均值之前执行。步骤 3 必须包含：

```text
METRICS=...,angular,...
```

步骤 6 直接加载 `angular_mean`；不存在从 angular 自动回退到 cosine 近似的逻辑。如果所选 region 缺少 angular，ReplaceMe strategy 无法构建。

已有的 `cosine_scenes` 只能构造：

```text
acos(mean_token(cosine)) / pi
```

而严格 angular 是：

```text
mean_token(acos(cosine) / pi)
```

两者通常不相等，所以不能用旧 cosine artifact 精确恢复。

## 4. 物理 drop set 与 selector

### 4.1 unit 0 受保护

所有当前 strategy 都禁止删除 unit 0：

```text
0 not in drop
```

基于边界输出的区间 `[a,b]` 必须满足 `a>=1`，因为当前 redundancy graph 没有单独保存“模型输入到 unit 0 输出”的边。

这是 ShortGPT 相对官方实现仍保留的一项差异：官方可以为第一层计算输入–输出 BI。

### 4.2 selector 支持任意多个区间

物理 drop set 会被拆成全部最大连续区间。例如：

```text
drop = [1,3,5,6,9]
selector = i1_j1_i3_j3_i5_j6_i9_j9
```

因此当前 selector 不再限制为最多两个 `iS_jE` 片段。该扩展主要用于官方 ShortGPT。

规范化规则：

- 区间按起点升序；
- 相邻或重叠区间按物理并集合并；
- 新结果只写规范名称；
- 历史双区间反序名称仍可读取；
- 多个 strategy 命中同一规范 selector 时共享一次真实评测。

### 4.3 不同 strategy 的区间约束不同

不存在适用于所有方法的“最多两个区间”公共约束。

| 方法 | 区间约束 |
|---|---|
| Ours_CKA / Ours_Cosine / Ours_INV_MSE Double、GT Double | 最多两个；若为两个，中间至少保留 1 个 unit |
| ReplaceMe Double | 两个等长 block；只禁止重叠；允许相邻 |
| ShortGPT | 不限制区间数量或间隔 |
| SIMPLER | 一个 suffix |

ReplaceMe 选出的两个 block 若相邻，纯删层后的物理并集是一个连续区间，因此规范 `eval_selector` 会合并成单区间；它仍属于 `replaceme_double`，因为算法语义是 `num_A=2`。

## 5. 通用区间边界分数

Ours、Cosine 和 INV_MSE 从 unit pairwise similarity 构造区间 corner score。

删除闭区间 `[a,b]` 时：

```text
C[a,b] = S[a-1,b]
```

其中：

- `S` 是对应 metric 的 unit-output pairwise similarity；
- `a>=1`；
- `b>=a`；
- `C[0,:]=NaN`。

`C[a,b]` 表示删除 `[a,b]` 后，区间输入边界 `a-1` 与原模型区间末端输出 `b` 的相似程度。

ReplaceMe 同样采用边界映射，但输入是 distance：

```text
A[a,b] = angular_output_distance[a-1,b]
```

ShortGPT 不使用区间 corner score，而使用每个 unit 自身的输入–输出 BI。

## 6. Ours

### 6.1 分数

Ours 首先把 CKA corner similarity 转换成 proxy drop：

```text
D[a,b] = 1 - CKA[a-1,b]
```

因此 Ours 是 **越小越好**，而不是直接最大化 CKA：

```text
single score = D[a,b]
double score = D[a,b] + D[c,d]       # sum 模式
double score = max(D[a,b], D[c,d])   # max 模式
```

双区间联合目标由
`OURS_GT_TWO_INTERVAL_SCORE_AGGREGATION = "sum" | "max"` 硬编码控制：

- `sum`：默认值，保持原有“两个区间下降量相加”的近似；
- `max`：联合下降由两个分量中更大的下降量决定。

该开关只作用于 GT Double 和 Ours 系列的 Double 版本，即
Ours_CKA、Ours_Cosine、Ours_INV_MSE；所有 Single 版本以及 ReplaceMe、
ShortGPT、SIMPLER 均不受影响。每个 Double strategy 的 `source` 和每个
预算点的 `joint_score_aggregation` 都会记录实际模式。

两个分量均来自原始模型的一次性 CKA graph：

```text
scores_recomputed_after_first_drop = false
```

### 6.2 Single 与 Double

`ours_single`：

- 每个预算只允许一个连续区间；
- 最小化该区间的 proxy drop。

`ours_double`：

- 候选可以是一个区间或两个分离区间；
- 两个区间之间至少保留一个 unit；
- 双区间只有比最佳单区间至少改善超过 `0.01` 才会被采用；
- 不足 `0.01` 视为同等，选择单区间。

因为越小越好，双区间判定为：

```text
score_double < score_single - 0.01
```

### 6.3 跨预算路径约束

当前 Ours 不是每个预算完全独立。候选按删层预算从大到小选择，并要求预算 `n` 的每个 component 至少与预算 `n+1` 已选方案中的一个 component 相交。

这会提高 drop curve 的路径连续性，但属于当前 Ours 的设计，不应错误描述为其他 baseline 的官方要求。

## 7. GT

GT 从两个 training datasets 的单区间真实下游结果构造 degradation heatmap。

当前使用七个指标：

```text
ape
are
rpe_trans
rpe_rot
acc
comp
chamfer_distance
```

NC 不参与 GT 七指标均值。

每个指标先换算为相对 upper model 的退化倍数，再在自己的单区间 heatmap 上归一化，随后：

```text
七指标求均值 → dataset GT heatmap
两个 training datasets 求均值 → 最终 G[a,b]
```

`G[a,b]` 越小越好。

`gt_single`：

```text
min G[a,b]
```

`gt_double`：

```text
single = G[a,b]
double = G[a,b] + G[c,d]       # sum 模式
double = max(G[a,b], G[c,d])   # max 模式
```

并采用与 Ours 相同的：

- 最多两个区间；
- 至少一个保留 unit 的间隔；
- `0.01` 单/双区间改善阈值；
- 平分时选择单区间；
- 从大预算到小预算的 component-overlap 路径约束。

GT 双区间始终假设删除第一个区间后，第二个区间基于原模型测得的下降近似
不变；在 `sum` 模式下进一步采用可加性假设，在 `max` 模式下则采用
“联合下降由较差分量主导”的假设：

```text
删除第一个区间后，第二个区间的下降近似不变；
联合下降近似为两个原始单区间下降之和，或二者的最大值。
```

步骤 6 不使用真实双区间结果参与选择。真实交互由步骤 7 的物理 pruning eval 检验。

GT 是本项目内部的近似 oracle，不是外部论文 baseline。

## 8. ReplaceMe 官方复现

参考：

- 官方仓库：`submodules/ReplaceMe`
- 论文：https://arxiv.org/abs/2505.02819
- 上游仓库：https://github.com/mts-ai/ReplaceMe

### 8.1 官方选择逻辑

官方先对固定长度 `layers_to_skip=m` 的所有 block 计算 angular distance：

```text
distance(start,m)
```

然后：

1. 按每个 block 自己的 distance 从小到大稳定排序；
2. 依次接受与已选 block 不重叠的候选；
3. 直到选满 `num_A` 个 block。

这是贪心算法，不是所有 block 组合的联合最优搜索，也不以多个 block 的分数之和作为选择目标。

### 8.2 当前 Single / Double

`replaceme_single`：

```text
num_A=1
layers_to_skip=n
```

每个总预算 `n` 选择一个长度为 `n` 的最小 angular-distance block。

`replaceme_double`：

```text
num_A=2
total_drop = 2 * layers_to_skip
```

因此：

- 只定义偶数总预算；
- 两个 block 严格等长；
- 按官方 individual-distance 顺序贪心选择；
- 只禁止重叠；
- 允许相邻；
- `merge_consecutive=False`；
- 不使用 0.01 阈值；
- 不进行 joint-score 枚举；
- 选完第一个 block 后不重算原模型 distance。

plan 中记录两个已选 distance 的均值仅用于诊断，不参与候选选择。

### 8.3 忠实部分与适配部分

| 项目 | 当前状态 |
|---|---|
| angular distance 公式 | 对齐 |
| 固定 `layers_to_skip` | 对齐 |
| `num_A=1/2` | 对齐 |
| `num_A=2` 两块等长 | 对齐 |
| 按 individual distance 贪心 | 对齐 |
| 只禁止重叠、允许相邻 | 对齐 |
| 删除第一块后重算 distance | 官方和当前都不重算 |
| calibration 输入 | 从 LLM 文本/最后非 padding token 适配为 3D scene/token |
| unit 0 | 当前保护，官方方法本身没有该公共限制 |
| linear transformation recovery | 不属于步骤 6/7 的纯删层 plan；完整 ReplaceMe 需单独使用 recovery pipeline |

官方配置文件中出现 `min_distance_layer=20`，但上游实际 distance selection 代码没有用它过滤候选。当前不人为增加“只从后 20/25 层开始”的约束，因此与实际官方代码行为一致。ReplaceMe 不应被标记为使用固定后层起点先验。

## 9. ShortGPT 官方复现

参考：

- 官方仓库：https://github.com/icip-cas/ShortGPT
- 论文：https://aclanthology.org/2025.findings-acl.1035/

### 9.1 BI

官方 Block Influence：

```text
BI[k] = 1 - E_token(cos(input_k, output_k))
```

在当前顺序执行的 unit catalog 中，对于 `k>=1`：

```text
input_k  = output_(k-1)
output_k = output_k
```

所以当前使用：

```text
BI[k] = 1 - S_cos[k-1,k]
```

按 BI 从小到大排序，等价于按相邻 cosine redundancy 从大到小排序。

### 9.2 当前选择已恢复官方全局 order

当前只计算一份全局 ranking：

```text
ranking = argsort(BI ascending)
```

预算 `n` 直接取：

```text
drop(n) = sorted(ranking[:n])
```

因此：

- 不限制区间数量；
- 不限制区间间隔；
- 不比较单区间与双区间；
- 不使用 0.01 阈值；
- 不进行每个预算独立组合优化；
- 不同预算严格嵌套：

```text
drop(n) ⊂ drop(n+1)
```

这与官方 `np.argsort(importances)[:n_prune_layers]` 的选择语义一致。

### 9.3 仍保留的 3D 适配

| 项目 | 当前状态 |
|---|---|
| BI 公式 | 对齐 |
| 一次性全局排序 | 对齐 |
| 预算取 ranking 前缀 | 对齐 |
| 任意多个区间 | 对齐 |
| 删除后重算 BI | 官方和当前都不重算 |
| calibration 数据 | 从 PG19 文本适配为两个 3D training datasets |
| token 权重 | 当前先做 scene/region/dataset 聚合，与官方 token 累加权重不完全相同 |
| unit 0 | 当前无法评分并受保护；官方可计算第一层输入–输出 BI |

当前静态删层对应论文的 ShortGPT，而不是只对 prompt token 跳层、生成 token 走完整模型的 ShortGPT-gen。

## 10. SIMPLER：公式对齐，但输出方式不是严格官方方法

参考：

- 论文：https://arxiv.org/abs/2603.19873
- 官方仓库：https://gitlab.citius.gal/hpc4rs/simpler

对于 cutoff `c`：

```text
TL = S[:c,:c]
BR = S[c:N,c:N]
delta(M) = mean(abs(diff(M, axis=0)))
score(c) = delta(TL) - delta(BR)
```

论文在：

```text
c in {2,...,N-2}
```

中选择唯一：

```text
c* = argmax score(c)
```

并删除 suffix：

```text
[c*,N-1]
```

当前实现的 score 公式、合法 cutoff 范围以及 suffix 语义与论文一致，但 strategy 实际输出的是完整预算 sweep：

```text
n → [N-n,N-1]
```

虽然代码记录了 `best_cutoff`，却没有只输出该唯一点。因此当前 pruning curve 本质上是：

```text
SIMPLER-style budget-matched suffix / Last-n sweep
```

而不是论文中“算法自行决定一个删层量”的严格 SIMPLER。论文中的先验更准确地说是 **deep-tail redundancy prior**，不是专属于 LLM 的先验；当前图例中的 “LLM depth prior” 只是实验叙事标签。

此外，论文使用任务图像的 layer representation；当前使用 3D scene/token graph，并对两个 plan datasets 聚合，这也是输入适配。

## 11. Cosine 与 INV_MSE ablation

这两个 strategy 使用公共一/二区间枚举器，并按每个预算独立选择。

Cosine：

```text
single = C_cos[a,b]
double = C_cos[a,b] + C_cos[c,d]
maximize
```

INV_MSE：

```text
single = C_inv_mse[a,b]
double = C_inv_mse[a,b] + C_inv_mse[c,d]
maximize
```

共同规则：

- 最多两个区间；
- 两区间之间至少保留一个 unit；
- 双区间需比单区间高超过 `0.01`；
- 平分时选择单区间；
- 每个预算独立；
- 所有分数来自原始模型 graph，不在第一次删除后重算。

由于双区间是两个正 similarity score 直接相加，而单区间只有一个 score，这两个 ablation 在数值尺度上可能天然偏向双区间。它们不是当前 ReplaceMe 的实现；ReplaceMe 已改为官方 angular-distance 等长贪心算法。

## 12. Strategy 对照表

| Strategy | 目标 | 预算选择 | 区间数 | 跨预算关系 | 官方复现结论 |
|---|---|---|---:|---|---|
| Ours Single | 最小化 `1-CKA` | 单区间搜索 | 1 | component overlap path | 本项目方法 |
| Ours Double | 最小化 proxy-drop 的 `sum`/`max` | 单/双自适应，阈值 0.01 | ≤2 | component overlap path | 本项目方法 |
| GT Single | 最小化真实单区间退化 | 单区间搜索 | 1 | component overlap path | 内部 oracle |
| GT Double | 最小化退化的 `sum`/`max` 近似 | 单/双自适应，阈值 0.01 | ≤2 | component overlap path | 内部 oracle |
| ReplaceMe Single | 最小 angular distance | `num_A=1` 贪心 | 1 | 各 `layers_to_skip` 独立 | 选择逻辑对齐，输入为 3D 适配 |
| ReplaceMe Double | 最小 individual angular distance | `num_A=2` 等长贪心 | 先选 2 blocks | 偶数预算 | 选择逻辑对齐，输入为 3D 适配 |
| ShortGPT | 最小 BI | 全局 ranking 前缀 | 任意 | 严格嵌套 | 选择逻辑对齐，输入和 unit 0 有适配 |
| SIMPLER | 最大 TL-BR cutoff score | 当前输出 suffix sweep | 1 | suffix 天然嵌套 | 公式对齐，输出不是严格官方 |
| Cosine | 最大 corner similarity 和 | 每预算独立 | ≤2 | 无 | 内部 ablation |
| INV_MSE | 最大 corner similarity 和 | 每预算独立 | ≤2 | 无 | 内部 ablation |

## 13. 步骤 7：增量真实评测

步骤 7 将 strategy 内部别名：

```text
ours_double_k10
replaceme_double_k10
shortgpt_k10
```

映射为规范物理 selector，然后跨 strategy 去重。

结果目录：

```text
analysis_outputs_v3/<backbone>/<dataset>/eval/pruning/<eval_selector>/
```

唯一完成条件：

```text
<eval_selector>/summary.json 存在
```

只有目录、scene 文件或 `eval_record_v3.json` 不算完成。

执行顺序：

```text
scannet_dev50
nuscenes_train
→ training 两数据集严格平均图
scannet_test50
7scenes_test
nuscenes_val
waymo_validation
→ eval 四数据集严格平均图
```

每个 dataset 完成后生成单数据集 pruning plot。分组平均只纳入组内所有 dataset 都完整存在的 strategy-budget 点，不做静默部分平均。

training datasets 的单区间 brute-force 结果应已存在：

- 所需物理 selector 只有一个区间且 `summary.json` 缺失时，立即报错；
- 所需 selector 包含多个区间时，只补跑缺失结果；
- ShortGPT 现在可能产生三个或更多区间，步骤 7 和磁盘 selector 已支持这种情况。

## 14. Plot 语义

Pruning metric plot 只绘制七指标 mean degradation。

- 颜色表示方法；
- 虚线表示显式 Single 版本；
- 实线表示 Double/adaptive 版本以及无约束 multi-interval ShortGPT；
- 点线表示带 deep-tail depth prior 的 SIMPLER-style suffix。

同一方法的 Single/Double 共用颜色，避免为每种“方法 × 线型”组合重复占用 legend。

## 15. 解释结果时必须保留的边界

1. Ours 系列和 GT Double 的双区间分数是原始模型两个单区间 proxy
   的和或最大值（由硬编码开关控制），两种模式都没有重新测量删层交互。
2. Ours/GT 当前存在跨预算 component-overlap 路径约束；Cosine、INV_MSE、ReplaceMe 没有该路径；ShortGPT 使用官方全局 ranking 前缀。
3. ReplaceMe 的 block 选择逻辑已对齐官方，但步骤 6/7 的纯删层结果不包含 ReplaceMe 的线性变换恢复，完整端到端方法需单独报告 recovery。
4. ShortGPT 的选择逻辑已恢复官方，但 calibration 数据、聚合权重和 unit 0 保护仍是 3D pipeline 适配。
5. SIMPLER 的 cutoff score 公式对齐论文，但完整预算 suffix curve 不等于官方只选择一个 `c*`。
6. Cosine 和 INV_MSE 是内部 ablation，不应写成 ReplaceMe 或其他论文的严格复现。
7. “使用原始模型一次性分数、不在第一次删除后重算”对当前所有相似度 baseline 都成立；这与 ReplaceMe、ShortGPT 官方的一次性选择一致，但对 Ours/GT 是显式建模假设。
