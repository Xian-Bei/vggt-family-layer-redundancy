# LaRe 下标、区间、Cache 与 Recovery 语义规范

本文是 LaRe 当前 unit 下标、heatmap、剪枝 selector、activation cache 和 recovery
boundary 的语义规范。实现依据是 `analysis/prune_plan/__init__.py`、
`analysis/prune_plan/recovery_plan.py` 与 `analysis/recovery/`。

各 recovery 算法的拟合目标、训练方式、多区间行为和 checkpoint 形态见
[recovery_methods.md](recovery_methods.md)。

## 1. Unit 编号与执行顺序

所有可剪枝 unit 使用从 0 开始的全局线性下标。下标的唯一权威来源是：

```python
unit_ids = [unit.unit_id for unit in adapter.layer_unit_catalog(model)]
```

当前 VGGT、VGGT-Omega 和 Pi3 的顺序是：

```text
unit 2L     = frame_L
unit 2L + 1 = global_L
```

因此公共代码必须使用 plan/cache 中保存的 `unit_ids` 映射 index 和 ID，不得仅根据
奇偶下标推导 kind。若模型共有 `N` 个 unit，单个下标的取值域是 `[0, N)`。

## 2. 闭区间与任意 drop set

所有 `(i, j)`、`[i, j]`、`start/end` 和 `iX_jY` 都表示零基闭区间：

```text
drop = [i, i+1, ..., j]
0 <= i <= j < N
drop_count = j - i + 1
```

当前 pruning/recovery 还支持多个不连续区间。任意 drop set 会被拆成全部最大连续
component，并写成规范物理 selector：

```text
drop = [1, 3, 5, 6, 9]
selector = i1_j1_i3_j3_i5_j6_i9_j9
```

`intervals` section 中的一条 interval record 只描述一个连续区间；`sets` section
描述 strategy 产生的任意 drop set。相邻或重叠 component 在物理 selector 中合并。

当前所有合法 prune plan 都保护 unit 0：

```text
0 not in drop
```

所以当前可执行 pruning/recovery selector 的每个首 component 都满足 `start >= 1`。

## 3. Heatmap 坐标

区间 heatmap 的有效格点 `(i, j)` 表示删除闭区间 `[i, j]`：

- 行坐标是起点 `i`；
- 列坐标是终点 `j`；
- `i <= j` 有效，`j < i` 无效；
- 对角线 `(i, i)` 删除一个 unit；
- 删除数量始终为 `j-i+1`。

从 unit-output pairwise graph `S` 构造 boundary score 时：

```text
C[i,j] = S[i-1,j], i >= 1
C[0,:] = NaN
```

当前 graph 没有把模型输入到 unit 0 输出作为一条普通 unit-output edge，因此 unit 0
既不能通过该 heatmap 评分，也被 prune plan schema 明确禁止删除。tooltip、tick、
selector 和保存的 plan 不得在 UI 层额外做 `+1/-1` 转换。

## 4. Cache activation

原模型 cache 中：

```text
unit_map[unit_id] = 对应物理 unit 执行完成后的输出
layer_-1          = catalog 中 unit 0 执行前的真实初始 tokens
```

标准 unit activation 形状是 `[B, S, P, C]`。某些 replay engine 可临时构造
`[B, S, P, 2C]` 的 concat feature，但不改变单个 unit activation 的边界含义。

`layer_-1` 用于从模型起点 replay。当前 VGGT、VGGT-Omega 和 Pi3 adaptor
都显式捕获或构造真实 unit-0 输入；不得用 unit 0 输出冒充 `layer_-1`。

minimal cache 保存 recovery record 的全部 `cache_unit_ids`，以及方法需要的
`auxiliary_cache_ids`。ReplaceMe 方法会额外请求
`replaceme_residual:<source_unit_id>`。

## 5. Recovery boundary 的公共规则

对一个最大连续删除 component `[i, j]`：

```text
source_unit_index = i - 1
target_unit_index = j
source_unit_id    = unit_ids[i - 1]
target_unit_id    = unit_ids[j]
```

因为 unit 0 受保护，当前 recovery 的 `source_unit_index` 总是合法物理 unit，不存在
`layer_-1` source 或 unit-0 pre-hook 分支。

拟合使用未剪枝 cache 的同一对边界：

```text
X = source unit 的原模型输出
Y = target unit 的原模型输出
fit X -> Y
```

sequential 拟合路径是例外：第一个 component 仍等价于 Full-model source/target；从第二个
component 起，source 和 target 都来自已经应用前序 recovery 和剪枝的当前模型。
`recovery_ghost_interval_sequential`、其 token-grouped 变体、
以及恢复为官方行为的 `recovery_replaceme_cos`，都以删去当前 component 前的当前模型
为恢复目标。

`recovery_streamline_ffn` 使用同一 boundary 定义，但优化的是带 bias 的
`C -> 4C -> C` ReLU FFN，并以原模型 cache 中的 `MSE(FFN(X), Y)` 为目标。

部署时先把 drop units 替换为 skip/identity，再在 source unit 输出处安装或融合 recovery
operator。训练和推理必须观察同一个 source boundary；禁止恢复旧的 `source=i` 语义。

## 6. 不同 recovery 方法如何拆 boundary

`analysis.prune_plan.recovery_plan._boundary_records()` 从任意 drop set 生成
`boundaries`：

| 方法 | boundary 拆分 |
|---|---|
| `recovery_ghost` | 每个被删物理 unit 一个 `[k,k]` boundary；所有 token 共用一个矩阵 |
| `recovery_ghost_interval` | 每个最大连续 component 一个 boundary；所有 token 共用一个矩阵 |
| `recovery_ghost_interval_sequential` | 同上；后续 component 从逐步修改后的模型同时采集 source 和 target |
| `recovery_ghost_interval_sequential_grouped` | 与 sequential Ghost 相同，但将 camera/register 合为 `special` 组，patch 单独成组 |
| `recovery_replaceme_ls` | 每个最大连续 component 一个 boundary，并使用 Full-model source/target/residual |
| `recovery_replaceme_cos` | 每个最大连续 component 一个 boundary，并使用 residual；多 component 时按官方顺序从逐步修改后的模型采集 source/residual/target |
| `recovery_streamline_ffn` | 每个最大连续 component 一个 boundary；从 Full-model cache 以 boundary MSE 训练 4x-ReLU FFN |
| `recovery_linearpatch` | 单 component 时一个 boundary；多 component 时退化为每个被删 unit 一个 boundary |
| `recovery_lora` / `recovery_train` | 使用 recovery bundle 的 span training 配置，并保留派生的 boundary 列表 |

顶层 record 为兼容公共调用保留第一个 boundary 的 source/target，同时
`record["boundaries"]` 保存完整列表。cache unit 是所有 boundary source/target 的并集。

Pi3 adaptor 只声明一个 `patch` token 组，因此 grouped Ghost 与其对应的非 grouped
sequential 方法完全等价。Pi3 runner 会先保证基础方法完成，再把
grouped recovery checkpoint 目录和 eval 目录创建为指向基础结果的相对符号链接。

## 7. 示例

以 frame-first catalog 中删除 `[11,15]` 为例：

```text
drop indices = [11, 12, 13, 14, 15]
drop count   = 5
source index = 10 = frame_5
target index = 15 = global_7
```

派生 boundary：

```json
{
  "drop_start_index": 11,
  "drop_end_index": 15,
  "source_unit_index": 10,
  "source_unit_id": "frame_5",
  "target_unit_index": 15,
  "target_unit_id": "global_7",
  "cache_unit_ids": ["frame_5", "global_7"]
}
```

对于 `drop=[1,2,5,6]`，interval Ghost 和 ReplaceMe 会产生 `[1,2]`、`[5,6]`
两个 boundary；unit-wise Ghost 则产生 `[1,1]`、`[2,2]`、`[5,5]`、`[6,6]`。

## 8. Trainable recovery

对单个 `[i,j]`，LoRA anchor 从左边界 `i-1` 向前扩展 `lora_prefix` 个 unit，teacher
target 是 `j`。cache replay seed 是最小训练 unit 的前驱；若 replay 从 catalog 起点
开始，则使用合法的 `layer_-1` initial tokens。

`recovery_lora` 和 `recovery_train` 的 span-specific `cache_unit_ids`、`lora_indices` 和
`train_unit_indices` 保存在 recovery bundle 的 `methods.<name>.spans` 中。当前默认
`RECOVERY_METHODS` 默认启用 training-free 与 Streamline local-fit 方法；显式启用 replay-trainable 方法时仍必须使用
同一 v3 bundle。

## 9. Schema

Prune plan 当前唯一合法 schema：

```text
closed_interval_prune_plan_bundle_v5
```

Recovery plan 文件固定为：

```text
<OUTPUT_ROOT>/<backbone>/recovery_plan_all.json
schema_version = recovery_plan_bundle_v3
```

v3 recovery bundle 顶层保存 backbone、dataset、`num_units`、`unit_ids` 和统一
`methods` map。training-free method entry 只需保存 `method`；boundary 在与具体 prune
selector 组合时派生。trainable entry 另外保存 LoRA 参数和 span records。

加载器不兼容旧 recovery schema。修改 schema、unit catalog、boundary 拆分或 operator
后，必须重新生成 recovery plan 和 checkpoint，并重跑 recovery/testtime eval。

## 10. 命名与输出路径

工程标识统一使用 `recovery`：

```text
Python package : analysis.recovery
shell entry    : scripts/run_recovery.sh
checkpoint dir : <OUTPUT_ROOT>/<backbone>/recovery/<selector>/<method>/...
evaluation dir : <OUTPUT_ROOT>/<backbone>/<dataset>/eval/recovery/<selector>/<method>/...
test-time dir  : <OUTPUT_ROOT>/<backbone>/<dataset>/eval/testtime/<method>/<compression>/<selector>/...
JSON kind      : recovery
```

不得重新引入 `analysis.recover`、`run_recover.sh`、`/recover/`、`eval/recover/` 或
`kind=recover`。

## 11. 产物复用

- 全量 cache 只有在 unit catalog、activation 定义和模型输入预处理未变时才可复用；
- minimal cache 必须覆盖当前 selector 与全部 recovery methods 派生出的 union；
- boundary 拆分、source/target 规则或 auxiliary cache 变化后必须重建 minimal cache；
- checkpoint topology、recovery operator 或 schema 变化后必须重跑 recovery eval；
- testtime compression 结果不能覆盖普通 recovery baseline。

## 12. 提交前检查

- [ ] index 来自保存的 `unit_ids`，没有假设固定 frame-first 奇偶映射；
- [ ] interval 使用闭区间，单区间删除数量是 `j-i+1`；
- [ ] 任意 drop set 的物理 selector 已按最大连续 component 规范化；
- [ ] unit 0 未出现在 drop 中；
- [ ] 每个 boundary 的 source 是 component start `-1`，target 是 component end；
- [ ] cache union 包含全部 source/target 和 auxiliary IDs；
- [ ] schema 分别是 prune v5 和 recovery v3；
- [ ] 没有新增旧 `recover` 工程路径或兼容别名。
