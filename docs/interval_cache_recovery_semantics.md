# LaRe 下标、区间、Cache 与 Recovery 语义规范

本文是 LaRe 中 unit 下标、heatmap、剪枝区间、cache 和 recovery 的唯一语义规范。
新增代码、脚本、JSON schema 和可视化必须遵循本文；不再支持旧的半开区间或
`source=i` 语义。

## 1. Unit 编号

所有可剪枝 unit 使用从 0 开始的全局线性下标。对于第 `L` 层：

```text
unit 2L     = frame_L
unit 2L + 1 = global_L
```

例如：

| Unit index | Unit ID |
| ---: | --- |
| 9 | `global_4` |
| 10 | `frame_5` |
| 11 | `global_5` |
| 15 | `global_7` |

若模型共有 `N` 个 unit，合法的单个 unit 下标范围是 `[0, N)`。这里的 `[0, N)`
只描述单个下标的取值域，不是剪枝区间语义。

## 2. 剪枝区间统一为闭区间

所有 `(i, j)`、`[i, j]`、`start/end` 和 `iX_jY` selector 都表示零基闭区间：

```text
drop = [i, i+1, ..., j]
0 <= i <= j < N
drop_count = span = j - i + 1
```

因此：

- `[11, 15]` 删除 5 个 unit：`11, 12, 13, 14, 15`；
- `[13, 13]` 删除 1 个 unit；
- 横坐标“删除数量”必须使用 `j-i+1`；
- 禁止把 `[i, j]` 解释为 Python slice `[i, j)`；
- 禁止使用 `j-i` 作为删除数量。

JSON 中推荐同时保存显式边界和 drop 列表：

```json
{
  "i": 11,
  "j": 15,
  "start": 11,
  "end": 15,
  "span": 5,
  "drop_count": 5,
  "drop": [11, 12, 13, 14, 15]
}
```

这些字段必须互相一致。`drop` 必须连续，不允许用同一条 interval record 表示多个
离散区间。

## 3. Heatmap 坐标语义

区间 heatmap 中一个有效格点 `(i, j)` 表示“删除闭区间 `[i, j]`”。

- 行坐标：区间起点 `i`；
- 列坐标：区间终点 `j`；
- 有效区域：`i <= j`；
- 对角线 `(i, i)`：删除单个 unit；
- `j < i` 的下三角区域无效，不得赋予剪枝语义。

若 heatmap 存的是边界相似度，则格点 `(i, j)` 的两个边界 activation 是：

```text
source boundary = unit i-1 的输出
target boundary = unit j 的输出
```

当 `i=0` 时，source boundary 是 `layer_-1`，即 unit 0 的输入。它不是 unit 0 的
输出。

可视化的 tooltip、tick label、导出的 selector 和点击后保存的计划必须保持同一组
`(i, j)`，不得在 UI 层做 `+1` 或 `-1` 转换。

## 4. Cache activation 语义

原模型 cache 中：

```text
frame_L / global_L = 对应 block 执行完成后的输出
layer_-1           = unit 0 执行前的初始 tokens
```

所有 unit activation 的标准形状为 `[B, S, P, C]`。按层 replay 时可以临时把
`frame_L` 与 `global_L` 拼成 `[B, S, P, 2C]`，但这不会改变它们各自表示 block
输出的语义。

`layer_-1` 必须满足以下不变量：

```text
cached layer_-1 == 未剪枝模型 unit 0 的实际输入
```

不得为了节省一次 patch embedding forward，把 `frame_0` 输出保存为 `layer_-1`。

## 5. Training-free Recovery 边界

对于删除闭区间 `[i, j]`：

```text
source_unit_index = i - 1
target_unit_index = j
source_unit_id    = layer_-1                      if i == 0
                    unit_index_to_unit_id(i - 1)  if i > 0
target_unit_id    = unit_index_to_unit_id(j)
```

拟合阶段使用原模型 cache：

```text
X = source boundary activation
Y = target boundary activation
拟合 recovery operator: X -> Y
```

推理阶段先把 `[i, j]` 中的 block 替换为 identity，再安装 recovery：

- `i > 0`：在保留的 unit `i-1` 输出上安装 forward hook；
- `i == 0`：在 unit 0 上安装 forward pre-hook，输入为 `layer_-1`；
- patch 后的 activation 通过 identity 区间传到 unit `j+1`。

因此拟合和推理必须看到同一个 source boundary。禁止以下旧语义：

```text
drop = [i, ..., j]
source = i
```

因为 cache 中 unit `i` 是原 block 的输出，而剪枝推理中 unit `i` 已是 identity，
它的输出等于原 unit `i-1` 的输出，两者不是同一个 activation。

## 6. 完整示例：删除 `[11, 15]`

```text
drop indices = [11, 12, 13, 14, 15]
drop count   = 5
source index = 10 = frame_5
target index = 15 = global_7
```

正确的 recovery record：

```json
{
  "drop": [11, 12, 13, 14, 15],
  "source_unit_index": 10,
  "source_unit_id": "frame_5",
  "target_unit_index": 15,
  "target_unit_id": "global_7"
}
```

数据流为：

```text
拟合：原模型 frame_5 输出  -> 原模型 global_7 输出
推理：保留的 frame_5 输出 -> recovery patch -> identity units 11..15
```

## 7. LoRA / Trainable Recovery

对于 `[i, j]`，LoRA recovery 的左边界 anchor 同样是 `i-1`：

```text
drop units       = i..j
LoRA anchor      = i-1
teacher target   = j
cache replay seed = 最小 LoRA unit 的前驱
```

例如 `[11, 15]` 且 LoRA prefix 为 1：

```text
LoRA anchor = unit 10
replay seed = unit 9
teacher     = unit 15
```

`i=0` 没有物理左边界 unit，因此当前 LoRA recovery 必须拒绝该区间；不得静默改用
unit 0 作为 anchor。

## 8. Schema 与运行时校验

Recovery plan 当前唯一合法 schema：

```text
closed_interval_recovery_plan_bundle_v2
```

加载 recovery plan 时必须验证：

```text
drop 连续
source_unit_index == min(drop) - 1
target_unit_index == max(drop)
source/target unit ID 与 index 映射一致
```

旧的 `training_plan_bundle_v1` 和 `source=min(drop)` 记录必须报错，不允许自动迁移或
兼容加载。

## 9. 命名规范

工程标识统一使用 `recovery`：

```text
Python package : analysis.recovery
shell entry    : run_recovery.sh
checkpoint dir : <OUTPUT_ROOT>/<backbone>/recovery/...
evaluation dir : <OUTPUT_ROOT>/<backbone>/<dataset>/eval/recovery/...
JSON kind      : recovery
```

不得重新引入 `analysis.recover`、`run_recover.sh`、`/recover/`、`eval/recover/` 或
`kind=recover`。自然语言中的英语动词 `recover`、形容词 `recovered` 不受此限制。

## 10. 产物失效与重跑规则

以下变更会使既有 recovery 产物失效：

- 区间从半开改为闭区间；
- source 从 `i` 改为 `i-1`；
- `layer_-1` 的定义或生成逻辑改变；
- recovery schema 或 recovery operator 改变。

失效后必须重新生成 recovery plan，并重新拟合或训练 checkpoint；已有 recovery eval
和汇总也必须重跑。

Cache 是否可复用：

- 全量 cache：对 `i>0`，若包含 `i-1` 和 `j`，通常可以复用；
- 根据旧 plan 保存的 minimal cache：必须重跑，因为可能缺少 `i-1`；
- `i=0`：只有确认 `layer_-1` 是真实初始 tokens 时才能复用。

## 11. 当前已知限制

截至本文建立时，VGGT Omega fused cache 路径仍会把 `frame_0` 输出作为
`layer_-1` 返回。该行为违反第 4 节的不变量。因此在修复并重新生成 cache 前，禁止
使用 VGGT Omega 的 `i=0` training-free recovery。`i>0` 不受这个特例影响。

## 12. 提交前检查清单

- [ ] selector、JSON、heatmap 和绘图横坐标都使用 `j-i+1`；
- [ ] `drop == list(range(i, j+1))`；
- [ ] recovery source 是 `i-1`，target 是 `j`；
- [ ] cache 保存 block 输出，`layer_-1` 保存 unit 0 输入；
- [ ] 拟合 activation 与推理 hook activation 是同一边界；
- [ ] 没有新增 `recover` 工程路径或兼容别名；
- [ ] schema 和边界不变量测试通过。
