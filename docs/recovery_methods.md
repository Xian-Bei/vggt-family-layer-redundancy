# LaRe Recovery 方法总览

本文整理当前代码支持的 10 种 recovery 方法，重点说明它们的训练方式、拟合目标、
boundary 拆分、多区间行为、部署形态和产物。下标、闭区间、cache 与 boundary 的统一
定义见 [interval_cache_recovery_semantics.md](interval_cache_recovery_semantics.md)。实现的
主要入口是：

- plan：`analysis/prune_plan/recovery_plan.py`、`analysis/prune_plan/build_recovery_plan.py`；
- training-free 拟合：`analysis/recovery/recovery_prune_v3.py`；
- trainable recovery：`analysis/recovery/train/train_lora_skip_baseline.py`；
- checkpoint 安装：`analysis/recovery/operators.py`；
- shell 入口：`scripts/run_recovery.sh`。

## 1. 方法清单

| 方法 | 类别 | boundary 粒度 | 优化对象 | 部署方式 |
|---|---|---|---|---|
| `recovery_lora` | trainable | span plan 指定的左侧 anchor | anchor attention 的 `qkv`、`proj` LoRA | 保存 trainable delta |
| `recovery_train` | trainable | span plan 指定的左侧 anchor | anchor unit 的原始全部参数 | 保存 trainable delta |
| `recovery_linearpatch` | training-free | 单 component 一个；多 component 时逐 unit | Hadamard 域通道尺度 | source boundary 后置矩阵 |
| `recovery_ghost` | training-free | 每个被删 unit 一个 | 残差线性回归矩阵 | 被跳过 unit 间逐级矩阵 |
| `recovery_ghost_interval` | training-free | 每个最大连续 component 一个 | 残差线性回归矩阵 | component source 后置矩阵 |
| `recovery_ghost_interval_sequential` | training-free | 每个最大连续 component 一个 | 当前 source 到当前模型 target 的残差线性回归 | component source 后置矩阵 |
| `recovery_ghost_interval_sequential_grouped` | training-free | 每个最大连续 component 一个 | camera/register 共享矩阵，patch 使用独立矩阵 | 分组 boundary 矩阵 |
| `recovery_replaceme_ls` | training-free | 每个最大连续 component 一个 | source MLP residual 的最小二乘变换 | 融合进 source MLP `fc2` |
| `recovery_replaceme_cos` | training-free | 每个最大连续 component 一个 | 当前 source/residual 到当前模型 target 的 cosine 变换 | 融合进 source MLP `fc2` |
| `recovery_streamline_ffn` | local-fit | 每个最大连续 component 一个 | source 到原模型 component target 的 MSE | source boundary 后置 4x-ReLU FFN |

`analysis.sh` 当前默认生成以下 cache-fit 方法：

```text
recovery_linearpatch,recovery_ghost,recovery_ghost_interval,
recovery_ghost_interval_sequential,recovery_ghost_interval_sequential_grouped,
recovery_replaceme_ls,
recovery_replaceme_cos,recovery_streamline_ffn
```

`recovery_lora` 和 `recovery_train` 仍受支持，但需要显式加入 `RECOVERY_METHODS`。

## 2. 公共流程与符号

对最大连续删除区间 `[i,j]`，公共 boundary 是：

```text
source = unit_ids[i - 1]
target = unit_ids[j]
```

记原模型 source/target activation 展平后的行为 `X,Y in R^(N x C)`。training-free
方法从 minimal cache 读取所需 activation，以分块方式在 GPU 上累计统计量或优化，最后
产生 FP32 变换。原有 LinearPatch、Ghost 和 ReplaceMe 均使用一个 `C x C` 矩阵；
token-grouped Ghost 按 token 分组分别拟合，保存为
`[G,C,C]`（只有一类时仍为 `[C,C]`）。boundary operator 在 source 输出处用
FP32 做矩阵乘法，再转回原 residual-stream dtype。

多 component 的 sequential 路径会重新前向，并从逐步修改后的当前模型同时采集 source
和 target。该语义用于 `recovery_ghost_interval_sequential`、其
token-grouped 变体，以及恢复为官方行为的 `recovery_replaceme_cos`。

ReplaceMe 还读取 source unit 的 MLP residual：

```text
R = replaceme_residual:<source_unit_id>
```

它不安装额外 hook，而是把拟合变换融合到 source block 的 MLP `fc2`；存在 LayerScale
时会先换算为与原 `gamma` 等价的权重变换。

所有方法都依赖 `recovery_plan_all.json`：

```text
schema_version = recovery_plan_bundle_v3
```

cache-fit entry 只保存方法名，boundary 在与具体 prune selector 组合时动态派生；
trainable entry 还保存 LoRA 参数和每个 span 的训练/cache 配置。

## 3. Trainable recovery

### 3.1 `recovery_lora`

对删除区间 `[i,j]`，anchor 从 `i-1` 向前取 `lora_prefix` 个可用 unit。每个 anchor
只在 attention 的 `qkv` 和 `proj` 注入 LoRA：

```text
plan name = recovery_lora_l<L>_r<R>
execution = lora_prune
```

其中 `L` 是 anchor 前缀长度，`R` 是 LoRA rank；当前 plan builder 的默认值是
`rank=32`、`alpha=64`、`dropout=0`。

训练使用 cache-first replay。当前总损失为：

```text
loss = w_align * loss_align + w_pred * loss_pred + w_reg * loss_reg
```

`loss_align` 把 anchor 的 student activation 与删除区间右端 `j` 的 teacher activation
对齐，内部可组合 MSE、cosine 和 CKA；`loss_pred` 使用 backbone replay engine 的预测
损失，默认 target 来自 teacher prediction。具体权重和 optimizer 由 backbone replay
engine 提供。

适用场景：允许训练，希望用少量增量参数恢复中间表示和最终预测。代价是需要多 epoch
反向传播，并且目标 backbone 必须实现 cache replay engine。当前 VGGT、VGGT-Omega 和
Pi3 均已实现。

### 3.2 `recovery_train`

anchor 选择与 `recovery_lora` 相同，但不注入 LoRA，而是解冻 anchor unit 的原始全部
参数：

```text
plan name = recovery_train_l<L>
execution = train_original
```

它与 LoRA 共用 cache-first replay、prediction loss、optimizer 和 checkpoint 训练流程。
需要特别注意：当前 activation alignment 循环由 `prune_plan.lora` 驱动，而
`recovery_train` 的 `prune_plan.lora` 为空，所以该方法当前没有 LoRA anchor alignment
项；实际训练主要由 prediction loss 驱动，若 adaptor 配置了正则项则再加 regularization
loss。

与 LoRA 相比，它能修改完整 anchor block，但 trainable 参数和 checkpoint delta 更大。

## 4. Training-free recovery

### 4.1 `recovery_linearpatch`

LinearPatch 不做梯度训练。令 `Q` 为正交 Hadamard 基，先按 calibration rows 统计各
变换通道的绝对值总量：

```text
s[c] = sum(abs(Y @ Q)[:,c]) / (sum(abs(X @ Q)[:,c]) + 1e-8)
W    = Q @ diag(s) @ Q.T
```

推理时输出 `X @ W`。模型宽度为 2 的幂时使用归一化 Sylvester Hadamard；非 2 的幂
宽度会构造到下一个 2 的幂，截取方阵后用 QR 得到正交基。

单一连续 component 只拟合一个 boundary。若 drop set 含多个 component，当前实现退化
为每个被删物理 unit 一个 boundary，而不是每个 component 一个。

### 4.2 `recovery_ghost`

Ghost 为每个被删物理 unit 保留一个独立变换。它拟合 source 到 target 的残差：

```text
(X.T @ X + ridge * I) @ Delta = X.T @ (Y - X)
W = I + Delta
```

即以 ridge 最小二乘近似 `X @ W = Y`。当前默认 `ridge=1e-6`。

若连续删除多个 unit，boundary 是 `[k,k]` 序列。中间 source 可能也是被删除 unit；
prune 后的 identity/skip block 仍作为 operator 挂载点，因此这些矩阵按物理执行顺序逐级
作用。该方法故意不把逐 unit operator 压成单个 interval operator，以保留逐级演化的
中间状态。

### 4.3 `recovery_ghost_interval`

拟合公式与 Ghost 相同，区别只在 boundary 粒度：每个最大连续删除 component 直接拟合
一个 `source -> target` 变换。多个 component 的各 boundary 都从原模型 disk cache
独立拟合，不考虑前面已剪枝或已恢复的 component 对后续 activation 的影响。

### 4.4 `recovery_ghost_interval_sequential`

该方法在多 component 时按 progressive 顺序处理；每个 component 都在剪掉当前区间之前，
从当前演化模型同时采集 boundary 两端：

```text
X_current = 当前模型 source 输出
Y_current = 当前模型 target 输出
Delta = Y_current - X_current
(X_current.T @ X_current + ridge * I) @ B = X_current.T @ Delta
W = I + B
```

随后才剪掉当前 component 并在 source 上安装 `W`。因此它恢复的是“删去当前 component
前的当前模型”。第一个 component 上当前模型就是原模型，故仍可直接复用 Full-model
disk cache；第二个及后续 component 动态采集两端。

### 4.5 `recovery_ghost_interval_sequential_grouped`

该变体保持相同的 sequential current-source/current-target 拟合流程，但把每帧 token
划分为 `special/patch` 两组。对 VGGT 和 VGGT-Omega，`special` 同时覆盖 camera token
和全部 register tokens，因此每个 boundary 只拟合两个矩阵：

```text
W_special -> camera + register
W_patch   -> patch
```

若不存在 register，camera 仍归入 `special`。Pi3 只有 `patch`，所以与非 grouped
sequential Ghost 完全相同并直接链接其结果。

### 4.6 `recovery_replaceme_ls`

ReplaceMe 不直接变换整个 source 输出，而是替换 source MLP residual 的线性作用。记
source unit 原输出为 `H`、target 输出为 `Y`、source MLP residual 为 `R`。融合变换
`A` 后，source 输出变为：

```text
H - R + R @ A
```

因此 LS 拟合目标是：

```text
D = Y + R - H
(R.T @ R + alpha * I) @ A = R.T @ D
```

当前 `alpha` 默认是 `0`。每个最大连续 component 独立使用原始 cache 拟合。若某些
global block 的 MLP 只更新 register/camera token 前缀，代码会把 `H`、`Y` 对齐到与
`R` 相同的 token 前缀。

### 4.7 `recovery_replaceme_cos`

输入和期望值仍是 `R` 与 `D=Y+R-H`，但不解正规方程，而是从单位阵初始化 `A`，用
Adam 最小化：

```text
loss = 1 - mean(cosine(R @ A, D))
```

当前 shell 默认参数是 `epochs=10`、`batch_size=1024`、`lr=1e-4`、`seed=0`。数据会
优先物化为连续 GPU FP32 tensor；显存不足时回退 CPU，再不足时保留 scene-sharded
存储。

当只有一个 boundary 时从原始 disk cache 拟合。当存在多个 component 时，该方法按
ReplaceMe 官方 pipeline 的语义自动切换到 front-to-back sequential 模式：在当前模型
上同时采集 `H_current/R_current/Y_current`，拟合并融合当前变换，剪掉当前 component，
再处理下一个。因此后续 boundary 的目标是：

```text
D = Y_current + R_current - H_current
```

这里 `Y_current` 是删去当前 component 前、已经包含前序删除和恢复的当前模型输出，
不是原 Full model cache 中的 `Y_full`。因此 `recovery_replaceme_cos` 虽然名称中没有
`sequential`，多区间语义实际是官方的 current-source/current-target sequential。
该语义使用新的 checkpoint provenance 版本；完成性检查会让旧 Full-target cosine 的
多区间结果重新拟合，但因单区间两种目标完全等价，旧单区间结果仍会自动复用。

### 4.8 `recovery_streamline_ffn`

该方法按 LLM-Streamline 官方实现中的 FFN 结构重建 replacement module：

```text
Linear(C, 4C, bias=True) -> ReLU -> Linear(4C, C, bias=True)
```

对每个最大连续删除 component `[i,j]`，从现有 recovery cache 读取
`X=output(i-1)` 与 `Y=output(j)`，用 AdamW 最小化 `MSE(FFN(X),Y)`。默认设置尽量沿用
官方 FFN 示例：1 epoch、`lr=2e-4`、`min_lr=5e-6`、`weight_decay=1e-3`、
`betas=(0.9,0.95)`，并使用 3% warmup 后的余弦学习率。由于本项目按 activation row
采样而不是按 2048-token 文本序列采样，默认 row batch size 设为 16384。实现参考官方仓库
`examples/mlp_mseloss_train_without_memory_issue/modeling_llama.py`；由于该仓库未附许可证，
本项目保留相同网络与目标语义，但使用自己的 cache、训练循环和 checkpoint 接口。

它不使用原模型 task loss，也不需要 label；checkpoint 保存 FFN 参数 delta，评测时在
source unit 输出处重建并安装 FFN。多个 component 分别使用 Full-model cache 拟合，部署时
按前向顺序自然串联。

## 5. 多区间行为对照

| 方法 | 多区间拟合数据 | 是否受前面 recovery 影响 |
|---|---|---|
| LinearPatch | 原模型 cache；逐被删 unit | 否 |
| Ghost | 原模型 cache；逐被删 unit | 拟合时否，部署时矩阵逐级作用 |
| Ghost interval | 原模型 cache；逐 component | 否 |
| Ghost interval sequential current-target | 第一个可用 disk cache；后续当前 `X/Y` | 是（source 和 target） |
| Ghost interval sequential special/patch | 当前 `X/Y`；camera/register 共享矩阵，patch 独立矩阵 | 是（source 和 target） |
| ReplaceMe LS | 原模型 residual cache；逐 component | 否 |
| ReplaceMe LS sequential current-target | 第一个优先 disk cache；后续当前 `H/R/Y` | 是（source 和 target） |
| ReplaceMe cosine | 第一个使用 disk cache；后续当前 `H/R/Y`；逐 component | 是（source 和 target） |
| Streamline FFN | 原模型 cache；逐 component 的 `X/Y` boundary MSE | 拟合时否，部署时会串联 |

trainable 两种方法不走上述 boundary-by-boundary 闭式拟合，而是按 recovery bundle 中该
span 的 anchor 配置统一 replay 和反向传播。目前 trainable span record 来自
`prune_plan_all.json` 的 `spans`，不是任意 strategy 的通用多-component 配置。

## 6. Ghost 派生产物复用

当前 runner 有一条产物级快捷规则：如果 drop set 是一个连续 component，且同 selector
下的 `recovery_ghost` 已完成，那么：

```text
recovery_ghost_interval
recovery_ghost_interval_sequential
```

会直接把自己的输出目录创建为指向 `recovery_ghost` 的相对符号链接，不再独立拟合。
这条判断要求“只有一个连续 component”，但不要求该 component 只含一个 unit。因此在
默认 `prune_spanK` 流水线里，比较这两种方法前应先检查输出目录是否为链接；只有没有
触发该复用时，上述 interval 与 sequential 拟合差异才会实际执行。

此外，Pi3 的 grouped Ghost 方法会在任意区间拓扑下链接到对应非 grouped sequential
方法，因为 Pi3 的 token layout 只有一个 `patch` 组；这条规则不仅复用 recovery
checkpoint，也复用各数据集的 eval 目录。旧的 Pi3 grouped 独立目录不再被视为最终完成
产物；下次 runner 处理它时会用上述相对符号链接替换。

## 7. Cache、显存与计算特点

| 方法 | 额外 cache | 主要持久统计/参数 | 是否反向传播 |
|---|---|---|---|
| LoRA / Train | replay seed、teacher units、teacher prediction 与几何 metadata | trainable model、optimizer | 是 |
| LinearPatch | source、target | `O(C)` 统计，最终 `C x C` | 否 |
| Ghost 系列 | source、target | 两个 FP64 `C x C` 正规方程矩阵 | 否 |
| ReplaceMe LS / LS sequential 系列 | source、target、source MLP residual | 两个 FP64 `C x C` 正规方程矩阵 | 否 |
| ReplaceMe cosine | 同 LS | `C x C` 参数及 activation row storage | 仅对变换 `A` 反向传播 |

training-free activation 默认每次最多向 GPU stage `16384` rows，可用
`RECOVERY_CHUNK_ROWS` 修改。这个分块只约束 activation workspace，不改变 Ghost/LS 的
`C x C` FP64 统计矩阵。

## 8. Plan 名称、运行与产物

方法名与 plan 名：

```text
recovery_lora  -> recovery_lora_l<L>_r<R>
recovery_train -> recovery_train_l<L>
training-free  -> 与方法名相同
```

典型运行方式：

```bash
BACKBONES=vggt \
DATASETS=scannet_dev50 \
PRUNE_PLAN_NAME=prune_span10 \
RECOVERY_PLAN_NAME=recovery_ghost \
EVAL=scannet_test50 \
bash scripts/run_recovery.sh
```

可用空格或逗号同时指定多个 prune plan / recovery plan。training-free jobs 会先汇总到
一个 process session，以共享已加载 cache 和可复用统计；trainable 方法各自进入 cache
replay 训练脚本。

输出位置：

```text
analysis_outputs_v3/<backbone>/recovery/<prune_selector>/<recovery_plan>/
```

共同关键文件是：

```text
config.json
checkpoints/latest.pt
```

training-free checkpoint 保存 `recovery_specs`、FP32 `recovery_weights`、prune plan 和
拟合诊断，`optimizer_state_dict` 为空。trainable checkpoint 保存当前可训练参数的 delta、
optimizer/scaler state，并按配置额外保存 `epoch_<N>.pt`。

评估时必须同时提供原 backbone、同一 prune selector、同一 recovery plan 和对应
checkpoint。只复制 `latest.pt` 而遗漏 plan bundle，可能导致 drop topology、operator
安装位置或训练参数集合不一致。

## 9. 选择方法时的实际区别

- 需要最低拟合成本时，LinearPatch 只做幅值统计；Ghost/LS 要解 `C x C` 线性系统。
- 希望一个连续删除段只放一个显式 boundary operator 时，使用 interval Ghost；普通
  Ghost 保留逐 unit 演化。
- 多个分离区间存在明显前后耦合时，sequential Ghost、token-grouped sequential Ghost、
  ReplaceMe LS sequential 或 ReplaceMe cosine 会在当前模型上拟合后续区间，但计算量
  更高并需要重新前向。这些 sequential 方法均以删去当前区间前的当前模型为 teacher。
- 希望不增加推理 hook/module 时，ReplaceMe 把变换融合进 source MLP；其前提是 adaptor
  能为该 block 暴露兼容的 MLP `fc2` 和 residual capture。
- 允许训练且 backbone 有 replay engine 时，LoRA 的增量参数少；`recovery_train` 的自由度
  更高，但会解冻完整 anchor unit。

论文、表格和可视化中应同时记录 recovery plan 名、prune selector、calibration datasets、
checkpoint、是否触发 Ghost 符号链接复用，以及 cosine/sequential 超参数；只写“Ghost”或
“ReplaceMe”不足以唯一描述当前实验。
