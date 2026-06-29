# VGGT / Pi3 官方训练参数与当前 LoRA 训练实现对比

本文档对比两类实现：

- **官方 full training**：各 backbone 官方训练代码中的 optimizer、scheduler、loss、gradient clipping 等设置。
- **当前 LoRA/cache replay training**：本项目在 prune + LoRA finetuning 阶段通过 backbone adapter / cache replay engine 使用的训练参数。

需要注意：当前训练范式不是官方 full training，而是从缓存中间激活开始 replay，只训练 LoRA 参数。因此本文区分“训练参数已对齐”与“训练范式天然不同”。

## 代码来源

| Backbone | 官方训练代码 | 当前 adapter 配置 | 当前 replay engine |
|---|---|---|---|
| VGGT | `backbones/vggt/training` | `analysis/backbones/adapters/_vggt_base.py` | `analysis/backbones/cache_replay/vggt_family.py` |
| Pi3 | `backbones/training_reference` | `analysis/backbones/adapters/pi3_adapter.py` | `analysis/backbones/cache_replay/pi3.py` |

## 总览

| 项 | VGGT 当前状态 | Pi3 当前状态 |
|---|---|---|
| Optimizer 类型 | 对齐官方：AdamW | 对齐官方：AdamW |
| LR 数值 | 对齐官方：`5e-5` | 对齐官方：`5e-5` |
| Weight decay | 对齐官方：`0.05` | 对齐官方：`0.05` |
| Scheduler 实现 | 已直接使用官方同款 `fvcore` scheduler | 已直接使用 PyTorch `OneCycleLR` |
| Scheduler 调用顺序 | 对齐官方：optimizer step 前更新 lr/wd | 外层 step 前查询，但内部状态同步为 Pi3 官方 step 后 scheduler 语义 |
| Gradient clipping 数值 | 对齐官方：`max_norm=1.0` | 对齐官方：`max_norm=1.0` |
| Prediction loss | 使用从官方 VGGT training loss 移植的 camera/depth loss | 使用从 Pi3 官方 training loss 移植的 `Pi3Loss` |
| 仍不同的部分 | LoRA/cache replay 训练范式不同于 full training | LoRA/cache replay 训练范式不同于 full training |

## VGGT 参数对比

### Optimizer

| 项 | VGGT 官方 | 当前 VGGT 系列 LoRA |
|---|---:|---:|
| Optimizer | `torch.optim.AdamW` | `torch.optim.AdamW` |
| LR | `5e-5` | `5e-5` |
| Weight decay | `0.05` | `0.05` |
| Betas | PyTorch AdamW 默认值 | PyTorch AdamW 默认值 |

官方位置：

- `backbones/vggt/training/config/default.yaml`
- `backbones/vggt/training/train_utils/optimizer.py`

当前位置：

- `analysis/backbones/adapters/_vggt_base.py`

### Scheduler

| 项 | VGGT 官方 | 当前 VGGT 系列 LoRA |
|---|---|---|
| Scheduler 类型 | `fvcore.common.param_scheduler.CompositeParamScheduler` | 同款 `CompositeParamScheduler` |
| Warmup | `LinearParamScheduler(1e-8, 5e-5)` | 同款 |
| Warmup 长度 | `0.05` | `0.05` |
| Decay | `CosineParamScheduler(5e-5, 1e-8)` | 同款 |
| Decay 长度 | `0.95` | `0.95` |
| Interval scaling | `["rescaled", "rescaled"]` | 同款 |
| Weight decay scheduler | `ConstantParamScheduler(0.05)` | 同款 |
| 调用顺序 | 先 `step_schedulers(where)`，再 `optimizer.step()` | 先 `engine.update_optimizer_schedule(...)`，再 `optimizer.step()` |

说明：

- VGGT 官方训练中，`where = exact_epoch / max_epochs`，scheduler 在 optimizer step 之前更新。
- 当前 VGGT LoRA replay engine 现在也使用官方同款 `fvcore` scheduler，并在 optimizer step 前更新 lr / weight decay。
- `where >= 1.0` 时，官方代码跳过 scheduler update；当前实现也避免在 `where == 1.0` 时调用 fvcore scheduler。

### Loss

| 项 | VGGT 官方 | 当前 VGGT 系列 LoRA |
|---|---:|---:|
| Camera loss weight | `5.0` | `5.0` |
| Camera loss type | `l1` | `l1` |
| Depth loss weight | `1.0` | `1.0` |
| Depth gradient loss | `grad` | `grad` |
| Depth valid range | `0.98` | `0.98` |
| Point loss | 默认 `null` | 未作为 prediction loss 使用 |
| Track loss | 默认 `null` | 未使用 |

当前 VGGT prediction loss 通过 `analysis/loss.py` 使用从官方 `backbones/vggt/training/loss.py` 移植的 `compute_camera_loss` 和 `compute_depth_loss`。

### Gradient clipping

| 项 | VGGT 官方 | 当前 VGGT 系列 LoRA |
|---|---|---|
| Clip max norm | `1.0` | `1.0` |
| Clip 对象 | `aggregator`、`depth`、`camera` 等完整模块参数 | 当前 LoRA trainable 参数 |
| Clip 时机 | backward 后、optimizer step 前 | backward 后、optimizer step 前 |

数值和时机对齐，但对象不同：官方 full training clip 完整模块；当前只训练并 clip LoRA 参数。

## Pi3 参数对比

### Optimizer

| 项 | Pi3 官方 | 当前 Pi3 LoRA |
|---|---:|---:|
| Optimizer | `AdamW` | `AdamW` |
| LR | `5e-5` | `5e-5` |
| Weight decay | `5e-2` | `5e-2` |
| Betas | `[0.9, 0.95]` | `(0.9, 0.95)` |
| Encoder LR | `5e-6` | 不适用：当前训练 LoRA 参数，不训练 encoder |

官方位置：

- `backbones/training_reference/configs/model/pi3.yaml`
- `backbones/training_reference/trainers/pi3_trainer.py`

当前位置：

- `analysis/backbones/adapters/pi3_adapter.py`

### Scheduler

| 项 | Pi3 官方 | 当前 Pi3 LoRA |
|---|---|---|
| Scheduler 类型 | `torch.optim.lr_scheduler.OneCycleLR` | 同款 `OneCycleLR` |
| `max_lr` | `5e-5` | `5e-5` |
| `pct_start` | `0.0` | `0.0` |
| `anneal_strategy` | `cos` | `cos` |
| `div_factor` | `1000.0` | `1000.0` |
| `final_div_factor` | `100.0` | `100.0` |
| 记录的 `initial_lr` | `5e-8` | 由 `OneCycleLR` 同样设置 |
| 记录的 `min_lr` | `5e-10` | 由 `OneCycleLR` 同样设置 |
| 实际开始 lr | 接近 `5e-5`，因为 `pct_start=0.0` 无 warmup | 同官方 |
| 调用顺序 | `optimizer.step()` 后 `lr_scheduler.step()` | 外层 step 前查询；engine 内部推进 `OneCycleLR` 状态，使当前 optimizer step 使用的 lr 与官方一致 |

说明：

- Pi3 官方 `pct_start=0.0`，因此没有 warmup 阶段。`initial_lr=5e-8` 是 PyTorch `OneCycleLR` 的 metadata，不代表第一个 optimizer step 实际使用 `5e-8`。
- 当前 Pi3 LoRA engine 直接使用 PyTorch `OneCycleLR`，不是手写近似。
- 因为共享训练循环统一在 optimizer step 前调用 `update_optimizer_schedule`，Pi3 engine 内部维护 scheduler 状态，只推进已经完成的 optimizer steps，从而让当前 step 使用的 lr 与 Pi3 官方训练循环一致。

### Loss

| 项 | Pi3 官方 | 当前 Pi3 LoRA |
|---|---|---|
| Point loss | scale-aligned local point L1 | 使用移植版官方 `Pi3Loss` |
| Normal loss | 高/中质量数据集启用 | 使用移植版官方 `Pi3Loss` |
| Camera loss | 相对位姿 loss，内部乘 `0.1` | 使用移植版官方 `Pi3Loss` |
| Confidence loss | `train_conf=True` 阶段启用 | 当前 LoRA prediction loss 默认 `train_conf=False` |
| Depth loss | 无独立 VGGT-style depth loss | 无独立 VGGT-style depth loss，且不暴露 `w_depth` 配置 |

当前 Pi3 prediction loss 通过 `analysis/backbones/cache_replay/pi3_official/loss.py` 使用官方 Pi3 loss 逻辑。

### Gradient clipping

| 项 | Pi3 官方 | 当前 Pi3 LoRA |
|---|---|---|
| Clip max norm | `1.0` | `1.0` |
| Clip 对象 | `self.model.parameters()` | 当前 LoRA trainable 参数 |
| Clip 时机 | backward 后、optimizer step 前 | backward 后、optimizer step 前 |

数值和时机对齐，但对象不同：官方 full training clip 模型全部训练参数；当前只 clip LoRA 参数。

## 当前 LoRA/cache replay 与官方 full training 的系统性差异

以下差异不是“训练参数未对齐”，而是当前方法本身和官方 full training 范式不同。

| 维度 | 官方 full training | 当前 LoRA/cache replay training |
|---|---|---|
| 可训练参数 | 完整模型或官方配置中未冻结的完整模块 | 只训练 prune plan 指定位置注入的 LoRA 参数 |
| 前向起点 | 从真实输入图像完整 forward | 从缓存的中间激活开始 replay，避免重复跑前面层 |
| 剪枝/跳层 | 官方没有 drop unit | 根据 prune plan 将部分 unit 替换为 skip / identity |
| 额外对齐 loss | 官方无 activation alignment loss | 使用 teacher activation 的 MSE / cosine / CKA alignment |
| 监督目标 | 官方 dataset GT | 默认使用 teacher cache；部分路径支持 dataset target |
| DataLoader | 官方动态 dataloader、官方采样和 batch 规则 | cache manifest 中的 scene/cache row 驱动训练 |
| Dummy images | 不需要 | 用于提供 shape、RoPE grid、head/replay 所需输入形状 |
| Total steps | 官方由 epoch、iters_per_epoch 或 dataloader 长度决定 | 由 cache scenes、epochs、`max_steps` 决定 |
| Gradient clipping 对象 | 完整可训练模块 | LoRA trainable 参数 |
| Checkpoint | 保存完整训练状态 | 保存 LoRA/prune 训练产物和 eval 记录 |

## 结论

当前实现的原则是：

1. **Backbone-specific 训练超参尽量对齐官方**：optimizer、scheduler、loss 权重、gradient clipping 数值和时机均按各自官方代码实现。
2. **训练范式保持项目目标**：仍然是 prune + LoRA + cache replay，不追求复现官方 full training 的数据流和完整参数更新。
3. **不同 backbone 使用各自官方 prior**：VGGT 系列使用 VGGT 官方 warmup + cosine；Pi3 使用 Pi3 官方 no-warmup OneCycleLR。
