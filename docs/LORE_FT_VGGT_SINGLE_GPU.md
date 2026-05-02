# VGGT 单卡 LoRA 微调方案（参考官方 fine-tuning，但为我们的改动定制）
  
本文件记录的是**我们的微调方案**：在“参考官方 `training/` fine-tuning 默认策略”的基础上，结合我们当前的改动与约束来设计训练配置与 loss。
  
我们的改动/目标包括：
- **结构改动**：会删除部分 `aggregator` 层（结构变更意味着更需要稳健的优化与 loss 设计）
- **参数高效微调**：注入 **LoRA 参数**并微调
- **输出头**：仅保留 **camera + depth**（不做 tracking；point head 也不做，直接删除不用）

约束与现实测量（用于指导 batch/序列长度设置）：
- **单卡 48GB 显存**
- 当前观察到：`batch_size=1` 且 **序列 30 张图**时，训练显存 **>40GB**，接近上限

我们最终采用的关键决策（请以此为准）：
- **冻结剩余 `aggregator`**（不训练其原参数）
- **只训练新增的 LoRA 参数**
- **head 有两种实验设置**：A) 冻结 head；B) 训练 head（camera/depth）
- **固定序列长度**：统一 **30 张图片**（不做动态长度采样）
- **图片预处理**：保持 VGGT 默认读取方式（不额外改 resize/crop 逻辑；仅按 dataloader 默认配置走）
- **先不做梯度累积**：`accum_steps` 暂不作为调参维度

目标：尽量沿用官方训练框架的 **优化器/调度、AMP、梯度裁剪、loss 的 mask/skip/outlier 过滤与 confidence 用法**，并把它们调整为适配“删 aggregator 层 + LoRA + 48GB 单卡 + 固定 30 帧序列”的可落地方案。
  
相关实现/配置入口（请以此为准）：
- 训练入口：`training/launch.py`
- 默认配置（fine-tuning）：`training/config/default.yaml` + `training/config/default_dataset.yaml`
- 训练循环：`training/trainer.py`
- loss：`training/loss.py`
- 动态 dataloader：`training/data/dynamic_dataloader.py`
- 数据组装：`training/data/composed_dataset.py`
- scheduler/optimizer wrapper：`training/train_utils/optimizer.py`
- freeze：`training/train_utils/freeze.py`
- gradient clip：`training/train_utils/gradient_clip.py`
  
---
  
## 0. 参考基线：官方 fine-tuning 默认配置（我们以此为对照）
  
仓库文档明确：`training/config/default.yaml` **就是 fine-tuning 默认配置**（resume 预训练 ckpt，并冻结 `aggregator` 模块）。
  
因此本文所说“官方默认/参考基线”主要指：
- AdamW + warmup→cosine（用 `where∈[0,1]` 驱动）
- camera/depth loss 的具体实现与权重（camera=5，depth=1）
- dynamic batch（`max_img_per_gpu` 控制每卡每步总图片数上限）
- AMP（bfloat16/float16）+ GradScaler
- 梯度裁剪按模块分组、并要求覆盖所有可训练参数

---
  
## 1) 我们的单卡 LoRA 微调：训练哪些参数（最终方案）
  
### 训练参数集合（固定）
- **LoRA 参数**：只对你注入 LoRA 的层启用 `requires_grad=True`
- **其余所有原生参数**：冻结（尤其是剩余 `aggregator`）

### head 的两种实验设置（只影响 head 参数的 requires_grad）
- **设置 A：冻结 head（更像“纯参数高效适配”）**
  - 训练参数 = LoRA only
  - 适用：你主要想验证“删层后的主干 + LoRA”是否能保持性能/做小幅域适配
- **设置 B：训练 head（更像“LoRA + 轻量头部重配”）**
  - 训练参数 = LoRA +（camera head + depth head）
  - 适用：删 aggregator 层后，head 可能需要重新适配输出分布；这时训练 head 往往更稳更快
  
### 冻结策略（最终）
- **冻结剩余 `aggregator`**（我们不训练其原生参数）
- backbone/encoder 等其它原生参数同样冻结
- 仅 LoRA（以及可选 head）参与更新
  
> 冻结实现见 `training/train_utils/freeze.py`（glob 匹配 module name，并锁定 train/eval + requires_grad）。
  
---
  
## 2) 数据与 batch 语义（固定 30 帧序列；不使用 dynamic length）
  
### 我们不采用 dynamic length 的原因

你已实测：`batch_size=1` 且序列 30 张图时显存 >40GB，已经接近 48GB 上限。为了避免训练过程中随机抽到更“极端”组合导致 OOM/不稳定，我们采用：
- **固定序列长度：30 张图**
- `batch_size_per_gpu` 基本恒为 1（除非你后续显存优化后再调大）

> 这意味着我们不再依赖 `max_img_per_gpu/img_nums` 的动态 batch 语义来稳定显存，而是用“固定序列长度”直接把最重的变量锁死。

###（参考）官方 dynamic batch 的语义仍值得了解

（下述是参考基线，便于未来恢复动态策略时对照。）

#### 关键点：`max_img_per_gpu` 控的是“每卡每步总图片数”，不是 batch size
  
官方 dataloader 每一步会随机采样：
- 序列长度 `S=image_num`（训练默认来自 `[2,24]`）
- 宽高比 `aspect_ratio`（训练默认来自 `[0.33,1.0]`）
  
并计算每卡的 **序列条数**（batch size per GPU）：
  
\[
\text{batch\_size\_per\_gpu}=\left\lfloor\frac{\text{max\_img\_per\_gpu}}{S}\right\rfloor,\quad \ge 1
\]
  
对应实现：`training/data/dynamic_dataloader.py` 的 `DynamicBatchSampler`。
  
---
  
## 3) 训练循环要点
  
### AMP（Automatic Mixed Precision）
- 目的：**省显存/提速**（bfloat16/float16）
- 实现：`training/trainer.py` 使用 `torch.cuda.amp.autocast(...)` + `GradScaler`
  
### 梯度裁剪（Gradient clipping）
- 目的：防止偶发大梯度导致发散
- 实现：`training/train_utils/gradient_clip.py`
  - 支持按模块名分组 clip
  - **强约束**：所有 `requires_grad=True` 参数必须被某个 clip 规则覆盖，否则报错（避免漏配）
  
---
  
## 4) Optimizer / 学习率 / schedule（参考官方 fine-tuning，按单卡 LoRA 做保守起步）
  
### Optimizer（官方默认）
- **AdamW**
- `weight_decay=0.05`
  
### 学习率（建议起点：更偏保守）
  
官方默认 `lr=5e-5`。
  
> 学习率与有效 batch（`max_img_per_gpu × GPU数 × accum_steps`）强相关。单卡下通常更敏感。
  
### Schedule（官方默认 保留）
  
官方采用 `where∈[0,1]` 表示训练进度，并用 fvcore scheduler 组合：
- **前 5%：Linear warmup**（`1e-8 → lr`）
- **后 95%：Cosine decay**（`lr → 1e-8`）
  
配置位置：`training/config/default.yaml` 的 `optim.options.lr`。
  
---
  
## 5) Prediction 与 Loss 设计（复用官方 camera/depth；删除 point）
  
`training/loss.py` 的 `MultitaskLoss` 会根据模型输出的 key 自动启用分支。我们的方案中只保留：
- camera：需要 `pose_enc_list`
- depth：需要 `depth` 与 `depth_conf`

我们不做：
- point head（直接删除不用）
- tracking（不启用）
  
### 5.1 Camera loss（pose）
  
#### 多阶段（multi-stage）监督
  
`pose_enc_list` 是一次 forward 内部产生的多阶段输出（粗→细），loss 对 stage 加权：
  
\[
w_k=\gamma^{(n\_stages-k-1)}\quad(\text{final stage 权重最大})
\]
  
官方默认：`gamma=0.6`。
  
#### 有效帧 mask（来自 GT）
  
camera loss 用 `point_masks` 判定序列是否有足够几何信号：
- `valid_frame_mask = point_masks[:,0].sum(...) > 100`
  
其中 `point_masks` 来自 dataset（GT 侧有效深度/点云 mask），不是预测的 confidence。
  
#### camera loss 类型与分解
- 官方默认 **L1**（比 smoothL1/L2 稳）
- 分解：`loss_T`（平移）+ `loss_R`（旋转）+ `loss_FL`（焦距/内参部分）
  - 默认权重：`weight_trans=1, weight_rot=1, weight_focal=0.5`
  
#### 任务权重（官方默认）
- `camera.weight = 5.0`
  
---
  
### 5.2 Depth loss（depth + confidence）
  
#### mask/skip 规则（来自 GT）
- mask 使用 `point_masks`（depth 生成 3D 点，因此与点 mask 一致）
- 若 `mask.sum() < 100`，该 batch 返回 dummy 0 loss（跳过，避免污染梯度）
  
#### confidence 的用法（关键式）
  
对 mask 内每点：
- `loss_reg = ||pred - gt||`
- `loss_conf = gamma * loss_reg * conf - alpha * log(conf)`
  
默认参数：`gamma=1.0, alpha=0.2`（`loss.py`）。
  
> 注意：不是只“加权”，还有 `-log(conf)` 正则项，防止模型把 conf 全压小来逃避回归。
  
#### 梯度/平滑项与多尺度
- 官方默认 `gradient_loss_fn="grad"`，并通过 `gradient_loss_multi_scale_wrapper` 做多尺度下采样约束
- `valid_range=0.98`：对 `loss_conf` 与 `loss_reg` 做分位数过滤，剔除极端 outlier（稳定训练）
  
#### 任务权重（官方默认）
- `depth.weight = 1.0`
  
Depth 总合成：`loss_conf_depth + loss_reg_depth + loss_grad_depth`，再乘 `depth.weight`。
  
---
  
## 6) 我们的一套“建议起始值”（单卡 48GB + 删 aggregator 层 + LoRA + 固定 30 帧）
  
请把下列视作“起点”，再按显存与稳定性微调：
  
- **数据（固定）**
  - `fix_img_num`: **30**
  - `batch_size_per_gpu`: **1**（在 48GB 下基本是现实约束）
  - 图片预处理：按 VGGT 默认读取方式（不额外修改）
  
- **训练**
  - AMP：开（bfloat16 优先）
  - 梯度裁剪：开，`max_norm=1.0`
  
- **优化**
  - AdamW，`weight_decay=0.05`
  - `lr`: **3e-5** 起步（稳定后可到 5e-5；不稳降到 1e-5~2e-5）
  - schedule：保留 warmup(5%)+cosine(95%)
  
- **loss（任务权重）**
  - camera: 5.0（参考官方）
  - depth: 1.0（参考官方）
  - depth：`valid_range=0.98`（参考官方）
  - depth：`gradient_loss_fn="grad"`（参考官方）
  
