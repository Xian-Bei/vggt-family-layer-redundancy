# Backbone Adaptor 公共契约

本文档定义目标形态：公共流程不再按 `vggt`、`vggt_omega`、`pi3` 分支，而是只调用每个 backbone adaptor 暴露的接口。新增 backbone 时，只需要实现这些接口，并注册到 `analysis/backbones/registry.py`。FastVGGT 与 QuantVGGT 属于推理时压缩，使用 `analysis/inference_compression` 下的独立 adaptor contract。

可复制的代码骨架见 `analysis/backbones/adapters/backbone_adapter_template.py`，逐项填充说明见 `docs/backbone_adapter_template.md`。

## 总体原则

公共流程只理解三类抽象：

- `LayerUnit`：统一的交替结构单元，命名为 `frame_i`、`global_i`。
- `PredictionBundle`：统一预测结果，包含相机、深度、点云；不支持的字段为 `None`。
- `ReplayOps`：训练 / skip eval 回放时运行一个 frame block 或 global block 的最小操作集合。

backbone adaptor 负责把模型内部结构映射到这些抽象。例如 VGGT-Omega 的 `inter_frame_blocks` 映射为 `global_i`，Pi3 的 decoder 偶数/奇数 block 映射为 `frame_i/global_i`。

## 公共路径

### 1. 加载模型

公共代码：

```python
adapter = get_backbone(backbone_name)
model = adapter.load_pretrained(
    hf_model=hf_model or adapter.default_hf_model(),
    device=str(device),
    dtype=None,
    eval_mode=True,
)
```

adaptor 接口：

```python
def default_hf_model(self) -> str

def pythonpath_entries(self, repo_root: Path) -> list[Path]

def load_pretrained(
    self,
    *,
    hf_model: str,
    device: str,
    dtype: torch.dtype | None = None,
    eval_mode: bool = True,
) -> nn.Module
```

返回：

- `default_hf_model`：默认权重来源，可以是 HF id 或本地 checkpoint 路径。
- `pythonpath_entries`：vendor 包导入路径。
- `load_pretrained`：已移动到目标设备的模型；权重默认保持 fp32，公共流程用 autocast 控制前向精度。

### 2. 描述可操作单元

公共代码：

```python
units = adapter.layer_unit_catalog(model)
unit_ids = [u.unit_id for u in units]
```

adaptor 接口：

```python
def capabilities(self) -> BackboneCapabilities

def layer_unit_catalog(self, model: nn.Module) -> list[LayerUnitSpec]
```

返回：

- `BackboneCapabilities.name`
- `supports_cache`
- `supports_prune`
- `supports_lora`
- `supports_eval`
- `num_frame_blocks`
- `num_global_blocks`
- 每个 `LayerUnitSpec` 包含 `unit_id`、`kind`、`index`、`module_path`、`droppable`、`lora_target`

约定：

- `unit_id` 只能是 `frame_i` 或 `global_i`。
- 公共 prune plan、redundancy、cache、train 都只使用 `unit_id`，不直接访问 `model.aggregator` 或 `model.decoder`。

### 3. cache / redundancy 前向

公共代码：

```python
result = adapter.forward_cache_batch(model, images, device)
unit_map = result.unit_map
pred = result.pred
meta = result.meta
initial_tokens = result.initial_tokens
```

adaptor 接口：

```python
def forward_cache_batch(
    self,
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> CacheForwardResult

def extract_layer_units(
    self,
    model: nn.Module,
    images: torch.Tensor,
    unit_ids: set[str] | None = None,
    layer_callback: Any | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]

def extract_layer_units_and_predictions(
    self,
    model: nn.Module,
    images: torch.Tensor,
) -> CacheForwardResult | None

def forward_units_and_prediction(
    self,
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    unit_ids: set[str] | None = None,
) -> CacheForwardResult

def build_initial_tokens(
    self,
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor | None
```

返回：

```python
@dataclass
class CacheForwardResult:
    unit_map: dict[str, torch.Tensor]
    meta: dict[str, Any]
    pred: PredictionBundle
    initial_tokens: torch.Tensor | None = None
```

字段约定：

- `unit_map["frame_i"]` 和 `unit_map["global_i"]` 均为 `[B, S, P, C]`。
- `meta["patch_start_idx"]` 为 special token / patch token 分界。
- `meta["aggregated_concat"]` 可选，若存在每层为 `[B, S, P, 2C]`。
- `initial_tokens` 是 replay 从 layer 0 前开始时的 `layer_-1` seed，形状 `[B, S, P, C]`；不需要时为 `None`。

说明：

- `extract_layer_units_and_predictions` 是可选的 fused path，用于一次 forward 同时拿 activations 和 prediction。
- `forward_units_and_prediction` 是公共默认入口：优先调用 fused path；如果 fused path 返回 `None`，则先调用 `extract_layer_units`，再调用 `forward_predictions`。
- `build_initial_tokens` 应由 adaptor 提供，公共 cache 脚本不应按 backbone 名手写 patch embedding。

### 4. 预测归一化

公共代码：

```python
bundle = adapter.forward_predictions(model, images)
```

adaptor 接口：

```python
def forward_predictions(
    self,
    model: nn.Module,
    images: torch.Tensor,
) -> PredictionBundle

def prediction_bundle_from_raw(
    self,
    raw: dict[str, Any],
    images: torch.Tensor,
) -> PredictionBundle
```

返回：

```python
@dataclass
class PredictionBundle:
    pose_enc: torch.Tensor | None = None
    depth: torch.Tensor | None = None
    depth_conf: torch.Tensor | None = None
    points: torch.Tensor | None = None
    local_points: torch.Tensor | None = None
    conf: torch.Tensor | None = None
    camera_poses: torch.Tensor | None = None
    raw: dict[str, Any] = field(default_factory=dict)
```

字段语义：

- `camera_poses`：统一为 c2w-like `[B, S, 4, 4]` 或 `[S, 4, 4]`。
- `points`：世界坐标点云或全局点云，形状通常为 `[B, S, H, W, 3]`。
- `depth`：深度图；Pi3 不预测 dense depth，返回 `None`。
- `pose_enc`：VGGT 系列 raw pose encoding；Pi3 可以为 `None`。
- `raw`：保留官方模型原始输出，供调试或 backbone 特定 loss 使用。

推荐公共 helper：

```python
def bundle_from_pose_depth(
    *,
    raw: dict[str, Any],
    images: torch.Tensor,
    backbone_name: str,
) -> PredictionBundle
```

VGGT、VGGT-Omega 只需要把 raw dict 交给这个 helper。Pi3 adaptor 自己把 `points/local_points/conf/camera_poses` 填入 `PredictionBundle`，`depth=None`。

### 5. 剪枝和 LoRA

公共代码：

```python
drop_actions, lora_actions = prune_actions_from_plan(plan)
adapter.apply_prune_plan(model, drop_actions)
trainable = adapter.inject_lora_plan(model, lora_actions)
```

adaptor 接口：

```python
def block_for_unit(self, model: nn.Module, unit_id: str) -> nn.Module

def apply_unit_operation(
    self,
    model: nn.Module,
    unit_id: str,
    operation: str,
    **kwargs: Any,
) -> Any

def apply_prune_plan(
    self,
    model: nn.Module,
    actions: Sequence[PruneAction],
) -> nn.Module

def inject_lora_plan(
    self,
    model: nn.Module,
    actions: Sequence[PruneAction],
) -> list[nn.Parameter]

def apply_checkpoint_structure(
    self,
    model: nn.Module,
    payload: dict[str, Any],
    *,
    prune_plan: UnitPrunePlan | None = None,
) -> None
```

输入：

- `PruneAction.action == "drop_unit"`：把对应 `frame_i/global_i` 替换成 identity/skip block。
- `PruneAction.action == "lora_unit"`：在对应单元注入 LoRA。

返回：

- `block_for_unit` 只给 adaptor / backbone-specific replay engine 使用；公共路径不应根据 `unit_id` 推断内部 block 下标。
- `apply_unit_operation` 是 `drop_unit/lora_unit` 的单元级入口，`apply_prune_plan` 和 `inject_lora_plan` 可以委托它。
- `apply_prune_plan` 返回被修改后的模型。
- `inject_lora_plan` 返回需要 optimizer 训练的参数列表。
- `apply_checkpoint_structure` 负责先恢复 skip/LoRA 结构，再加载权重。

### 6. cache-first 训练回放

公共训练 loop 不直接写 VGGT/Omega/Pi3 的循环；cache-first replay 由 adaptor 或 adaptor 返回的 replay engine 提供。

adaptor 接口：

```python
def required_head_layers_for_cache(
    self,
    model: nn.Module,
    num_layers: int,
) -> list[int]

def replay_student_cache_first(
    self,
    model: nn.Module,
    get_layer_cpu: Callable[[int], torch.Tensor],
    num_layers: int,
    **kwargs: Any,
) -> Any

def cache_replay_engine(self) -> Any
```

公共训练路径：

```python
adapter = get_backbone(plan.backbone)
engine = adapter.cache_replay_engine()
required = adapter.required_head_layers_for_cache(model, num_layers)
student_pred, activations = adapter.replay_student_cache_first(
    model,
    get_layer_cpu,
    num_layers,
    target_layers=required,
    seed_layer=seed_layer,
    **adapter.train_replay_extra_kwargs(h=h, w=w),
)
```

各 backbone 的差异落点在 adaptor / `analysis/backbones/cache_replay/*` 内部。公共训练代码只传 `unit_id`、cache batch、seed layer 和 target layers。

### 7. 训练目标和 loss

训练目标确实存在 backbone 差异，因此作为独立能力保留。

adaptor 接口：

```python
def validate_cached_prediction(
    self,
    pred: dict[str, torch.Tensor],
) -> None

def build_target_batch_from_teacher(
    self,
    teacher_pred: dict[str, torch.Tensor],
    *,
    images_batched: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]

def prediction_loss(
    self,
    student_pred: dict[str, torch.Tensor],
    target_batch: dict[str, Any],
    *,
    weights: PredictionLossWeights,
) -> tuple[torch.Tensor, dict[str, float]]
```

语义：

- VGGT 系列：teacher 需要 `pose_enc/depth/depth_conf`，loss 是 camera + depth。
- Pi3：teacher 需要 `points/camera_poses`，loss 是官方 Pi3 point/camera 蒸馏；`depth` 缺失是合法状态。

### 8. eval

公共代码：

```python
metrics = adapter.eval_scene(
    model,
    dataset_name=dataset_name,
    batch=batch,
    ctx=ctx,
)
```

adaptor 接口：

```python
def periodic_eval_skip_depth(self) -> bool

def eval_scene(
    self,
    model: nn.Module,
    *,
    dataset_name: str,
    batch: Any,
    ctx: EvalSceneContext,
) -> dict[str, float]
```

返回：

- 每个 scene 的指标 dict。
- 公共 `eval_v3` 负责保存 `metrics.json` 和汇总 `eval_record_v3.json`。

推荐公共 eval 路径：

- 所有 backbone 先通过 `forward_predictions` 得到 `PredictionBundle`。
- `eval_impl` 根据 `PredictionBundle` 统一计算 pose、point、depth 指标。
- `ctx.skip_depth=True` 时跳过深度指标，例如 Pi3。

## 新增 backbone checklist

1. 新建 `analysis/backbones/adapters/<name>_adapter.py`。
2. 实现 `default_hf_model`、`pythonpath_entries`、`load_pretrained`。
3. 实现 `layer_unit_catalog`，把内部 blocks 映射成 `frame_i/global_i`。
4. 实现 `extract_layer_units` 或 fused `extract_layer_units_and_predictions`。
5. 实现 `forward_predictions` 和 `prediction_bundle_from_raw`。
6. 实现 `build_initial_tokens`，用于 `layer_-1` replay seed。
7. 实现 `apply_prune_plan` 和 `inject_lora_plan`。
8. 实现 `required_head_layers_for_cache`、`replay_student_cache_first`、`cache_replay_engine`，让公共训练能够从 cache seed replay。
9. 实现 teacher target 和 prediction loss；若可复用 VGGT 系列 loss，直接委托公共 helper。
10. 实现或复用 `eval_scene`。
11. 在 `analysis/backbones/registry.py` 注册 adaptor。
12. 增加最小单测：unit catalog、cache result shape、PredictionBundle 字段、prune/lora action 映射。

## 当前实现状态

当前 `analysis/backbones/contracts.py` 已提供上述接口。公共 cache、training cache unit、cache replay registry、LoRA cache-first training、skip eval 均通过 adaptor 获取能力；直接访问 `aggregator/decoder` 的逻辑被限制在 adaptor 或 backbone-specific replay engine 内部。
