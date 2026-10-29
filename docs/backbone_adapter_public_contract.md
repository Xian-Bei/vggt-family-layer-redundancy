# Backbone Adaptor 公共契约

本文档定义当前 backbone adaptor contract。公共流程不按 `vggt`、`vggt_omega`、
`pi3` 分支，而是调用 registry 中 adaptor 暴露的
能力。新增 backbone 时实现相应接口并注册到 `analysis/backbones/registry.py`。
FastVGGT、LoRA Merge 与 QuantVGGT 属于推理时压缩，使用
`analysis/inference_compression` 下的独立 contract。

可复制的代码骨架见 `analysis/backbones/adapters/backbone_adapter_template.py`，逐项填充说明见 `docs/backbone_adapter_template.md`。

## 总体原则

公共流程主要理解以下抽象：

- `LayerUnitSpec`：按实际执行顺序排列的可操作单元，命名为 `frame_i`、`global_i`。
- `PredictionBundle`：统一预测结果，包含相机、深度、点云；不支持的字段为 `None`。
- `CacheForwardResult`：unit activation、metadata、prediction 与可选 initial tokens。
- `CacheReplayEngine`：backbone-specific cache-first 训练回放。
- `DatasetAdaptor`：scene 读取、CPU 预处理、prediction scoring 与 dataset-specific GT。

backbone adaptor 负责把模型内部结构映射到这些抽象。例如 VGGT-Omega 的
`inter_frame_blocks` 映射为 `global_i`，Pi3 decoder 映射为 `frame_i/global_i`。
线性 unit index 永远是 `layer_unit_catalog()` 返回列表的位置，不能从 unit 名称奇偶
关系推导。

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
- `num_frame_blocks`、`num_global_blocks`（静态数量已知时填写；动态 checkpoint 以
  `layer_unit_catalog()` 为准）
- 每个 `LayerUnitSpec` 包含 `unit_id`、`kind`、`index`、`module_path`、`droppable`、`lora_target`

约定：

- `unit_id` 只能是 `frame_i` 或 `global_i`。
- catalog 顺序必须是实际物理执行顺序；不要求所有 backbone 都 frame-first。
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
- `initial_tokens` 只在 adaptor 实际返回真实 unit-0 输入时使用；非 fused fallback 可以为
  `None`，需要起点 replay 的调用者必须显式调用 `build_initial_tokens()`。

### 3.1 增量 skip eval

当前五个已注册 backbone 都实现 scene-local 增量 skip replay：

```python
def supports_incremental_skip_eval(self) -> bool

def iter_skip_span_predictions(
    self,
    model: nn.Module,
    images: torch.Tensor,
    spans: Sequence[tuple[int, int]],
    *,
    defer_cpu_geometry: bool = False,
) -> Iterable[tuple[int, int, PredictionBundle]]
```

`spans` 使用闭区间。实现可复用每个 scene 的 prefix/intermediate state，但必须逐个
yield 与完整物理 skip 等价的标准化 prediction。公共 skip-eval 不应知道具体 replay
循环或 block 容器。

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
    *,
    defer_cpu_geometry: bool = False,
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
- `defer_cpu_geometry=True`：异步 eval 可推迟 CPU-only 几何后处理，由 dataset scorer
  在消费 prediction 时完成。

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

### 4.1 Eval 输入与 GT 对齐

模型输入预处理仍属于 backbone adaptor，因为不同官方模型的 resize、normalization、
patch grid 和相机约定不同：

```python
def load_eval_images(
    self,
    image_paths: Sequence[Path | str],
) -> torch.Tensor

def load_driving_depth_stack(
    self,
    depth_paths: Sequence[Path | str],
    *,
    output_hw: tuple[int, int],
) -> tuple[Any, int]

def load_scannet_gt_depth(
    self,
    scene: Any,
    *,
    output_hw: tuple[int, int],
) -> Any

def align_7andn_gt_depth(
    self,
    view: Any,
    *,
    output_hw: tuple[int, int],
) -> Any
```

`load_eval_images()` 返回 `[S,3,H,W]`、范围 `[0,1]` 的模型输入。dataset adaptor
调用这些能力准备 scene，但不复制 backbone-specific resize/geometry 逻辑。

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
replay = engine.replay_student_cache_first(
    model,
    get_layer_cpu,
    num_layers,
    alignment_capture_layers=required,
    seed_layer=seed_layer,
    **replay_kwargs,
)
student_pred = replay.pred
activations = replay.concat_by_layer
```

各 backbone 的差异落点在 adaptor / `analysis/backbones/cache_replay/*` 内部。公共训练
代码传 cache reader、skip intervals、LoRA/alignment layers、seed 与 device state，不直接
执行 vendor blocks。

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
    dataset_name: str = "",
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

训练超参数和 optimizer 也由 backbone adaptor/replay engine 提供：

```python
def training_config(self) -> TrainingHyperparams

def build_training_optimizer(
    self,
    trainable: Sequence[nn.Parameter],
    *,
    config: TrainingHyperparams | None = None,
) -> torch.optim.Optimizer
```

`dataset_name` 允许 adaptor 对不同 teacher target 约定做显式校验，公共 loop 不按
backbone 或 dataset 名称复制 loss 分支。

### 8. Eval 的职责边界

`BackboneAdapter` 不再提供 `eval_scene()`。当前调用关系是：

```python
dataset = get_dataset(dataset_name)
metrics = dataset.evaluate_scene(adapter, model, batch, ctx)
```

支持异步 scoring 时，dataset adaptor 将一次 scene 拆为：

```python
prepared = dataset.prepare_eval_scene_cpu(adapter, batch, ctx)
prediction = adapter.forward_predictions(
    model,
    prepared.images,
    defer_cpu_geometry=True,
)
metrics = dataset.score_eval_prediction(adapter, prediction, prepared, ctx)
```

职责分工：

- backbone adaptor：官方图像预处理、模型 forward、`PredictionBundle`、必要的 GT resize/
  geometry helper；
- dataset adaptor：scene 枚举、GT state、pose/point/depth metric 与 prediction scoring；
- `analysis.eval.eval_v3`：调度、异步 pipeline、每 scene `metrics.json` 和
  `eval_record_v3.json`。

## 新增 backbone checklist

1. 新建 `analysis/backbones/adapters/<name>_adapter.py`。
2. 实现 `default_hf_model`、`pythonpath_entries`、`load_pretrained` 和
   `load_eval_images`。
3. 实现 `layer_unit_catalog`，把内部 blocks 映射成 `frame_i/global_i`。
4. 实现 `extract_layer_units` 或 fused `extract_layer_units_and_predictions`。
5. 实现 `forward_predictions` 和 `prediction_bundle_from_raw`。
6. 实现 `build_initial_tokens`，用于 `layer_-1` replay seed。
7. 实现 `apply_prune_plan` 和 `inject_lora_plan`。
8. 实现 `required_head_layers_for_cache`、`replay_student_cache_first`、`cache_replay_engine`，让公共训练能够从 cache seed replay。
9. 实现 `training_config`、optimizer、teacher target 和 prediction loss；若可复用
   VGGT 系列 loss，委托公共 helper。
10. 实现 `supports_incremental_skip_eval` / `iter_skip_span_predictions`，或明确保持
    fallback 不支持状态。
11. 根据 dataset 需求实现 driving/ScanNet/7Scenes-NRGBD GT 对齐 helper。
12. 在 `analysis/backbones/registry.py` 注册 adaptor。
13. 增加最小单测：unit catalog/执行顺序、cache shape、PredictionBundle、prune/LoRA、
    incremental skip 等价性和 dataset preprocessing。

## 当前实现状态

当前 `analysis/backbones/contracts.py` 已提供上述接口。公共 cache、training cache unit、
cache replay registry、cache-first training 和 skip eval 均通过 adaptor 获取能力；
dataset-specific eval 通过 `analysis/datasets/contracts.py` 获取能力。直接访问
`aggregator/decoder/info_sharing` 的逻辑应限制在 adaptor 或 backbone-specific replay
engine 内部。
