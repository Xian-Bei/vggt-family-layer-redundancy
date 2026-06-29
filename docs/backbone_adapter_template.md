# Backbone Adapter 模板使用说明

新增 backbone 时，复制代码模板：

```bash
cp analysis/backbones/adapters/backbone_adapter_template.py analysis/backbones/adapters/<name>_adapter.py
```

然后把 `NewBackboneAdapter` 改名为 `<Name>Adapter`，把 `name = "new_backbone"` 改成公共 backbone 名，并在 `analysis/backbones/registry.py` 的 `_load_defaults()` 中导入和 `register()`。

模板不会被默认导入，也不会影响现有 `vggt`、`vggt_omega`、`pi3`。

## 公共路径只认识的东西

公共代码只操作：

- `unit_id`：只能是 `frame_i` 或 `global_i`。
- `PredictionBundle`：统一相机、深度、点云字段；某个 backbone 不支持的字段必须返回 `None`。
- `PruneAction`：`drop_unit` 和 `lora_unit`。

所有内部映射都放在 adaptor 内部。公共路径不应该知道 `unit_id` 对应哪个 `aggregator`、`decoder`、`block list`、head 或 vendor 模块。

## 必填能力

`capabilities()`

返回 backbone 名称、是否支持 cache/prune/lora/eval，以及 `num_frame_blocks`、`num_global_blocks`。这两个数量应该和 `layer_unit_catalog()` 对齐。

`default_hf_model()` / `pythonpath_entries()` / `load_pretrained()`

负责导入 vendor 代码并加载模型。默认建议权重保持 fp32，由公共路径用 autocast 控制推理精度。若请求 CUDA，不要静默降级到 CPU。

`layer_unit_catalog()`

返回所有可操作单元的 `LayerUnitSpec`。`module_path` 是 adaptor 内部可解释的路径，公共路径只会使用 `unit_id`。

`block_for_unit()` / `apply_unit_operation()`

把 `unit_id` 转换成具体模块，并支持：

- `drop_unit`：替换为保持 shape 的 identity/skip。
- `lora_unit`：向该单元注入 LoRA，并返回 optimizer 需要训练的参数。

`apply_prune_plan()` 和 `inject_lora_plan()` 可以直接沿用模板里的循环，只要 `apply_unit_operation()` 实现正确即可。

`extract_layer_units()`

用于 cache/redundancy。输入通常是 `[B, S, 3, H, W]`，返回：

- `unit_map[unit_id]`: `[B, S, P, C]`
- `meta["patch_start_idx"]`
- 可选 `meta["aggregated_concat"]`

如果 vendor 一次 forward 可以同时拿 activation 和 prediction，可以实现 `extract_layer_units_and_predictions()`，否则保持返回 `None`。

`forward_predictions()` / `prediction_bundle_from_raw()`

`forward_predictions()` 跑完整推理，`prediction_bundle_from_raw()` 把 raw output 归一化为：

- `pose_enc`
- `camera_poses`
- `depth`
- `depth_conf`
- `points`
- `local_points`
- `conf`
- `raw`

例如 Pi3 不预测 dense depth，就令 `depth=None`、`depth_conf=None`，但仍然返回 `points/camera_poses`。

`build_initial_tokens()`

返回 cache-first replay 的 `layer_-1` seed，形状 `[B, S, P, C]`。如果该 backbone 暂不支持从 cache seed replay，可以返回 `None`，但 LoRA cache training 需要它或等价 replay engine 能力。

`validate_cached_prediction()` / `build_target_batch_from_teacher()` / `prediction_loss()`

align loss 属于公共训练路径，prediction loss 属于 adaptor。即使多个 backbone 共用同一套 loss，也要在各自 adaptor 里提供接口，然后委托公共 helper。

`required_head_layers_for_cache()` / `replay_student_cache_first()` / `cache_replay_engine()`

用于 LoRA cache-first training。公共训练只传 unit id、seed layer、target layers 和缓存读取函数；具体怎么从某个起点 replay 到 heads，由 adaptor 或 `analysis/backbones/cache_replay/<name>.py` 处理。

`eval_scene()`

用于四个数据集的统一 eval。推荐内部调用 `forward_predictions()` 得到 `PredictionBundle`，再复用公共 eval helper。`periodic_eval_skip_depth()` 用来声明是否跳过深度指标。

## 精度和 autocast 约定

公共 eval/cache 路径会使用 `analysis.amp_policy.autocast_device_dtype()` 包住 forward。adaptor 内部通常不要把整模型转成 bf16。

如果官方实现要求某些 head 或 loss 保持 fp32，可以只在这些局部禁用 autocast；不要为了保护 head 而把整个 aggregator 的 autocast 关掉。Pi3 就是这种情况：aggregator/decoder 可在 bf16 autocast 下跑，head 和部分几何后处理保持 fp32。

## 新增后的最小测试

建议给新 backbone 增加至少以下测试：

- `get_backbone("<name>")` 能注册并返回 adaptor。
- `layer_unit_catalog()` 的数量、`unit_id` 和 `capabilities()` 一致。
- `apply_unit_operation(..., "drop_unit")` 不改变前后张量 shape。
- `inject_lora_plan()` 返回非空 trainable parameters。
- `forward_predictions()` 返回合法 `PredictionBundle`。
- `forward_cache_batch()` 返回 activation、prediction、meta，且 activation shape 为 `[B, S, P, C]`。
- `prediction_loss()` 可对一批 teacher/student prediction 计算有限标量。
