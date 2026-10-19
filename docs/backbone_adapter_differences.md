# 骨干网络适配器差异

LaRe 当前 registry 中有五个公共 backbone：`vggt`、`pi3`、`vggt_omega`、
`mapanything` 和 `mapanything_v1`。`main.sh` 的默认实验集合仍是前三个；
MapAnything 两个版本需要通过 `BACKBONES` 显式选择。公共阶段通过
`analysis/backbones/contracts.py`、dataset adaptor contract 和 registry 调用它们，
backbone 内部结构差异收敛在 adaptor 内。

FastVGGT、LoRA Merge 与 QuantVGGT 不是独立 backbone，而是恢复模型后应用的部署
变换；支持矩阵见 [`inference_compression.md`](inference_compression.md)。

## 能力矩阵

| 能力 | VGGT | VGGT-Omega | Pi3 | MapAnything | MapAnything V1 |
|---|---|---|---|---|---|
| 公共 unit | `frame_i/global_i` | `frame_i/inter_frame_i` 映射为公共 unit | decoder 偶/奇 block 映射为 `frame_i/global_i` | information-sharing block 映射为 `global_i/frame_i` | 同 MapAnything |
| 第一个物理 unit | `frame_0` | `frame_0` | `frame_0` | `global_0` | `global_0` |
| 冗余、删层、LoRA | 支持 | 支持 | 支持 | 支持 | 支持 |
| 增量 skip eval | 支持 | 支持 | 支持 | 支持 | 支持 |
| cache-first recovery 训练 | VGGT replay | Omega replay + 9D pose | 官方 point/camera replay | 尚未提供 replay engine | 尚未提供 replay engine |
| 默认权重 | `facebook/VGGT-1B` | 本地 VGGT-Omega checkpoint | `yyfz233/Pi3` | `facebook/map-anything` | `facebook/map-anything-v1` |
| vendor 路径 | `backbones/vggt` | `backbones/vggt-omega` | `backbones/Pi3` | `backbones/map-anything` | `backbones/map-anything` |

公共 unit 的线性下标始终由 `layer_unit_catalog(model)` 返回列表中的实际执行顺序决定，
不能在公共代码中假设一定是 `frame_0, global_0, ...`。该假设只适用于当前
VGGT、VGGT-Omega 和 Pi3 adaptor；MapAnything 是 global-first。

实现位于 `analysis/backbones/adapters/`，训练回放位于
`analysis/backbones/cache_replay/`。模型通常保留 fp32 master weights，公共前向路径
根据设备使用 autocast；局部 head 或几何计算可由 adaptor 保持 fp32。

## Eval 模式

- `upper`：完整预训练模型。
- `lower`：仅应用 prune plan。
- `model`：应用 prune plan、恢复 checkpoint 中的 prune/LoRA/recovery topology，并加载权重。

MapAnything adaptor 支持 LoRA 注入，但当前没有 `cache_replay_engine()`，所以不能把
“支持 LoRA operation”理解为已支持步骤 9 的 cache-first trainable recovery。

场景准备和指标计算属于 `DatasetAdaptor`；backbone adaptor 负责图像预处理和
`PredictionBundle` 推理。`scripts/run_eval.sh` 可在模型恢复后应用一个 inference
compression adapter。部署压缩不参与 cache、冗余分析、prune-plan 生成或训练。

## 最小验证

```bash
conda run --no-capture-output -n vggt python - <<'PY'
from analysis.backbones.registry import list_backbones
from analysis.inference_compression.registry import list_inference_compressions

assert list_backbones() == [
    'mapanything', 'mapanything_v1', 'pi3', 'vggt', 'vggt_omega'
]
assert list_inference_compressions() == ['fastvggt', 'lora_merge', 'quantvggt']
PY
```

项目测试必须在沙箱外、`vggt` conda 环境和 GPU 上运行；上述命令只检查注册表，
不加载模型。
