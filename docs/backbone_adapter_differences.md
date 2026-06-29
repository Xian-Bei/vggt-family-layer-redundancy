# 骨干网络适配器差异

LaRe 的独立 backbone 只有 `vggt`、`pi3` 和 `vggt_omega`。公共阶段通过
`analysis/backbones/contracts.py` 与 registry 调用它们，不按模型名称直接分支。
FastVGGT 与 QuantVGGT 不属于本表；它们是恢复 LaRe 模型后应用的部署压缩，见
[`inference_compression.md`](inference_compression.md)。

## 能力矩阵

| 能力 | VGGT | VGGT-Omega | Pi3 |
|---|---|---|---|
| 缓存 unit | `frame_i/global_i` | `frame_i/inter_frame_i` 映射为公共 unit | decoder 配对映射为公共 unit |
| 冗余、删层、LoRA | 支持 | 支持 | 支持 |
| 训练 | VGGT cache replay | Omega replay + 9D pose | points/conf replay |
| 默认权重 | `facebook/VGGT-1B` | 本地 VGGT-Omega checkpoint | `yyfz233/Pi3` |
| vendor 路径 | `backbones/vggt` | `backbones/vggt-omega` | `backbones/Pi3` |

实现分别位于 `analysis/backbones/adapters/`，训练回放位于
`analysis/backbones/cache_replay/`。所有模型保持 fp32 权重，并在公共前向路径使用
设备适配的 autocast。

## Eval 模式

- `upper`：完整预训练模型。
- `lower`：仅应用 prune plan。
- `model`：应用 prune plan、恢复 LoRA topology，并加载训练 checkpoint。

`run_eval.sh` 可在上述任一模式之后应用一个 inference compression adapter。部署压缩
不参与 cache、冗余分析、prune-plan 生成或 LoRA 训练。

## 最小验证

```bash
conda run --no-capture-output -n vggt python - <<'PY'
from analysis.backbones.registry import list_backbones
from analysis.inference_compression.registry import list_inference_compressions
assert list_backbones() == ['pi3', 'vggt', 'vggt_omega']
assert list_inference_compressions() == ['fastvggt', 'lora_merge', 'quantvggt']
PY
```
