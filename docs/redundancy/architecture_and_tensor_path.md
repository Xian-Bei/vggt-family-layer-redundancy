# VGGT Architecture and Tensor Path

This note fixes the concrete execution path used by the redundancy toolkit in `analysis/redundancy`.

## Model composition

- Top-level model: `vggt.models.vggt.VGGT`
- Backbone: `vggt.models.aggregator.Aggregator`
- Heads:
  - camera: `vggt.heads.camera_head.CameraHead`
  - depth: `vggt.heads.dpt_head.DPTHead`
  - point: `vggt.heads.dpt_head.DPTHead` (separate instance)
  - track: `vggt.heads.track_head.TrackHead`

## Forward tensor path

1. Input images are shaped `[B, S, 3, H, W]` in range `[0, 1]`.
2. `Aggregator.forward()` normalizes images and computes patch tokens.
3. Camera/register tokens are prepended, then alternating frame/global attention blocks run.
4. Aggregator returns:
   - `aggregated_tokens_list`: list of intermediate tensors `[B, S, P_total, 2C]`
   - `patch_start_idx`: index where patch tokens start (special tokens are before it)
5. Heads consume the list:
   - Camera head reads camera token and outputs `pose_enc_list`.
   - Depth and point heads consume multi-layer features and output dense maps.
   - Track head is optional and requires `query_points`.

## Scripted export

Run the following to dump a reproducible shape report:

```bash
python -m analysis.redundancy.architecture_map --batch-size 1 --seq-len 2 --image-size 518
```

Output JSON defaults to `analysis_outputs/architecture_map.json`.

