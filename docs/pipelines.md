# Experiment entrypoints

The entrypoints keep stage order visible. `scripts/pipeline_lib.sh` contains
common defaults, pruning-evaluation helpers, and recovery-plan construction;
the existing `scripts/run_*.sh` files execute the individual workloads.

The stages that were temporarily commented out are enabled. Stage numbers
are retained for compatibility with research notes; the historical gap at
stage 5 is not a missing executable stage.

| `analysis.sh` stage | Work |
|---|---|
| 1 | Evaluate original models |
| 2 / 2.1 | Enumerate skip intervals; build normalization profiles |
| 3 | Compute similarity/redundancy graphs |
| 4 | Evaluate remaining skip intervals after selected pruning points |
| 6 | Build all-token and special/patch-mean candidate plans |
| 7 | Evaluate candidate plans on analysis training datasets |
| 7.5 | Select pointwise token-region winners |
| 7.6 | Evaluate finalized plans on held-out datasets |
| 8 | Build recovery plans |
| 9 | Generate activation caches, recover, evaluate, and clean caches |

| `main.sh` stage | Work |
|---|---|
| init / init.norm | Import reusable analysis results and build normalization profiles |
| 6 / 7 | Build Retention plans and evaluate missing selected plans |
| 8 | Build recovery plans |
| 9 | Cache four calibration datasets, recover, evaluate, and clean caches |
| 10 | Evaluate original and restored models, deployment compression, and efficiency |

## Parameters

| Variable | Default / meaning |
|---|---|
| `BACKBONES` | `vggt_omega,pi3,vggt` |
| `DEVICE` | `cuda` |
| `MAX_SCENES` | `0`: all scenes; positive values cap each dataset |
| `MAX_FRAMES` | `30` |
| `PLAN_MAX_DROP` | `24` |
| `NORMALIZATION_MAX_DROP` | `48`, independent of the plan budget |
| `ANALYSIS_DATASETS` | `scannet_dev50,nuscenes_train` |
| `CALIB_DATASETS` | `scannet_dev50,7scenes_train,nuscenes_train,waymo_training` |
| `EVAL_DATASETS` | `scannet_test50,7scenes_test,nuscenes_val,waymo_validation` |
| `OUTPUT_ROOT` | `analysis_outputs_v3` for analysis; `main_results` for formal experiments |
| `REFERENCE_RESULTS_ROOT` | `analysis_outputs_v3`, imported by `main.sh` |
| `RECOVERY_PRUNE_PLANS` | Even dropped-unit budgets 10–24 |
| `RECOVERY_FORCE_RERUN` | `0`: retain existing recovery resume/reuse behavior |
| `EPOCHS` / `EVAL_EPOCH` / `SAVE_EVERY_EPOCH` | `25` / `5` / `25` |

VGGT/Pi3/Omega stage-4 analysis points default to `12,16`, `16,20`, and `16,20`.
Explicitly empty point variables disable that backbone in stage 4.

Formal test-time operating points remain VGGT=10, Pi3=16, and Omega=18. The
Single/Double choice follows the calibration seven-error mean; Double must
improve by strictly more than 0.1. Held-out metrics do not select the winner.

`TESTTIME_SKIP_FULL_MODEL_ACCURACY=1` retains reuse of imported full-model
compression accuracy. Set it to `0` to recompute. Benchmarking defaults to
`TESTTIME_BENCHMARK_SKIP_FULL_VARIANTS=0`. QuantVGGT uses the explicitly named
official transfer parameters at `submodules/quantvggt/w4a4_official`; override
`QUANTVGGT_PARAMS_DIR` for calibration matched to a different topology.

## Outputs and caching

Each result root contains `<backbone>/<dataset>/eval/`, backbone-level plan
bundles, and `recovery/<selector>/<method>/checkpoints/`. Recovery generates
only the activation-cache union required by selected plans. The pre/post
recovery cache cleanup in the original scripts is preserved: these caches
are intentionally temporary, while result records and checkpoints persist.

Stage 7 schedules missing physical selectors and reuses complete summaries.
`PRUNING_EVAL_DRY_RUN=1` affects that scheduler only; it does not make the whole
pipeline a dry run. Use a separate `OUTPUT_ROOT` for limited experimental runs.
