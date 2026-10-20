# Short planned motions and observed hinge refinement

The microwave runner now defaults to `--motion-policy short`. It keeps the
partial-pose backend and its free wrist orientation. It does not demand a
complete 90-degree path before collecting motion observations.

Before an observed hinge fit is accepted, each requested pull is at most
3 degrees, with complete 2- or 1-degree plans as fallbacks. After an accepted
fit is applied, the horizon is at most 8 degrees. Every selected path finishes
before RGB-D is observed again. There are no execution-time collision,
constraint, tracking-error or progress vetoes.

The configured hinge fitter still requires multiple nondegenerate observed
handle positions. `hinge-calibration.json` records their control steps, fit
conditioning, acceptance and whether refinement was used. Predicted handle
motion can update a nominal scene when depth tracking is unavailable, but it
is excluded from the observed hinge history. A successful fit is an intermediate
measurement; native LIBERO predicates determine task success.

Calibration uses four endpoint hold controls per short plan. Subsequent
refined-model plans use six. The episode remains limited to 600 controls.
The existing post-run cleanup keeps opened/latest sensor archives, videos,
source snapshots, outcomes and necessary numeric geometry.

```bash
PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.runtime/envs/rekep-libero/bin/python scripts/run_mcvamp_microwave.py \
  --output results/my-short-motion-test \
  --tasks open_microwave --states 11 \
  --backend partial_pose_arc --motion-policy short --reconcile-handle-roles \
  --replay-grasp-source results/mcvamp-microwave-frozen-20261015/open_microwave-state011 \
  --arc-proposals 32 --arc-seconds 20
```

The replay flag is a targeted development diagnostic: the original approach and
physical closure are reused after matching RGB, proprioception, predicted
handle cloud and planning admission. The original 20-degree pull is not
executed. Omit the flag to use fresh grasp selection.

`--motion-policy full` or `--full-task-only` retains the complete remaining-arc
experiment. `--motion-policy legacy --backend endpoint_menu` selects the earlier
endpoint-menu policy and its full-task/20-degree/10-degree search order. Old
results retain their original source snapshots and protocols.
