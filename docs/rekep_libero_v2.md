# Final strict ReKep/GraspGen runner

The final runner explicitly rejects unknown collision samples, independently of
trilinear distance interpolation. The original frozen experiment and its source
files remain intact. See `rekep_libero_v1.md` for runtime setup and the declared
camera, perception, grasp, planner and controller adaptations.

From `/home/duo/Rutgers/ARC/random/vlm_tamp`:

```bash
/home/duo/miniforge3/envs/vlm-tamp/bin/python scripts/setup_rekep_libero.py --verify-only
PYTHONPATH=src .runtime/envs/rekep-libero/bin/python scripts/run_rekep_libero_v2.py --phase evaluation
PYTHONPATH=src .runtime/envs/rekep-libero/bin/python scripts/report_rekep_libero_final.py
/home/duo/miniforge3/envs/vlm-tamp/bin/python scripts/verify_rekep_libero.py --run results/rekep-libero-v2 --report results/rekep-libero-strict-report
```

This correction uses the existing Luna-high answers with identical scene IDs and
request hashes; it makes no additional API calls. Learned grasp inference and
physical simulator/controller episodes run again. Completed outputs are preserved
and reused on subsequent invocation. To perform a fresh local confirmation, use
a separate versioned output configuration rather than deleting recorded results.

`results/rekep-libero-v1` and `results/rekep-libero-report` preserve the first frozen
run. The corrected run is under `results/rekep-libero-v2` and
`results/rekep-libero-strict-report`. Each has ten evaluation episodes; they are
versions of the same five states per task, not twenty independent evaluation
states. Initial development runs belong to the first version.

An independent replay of all500 retained bowl grasps found three individual tool
samples that were unknown under voxel classification but passed interpolated
clearance. Their grasps were rejected elsewhere. Explicitly blocking those samples
changed no candidate or episode outcome. The corrected class also keeps unknown
finger contacts blocked and preserves the exact known-current robot prior.

Reports distinguish task-program validity, grasp proposal generation, collision
and reachability admission, physical grasp attempts and actual LIBERO completion.
Episodes blocked before grasping do not evaluate complete ReKep manipulation
performance. Grounding can succeed visually while the physical pipeline stops at
the grasp admission stage.
