# Environment and collection notes

The session ran on Ubuntu 22.04.4 under WSL2, with an RTX 4090 (24 GB). The task brief's Ubuntu 24.04 target should be validated separately. Existing Miniforge was located at `/home/duo/miniforge3`; `vlm-tamp` is a new environment.

Python 3.10 accommodates official LIBERO's NumPy 1.22.4 wheel and avoids untested legacy upgrades. The simulator uses Robosuite 1.4.0, MuJoCo 2.3.7, Gym 0.25.2, BDDL 1.0.1, SciPy 1.10.1 and Numba 0.57.1. Pinning MuJoCo avoids installing a newer API into legacy Robosuite. `termcolor` was an undeclared Robosuite dependency found by the real smoke test and is included explicitly. LIBERO's training requirements (transformers, wandb, robomimic, hydra) are unnecessary for collection and are omitted.

Upstream LIBERO's outer namespace directory is omitted by its `find_packages()` setup on modern editable installs. `configure_libero()` adds the explicit checked-out root before importing the official package. This preserves upstream simulator code. It also writes a project-local `.runtime/libero/config.yaml` before import, preventing LIBERO's interactive dataset-path question and avoiding modification of another project's `~/.libero` configuration.

Headless EGL succeeded on the actual machine. Both `MUJOCO_GL` and the collection `--gl` option support `egl`, `glfw`, or `osmesa`, with EGL selected by default. Run with `DISPLAY` unset to verify that no visible window or display server is required:

```bash
env -u DISPLAY -u WAYLAND_DISPLAY python scripts/smoke_libero.py --gl egl --output data/headless-check
```

Only EGL was verified. `glfw` needs a working display; `osmesa` needs an OSMesa installation and is a diagnostic fallback, not an automatically selected substitute. Driver/system-library setup is outside the Conda package lock. Renderer startup failures should be fixed and re-tested before inference. CPU-only Torch does not establish whether EGL is hardware-accelerated; the collector does not claim GPU rendering speed.

LIBERO's published initial-state files are loaded through its official API. That API uses Torch serialization; load only the pinned, trusted official benchmark files. Full demonstrations are not downloaded. Collection saves the exact initial state and its hash, BDDL hash, benchmark commit, camera transform and image convention. Camera images and masks use identical vertical flipping. Instance IDs follow the mapping in Robosuite's segmentation sensor and were checked against that source and visible overlays. Only visible whole-object masks become 2D references.

The built-in sandbox could not start commands in this session (process creation returned “No such file or directory”). The same Linux commands succeeded using reviewed execution outside that sandbox. This was a tool-runtime problem rather than a missing attachment.
