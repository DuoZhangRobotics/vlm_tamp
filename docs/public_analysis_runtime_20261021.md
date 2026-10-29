# Learner analysis-runtime defect — 21 October 2026

Status: repaired and integrated after the original frozen block completed.
All 47 targeted tests passed. A fresh Luna/high session successfully used the
actual tool interface to import the libraries, rotate points, solve a linear
system and save/reload NPZ data. The separately registered corrected block
completed as `results/reflexion-checkpoint-study-20261021/study-v2`, with no
NumPy/SciPy import failures in its six actors. A subsequent reviewer-context
repair brought the targeted suite to 60 passing tests.

The public robot documentation says Python provides NumPy, SciPy and Pillow.
The restored server's isolated learner actually used `/usr/bin/python3` 3.12.3,
which had Pillow 10.2.0 but neither NumPy nor SciPy. The numerical robot backend
used a different, functioning environment. Earlier preflights checked the backend
and filesystem isolation without verifying the learner's documented imports.

This was an observed implementation defect. In the first Reflexion round, both
the no-new-memory actor and the record-only actor attempted NumPy computations
and received `ModuleNotFoundError`. A wider initial scan found the same error in
18 of 65 checked server-run logs, including earlier practice/transfer episodes,
checkpoint diagnostics and verification-prompt responses. Absence of an import
error in a log does not prove that the library was available; it may never have
been requested. The evidence is retained in
`results/reflexion-checkpoint-study-20261021/runtime-import-audit-private.json`.

The physical success predicates and recorded trajectories remain valid measured
outcomes. These runs do not measure what the actors could do with the analysis
environment they were promised, and this defect alone does not establish the
cause of any particular success or failure. Original results and failures are
preserved rather than discarded.

## Replacement and boundary checks

A separate public bundle contains pinned CPython 3.12 wheels for NumPy 1.26.4
and SciPy 1.14.1. It has 2,181 registered files, with manifest SHA-256
`0a66ac52b6adbb3c2f2475072490f4b9ad4c260d6afe6ba4126280dfe1bb5f9d`.
The bundle contains ordinary third-party numerical libraries; it contains no
robot source, recovery policy, private observations or credentials. It is copied
to a read-only `.analysis_site` inside an isolated workspace. The public import
path replaces the inherited project import path. Both initial and resumed
sessions receive that path. Setup now checks the documented capabilities before
calling a model, as well as checking that private files are inaccessible.

The Codex shell environment supports explicit variables through
`shell_environment_policy.set`; the scoped public import path is applied to
initial and resumed learner sessions through that setting.
[Configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).

`analysis-runtime-probe002` passed in the same OS sandbox used by learners:
array linear algebra, SciPy rigid rotation, compressed NPZ writing/reading,
Pillow pixels, read-only dependency enforcement, and denial of direct and symlink
reads of private robot source. Probe 001's numerical operations also worked,
but its check expected `PermissionError`; the sandbox correctly raised the
read-only-filesystem error instead. Its failed receipt and original source are
retained. The probe accepts either appropriate OS denial now. Six focused unit
tests also passed, including path, hash and symlink boundaries.

No numerical robot tool, physics parameter, collision rule, success predicate,
learner recovery hint or frozen learner memory was changed by this repair.
The original six-episode block completed before integration. Its 497 registered
source hashes passed immediately before the repair. The exact five-file patch,
before/after hashes and post-collection note are retained separately.

The generic Luna smoke took 10.79 seconds and involved no robot operations or
task examples. Its numerical tool command completed successfully. `rg` was not
available in that sandbox; Luna used `ls` instead. That optional search utility
is distinct from the now-verified numerical analysis dependencies.

## Future agents: setup and verification

Use the existing registered bundle while reproducing these experiments. Do not
silently rebuild or replace it during a frozen block. Normal subscription
launchers now prepare the public copy and check it before model access.

On this server, if the bundle is absent, recreate it with the validated Python
3.12 interpreter and pinned wheels. Registration intentionally refuses to
overwrite an existing payload:

```bash
source scripts/server_runtime_env.sh
python -m pip --python /usr/bin/python3 install --only-binary=:all: --no-compile --no-user \
  --target .runtime/public-analysis-cp312-v1/packages \
  --report /tmp/vlm-public-analysis-install-report.json numpy==1.26.4 scipy==1.14.1
python -c 'from vlm_tamp.public_analysis import DEFAULT_BUNDLE, register_bundle; register_bundle(DEFAULT_BUNDLE, "/tmp/vlm-public-analysis-install-report.json")'
python scripts/probe_public_analysis.py --output results/new-public-analysis-probe
```

Use a new output directory for the probe. On a different Python version or
platform, build and validate a matching bundle explicitly; this one contains
CPython 3.12 Linux wheels. Do not make the private robot environment readable to
the actor to work around missing dependencies.

When reviewing results, check `isolation.json` for the `analysis_runtime` probe,
and inspect actual actor command errors. A successful simulator/backend smoke
and a successful private-file-denial test are insufficient evidence that the
learner can execute its documented analysis code.
