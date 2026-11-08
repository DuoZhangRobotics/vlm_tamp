# Persistent tool documentation pilot

This experiment changes only whether Luna retains a tool schema after reading its help.
The existing actor can discard help when motion clears the temporary feedback tail, or when
later read-only calls displace it. The new condition keeps one exact schema per consulted
tool in `previously_consulted_tool_contracts`, separately from episode memories. It introduces
no examples, task state, action proposals, success labels, or extra model calls. The default
behavior remains unchanged while the explicit opt-in is compared.

The declared comparison uses fresh Luna/high actors for microwave closing states 17 and 49,
drawer closure state 0, and bowl placement state 0. Each case has one transient-documentation
and one persistent-documentation episode, with condition order alternating between cases.
Both use two native cameras, observed-cloud views, hand previews, eight anchored interaction
memories, 600 controls including reset, and 1800 seconds. Numerical backend files remain
unchanged from `e2c9860`. No practice memory, demonstrations, or generated policy code is supplied.

Before live execution, four recorded public contexts are selected deterministically: the first
post-motion repeated help request whose earlier schema has disappeared from the immediate
tail, within each predeclared source run. Fresh Luna decisions receive either the original
context or that context plus only schemas already consulted at the recording time. Their
returned requests are never executed. These deliberately selected cases test documentation
use, not representative decision accuracy or physical task success.

The live comparison measures native success, control use, wall time, model/client time,
robot-tool time, help requests and repeated help by tool. All attempts are retained. Eight
episodes are a development pilot, not evidence of a reliable improvement across tasks.
The existing memory/I/O/SSH watchdog and 32 GiB whole-job RAM limit remain in force.

Results and exact model inputs are published in
[watch.html](/common/users/dz323/vlm_tamp/results/luna-tool-documentation-20261107/watch.html).
The page uses manual refresh and writes study summaries at episode boundaries.

Implementation checks: 26 focused tests, configured Ruff checks, numerical-file hash comparison,
and verification that the two conditions differ only in documentation retention. Recorded
and live outcomes will be added after the declared comparisons finish.
