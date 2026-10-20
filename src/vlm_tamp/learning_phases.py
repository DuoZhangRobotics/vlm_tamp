"""Generic practice objectives released according to observed episode outcomes."""

SUCCESS_INSTRUCTIONS = '''Current objective: establish repeatable simulator success.
If an approved successful method is supplied, read it and reproduce its qualitative
procedure using freshly observed geometry and fresh plans. Preserve its tested
relative control choices as starting hypotheses, not universal constants. Do not
change the method merely to reduce tool use, increase trajectory length or save
time. If reproduction fails, diagnose that failure and seek success first.
Do not infer success from appearance alone; use the simulator's success boolean.
Record why each necessary change was made. After success, reset and repeat the
method. Ask practice_status for the current phase after finishing an attempt.
'''

EFFICIENCY_INSTRUCTIONS = '''Two consecutive simulator-successful episodes now
establish a provisional baseline in this scene. Preserve that working method and
its evidence. Efficiency is now a soft objective: test whether you can retain
success while reducing planning calls, other tool calls, control steps and total
elapsed time. A longer useful manipulation trajectory can reduce replanning, but
trajectory length itself is not a reward. Approach and closure remain separate
with this interface. Let observations determine which changes are appropriate.
Record failed plans and failed attempts as costs. Compare a candidate with the
baseline and confirm a promising cheaper method from a fresh reset before
preferring it. Keep the baseline when candidates regress. Record measured tool
time separately from whole-attempt time, which includes reasoning and pauses.
You may stop after investigating a useful efficiency hypothesis when further
feasible improvements are not justified, or at the budget limit. Explain the
comparison and uncertainty; success alone does not establish an efficiency gain.
Use practice/attemptNN/efficiency.json for candidate and baseline measurements.
This practice does not establish transfer to a different scene or mechanism.
'''


def learning_phase(outcomes):
    previous = None
    pair = []
    for row in outcomes:
        if row.get('official_task_success'):
            if previous is not None:
                pair = [previous, row['attempt']]
                break
            previous = row['attempt']
        else:
            previous = None
    return {'phase': 'efficiency' if pair else 'success_first',
            'verified_baseline_attempts': pair,
            'instructions': EFFICIENCY_INSTRUCTIONS if pair else SUCCESS_INSTRUCTIONS,
            'basis': 'Recorded simulator success booleans; no geometry or target thresholds.'}
