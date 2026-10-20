"""Summarize the sensor-fit diagnostic retry without changing its raw findings."""
import argparse
import html
import json
import os
from pathlib import Path

from report_subscription_goal import ROOT,read,digest,summarize,prune_unused_depth


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--baseline',default='results/subscription-agent-goal-v1-20261018')
    args=parser.parse_args();run=(ROOT/args.run).resolve();baseline=(ROOT/args.baseline).resolve()
    if not (run/'audit.json').exists():raise ValueError('The run must be complete')
    old=read(baseline/'source-hashes.json');new=read(run/'source-hashes.json')
    core=[name for name in old if name.startswith('src/vlm_tamp/')]
    components_unchanged=all(old[k]==new.get(k) for k in core)
    events=[json.loads(line) for line in (run/'tool-events.jsonl').read_text().splitlines()]
    conflicts=[]
    for event in events:
        measurement=event['response'].get('result',{}).get('measurement',{})
        diagnostic=measurement.get('diagnostics',{})
        if diagnostic.get('sensor_fit_passed_but_initial_pivot_proximity_failed'):
            conflicts.append({'event':event['sequence'],
                'candidate_hinge_world_m':diagnostic['sensor_fit_without_initial_pivot_prior'].get('hinge_world_m'),
                'distance_from_initial_prior_m':diagnostic['candidate_distance_from_initial_pivot_xy_m']})
    images={}
    for name in ['agentview','frontview','wrist']:
        p=baseline/f'attempt01/opened/{name}.png';q=run/f'attempt01/opened/{name}.png'
        if p.exists() and q.exists():images[name]=digest(p)==digest(q)
    comparison={'baseline':summarize(baseline),'fit_diagnostics_retry':summarize(run),
        'same_agent_prompt':digest(baseline/'agent-prompt.txt')==digest(run/'agent-prompt.txt'),
        'same_tool_documentation':digest(baseline/'public-inputs/tools.txt')==digest(run/'public-inputs/tools.txt'),
        'same_example':read(baseline/'protocol.json')['human_supplied_example']==read(run/'protocol.json')['human_supplied_example'],
        'existing_numeric_component_sources_unchanged':components_unchanged,
        'new_feedback_helper':'src/vlm_tamp/subscription_measurements.py',
        'bridge_change':'Opt-in sensor-only diagnostics; original fit acceptance and planner state are preserved.',
        'initial_rgb_identical':images,'prior_conflicts_exposed_to_agent':conflicts,
        'provided_ground_truth_or_replacement_coordinates':False,
        'interpretation':'Development retry, not a causal ablation. Initial agent selections and proposal sampling can differ. No held-out transfer was evaluated.',
        'storage':prune_unused_depth(run)}
    (run/'fit-feedback-comparison.json').write_text(json.dumps(comparison,indent=2))
    result=comparison['fit_diagnostics_retry']
    section='<article><h2>Separating sensor fit quality from prior agreement</h2><p>This retry added numerical diagnostics from the same RGB-D correspondences. It did not remove the existing acceptance gate or automatically replace the hinge. Luna retained control of model revisions.</p>'
    section+=f'<p>Native success: {result["native_successes"]}/{result["attempts"]}. Prior-only conflicts exposed: {len(conflicts)}.</p>'
    section+='<pre>'+html.escape(json.dumps(result.get('goal_claim'),indent=2))+'</pre>'
    section+='<details><summary>Controlled inputs, outcomes and cleanup</summary><pre>'+html.escape(json.dumps(comparison,indent=2))+'</pre></details>'
    section+='<p><a href="'+html.escape(os.path.relpath(baseline/'report/index.html',run/'report'))+'">Previous explicit-goal retry and its diagnosis</a></p></article>'
    page=run/'report/index.html';page.write_text(page.read_text().replace('<article><h2>Agent-authored findings</h2>',section+'<article><h2>Agent-authored findings</h2>'))
    summary=read(run/'report/summary.json');summary['fit_feedback_comparison']=comparison
    (run/'report/summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({key:comparison[key] for key in ['fit_diagnostics_retry','same_agent_prompt','same_example','existing_numeric_component_sources_unchanged','prior_conflicts_exposed_to_agent','storage']},indent=2))


if __name__=='__main__':main()
