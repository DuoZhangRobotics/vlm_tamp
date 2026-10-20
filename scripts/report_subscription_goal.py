"""Compare the explicit-goal pilot with the preceding development run."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(run):
    events=[json.loads(x) for x in (run/'tool-events.jsonl').read_text().splitlines()]
    episodes=read(run/'evaluation-private.json',{'attempts':[]})['attempts']
    return {'run':run.name,'native_successes':sum(e['official_task_success'] for e in episodes),
        'attempts':len(episodes),'control_steps':[e['control_steps'] for e in episodes],
        'goal_claim':read(run/'agent-workspace/completion.json'),
        'agent_wall_seconds':read(run/'agent-exit.json')['wall_seconds'],
        'tool_calls':len(events),'tool_wall_seconds':sum(r['wall_seconds'] for r in events),
        'failed_tools':[{'event':r['sequence'],'operation':r['request']['operation'],
            'error':r['response'].get('error')} for r in events if not r['response']['ok']],
        'execution_audit':read(run/'audit.json')['trajectory_audit']}


def prune_unused_depth(run):
    """Preserve raw depth if the agent ran custom array analysis; otherwise
    keep endpoints plus all RGB, measured landmarks and planner geometry."""
    commands=[]
    for line in (run/'agent-events.jsonl').read_text().splitlines():
        try:item=json.loads(line).get('item',{})
        except json.JSONDecodeError:continue
        if item.get('type')=='command_execution':commands.append(item.get('command',''))
    analysis=any(any(token in c for token in ['np.load','numpy.load','sensors.npz','glob(','rglob(']) for c in commands)
    removed=[]
    if not analysis:
        for attempt in (run/'agent-workspace/robot_data').glob('attempt*'):
            arrays=sorted(attempt.glob('obs*/sensors.npz'))
            for path in arrays[1:-1]:
                removed.append({'path':str(path.relative_to(run)),'bytes':path.stat().st_size,'sha256':digest(path)})
                path.unlink()
    private_reclaimed=sum(ep.get('storage',{}).get('bytes_reclaimed_this_pass',0)
        for ep in read(run/'evaluation-private.json',{'attempts':[]})['attempts'])
    record={'agent_array_analysis_detected':analysis,'pruned_sensor_arrays':removed,
        'public_bytes_reclaimed':sum(x['bytes'] for x in removed),
        'private_bytes_reclaimed':private_reclaimed,
        'retained_bytes':sum(p.stat().st_size for p in run.rglob('*') if p.is_file() and not p.is_symlink()),
        'scratch_removed':not Path(read(run/'protocol.json')['workspace']).exists()}
    (run/'storage-audit.json').write_text(json.dumps(record,indent=2))
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--baseline',default='results/subscription-agent-pilot-v2-20261018')
    args=parser.parse_args();run=(ROOT/args.run).resolve();baseline=(ROOT/args.baseline).resolve()
    if not (run/'audit.json').exists():raise ValueError('Run must be complete')
    oldhash=read(baseline/'source-hashes.json');newhash=read(run/'source-hashes.json')
    numeric=[k for k in oldhash if k.startswith('src/vlm_tamp/') or k=='scripts/serve_subscription_robot.py']
    unchanged=all(oldhash[k]==newhash.get(k) for k in numeric)
    inputs={}
    for name in ['agentview','frontview','wrist']:
        p=baseline/f'attempt01/opened/{name}.png';q=run/f'attempt01/opened/{name}.png'
        if p.exists() and q.exists():inputs[name]=digest(p)==digest(q)
    shared=['tools.txt','robot.py','example/readme.txt','example/matches.npz',
            'example/sensor-fit.json','example/before.png','example/after.png']
    shared_match={name:digest(baseline/'public-inputs'/name)==digest(run/'public-inputs'/name) for name in shared}
    old_prompt=(baseline/'agent-prompt.txt').read_text();new_prompt=(run/'agent-prompt.txt').read_text()
    comparison={'baseline':summarize(baseline),'explicit_goal':summarize(run),
        'numerical_tools_unchanged':unchanged,'initial_rgb_identical':inputs,
        'example_and_tool_documentation_identical':shared_match,
        'original_prompt_preserved_as_prefix':new_prompt.startswith(old_prompt),
        'added_instruction':new_prompt[len(old_prompt):] if new_prompt.startswith(old_prompt) else None,
        'different_task_strategy_or_coordinates_supplied':False,
        'held_out_generalization_evaluated':False,
        'interpretation':'Development follow-up on the same case. Model and proposal sampling can vary; this single comparison does not establish a causal success-rate improvement.'}
    comparison['storage']=prune_unused_depth(run)
    (run/'goal-comparison.json').write_text(json.dumps(comparison,indent=2))
    completion=read(run/'agent-workspace/completion.json',{})
    rows=[]
    for label,record in [('Previous pilot',comparison['baseline']),('Explicit-goal pilot',comparison['explicit_goal'])]:
        rows.append(f'<tr><td>{label}</td><td>{record["native_successes"]}/{record["attempts"]}</td>'
            f'<td>{record["control_steps"]}</td><td>{record["agent_wall_seconds"]:.1f}s</td></tr>')
    section='<article><h2>Explicit completion objective</h2><p>The only change to the agent input was an explicit 90-degree completion objective and a request to report completion evidence. The numerical tools, approved example and task case were retained.</p>'
    section+='<table><tr><th>Run</th><th>Native successes / attempts</th><th>Control steps</th><th>Agent time</th></tr>'+''.join(rows)+'</table>'
    section+='<h3>Agent completion assessment</h3><pre>'+html.escape(json.dumps(completion,indent=2))+'</pre>'
    section+='<details><summary>Comparison checks, prompt change, and cleanup</summary><pre>'+html.escape(json.dumps(comparison,indent=2))+'</pre></details>'
    section+='<p><a href="'+html.escape(os.path.relpath(baseline/'report/index.html',run/'report'))+'">Previous pilot and earlier interrupted attempts</a></p></article>'
    page=run/'report/index.html';body=page.read_text()
    body=body.replace('<article><h2>Agent-authored findings</h2>',section+'<article><h2>Agent-authored findings</h2>')
    page.write_text(body)
    summary=read(run/'report/summary.json');summary['goal_comparison']=comparison
    (run/'report/summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:comparison[k] for k in ['explicit_goal','numerical_tools_unchanged','initial_rgb_identical','storage']},indent=2))


if __name__=='__main__':main()
