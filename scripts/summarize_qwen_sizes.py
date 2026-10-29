"""Summarize the size comparison and apply its registered manipulation gate."""
import argparse
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/qwen-sampling-diagnostic-20261025'

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def selected_error(case):
    chosen=(case.get('actor_selection') or {}).get('chosen_ground_call',0)
    if not chosen:return None
    matches=[r for r in case.get('observer_geometry',[]) if r['id'].endswith('-ground'+str(chosen))]
    return matches[0].get('native_hinge_line_error_mm') if len(matches)==1 else None

def summarize(root):
    rows=read(root/'study/results.json',[]);geo=read(root/'geometry27b-private.json',[]);old=read(BASE/'geometry-private.json',[])
    matched=read(root/'matched27b/summary.json',{}).get('cases',[]);fresh=read(root/'fresh-summary.json',{}).get('cases',[])
    new_errors=[g['native_hinge_line_error_mm'] for g in geo if g['geometry_accepted']]
    old_errors=[g['native_hinge_line_error_mm'] for g in old if g['geometry_accepted']]
    median=lambda x:statistics.median(x) if x else None
    recovered=lambda gs:sum(bool((g.get('fit_diagnostics') or {}).get('handle_recovery')) for g in gs)
    old_median,new_median=median(old_errors),median(new_errors)
    matched_pairs=[{'id':r['id'],'new_selected_hinge_error_mm':selected_error(r),
                    'baseline_selected_hinge_error_mm':selected_error(r['baseline'])} for r in matched]
    fresh_summary=[{'id':r['id'],'model':r['model'],'case':r['case'],'selected_hinge_error_mm':selected_error(r),
                    'termination_reason':r['termination_reason'],'model_calls':r['model_calls'],
                    'model_seconds':r['model_seconds'],'tool_seconds':r['tool_seconds'],
                    'geometry_seconds':r['geometry_seconds']} for r in fresh]
    gates={'at_least_four_one_step_fits':len(new_errors)>=4,
        'median_hinge_error_improves_at_least_30_percent':new_median is not None and new_median<=old_median*.7,
        'matched_tool_selected_error_not_worse_each':len(matched_pairs)==2 and all(r['new_selected_hinge_error_mm'] is not None and r['new_selected_hinge_error_mm']<=r['baseline_selected_hinge_error_mm'] for r in matched_pairs),
        'both_fresh27b_selected_errors_under_30mm':len([r for r in fresh_summary if r['model']=='qwen27b'])==2 and all(r['selected_hinge_error_mm'] is not None and r['selected_hinge_error_mm']<30 for r in fresh_summary if r['model']=='qwen27b'),
        'no_more_one_step_handle_recoveries_than9b':recovered(geo)<=recovered(old)}
    complete=len(rows)==6 and len(geo)==6 and len(matched)==2 and len(fresh)==4
    failed_early=len(geo)==6 and not all(gates[k] for k in ['at_least_four_one_step_fits',
        'median_hinge_error_improves_at_least_30_percent','no_more_one_step_handle_recoveries_than9b'])
    by_old={r['id']:r for r in old};by_new={r['id']:r for r in geo}
    paired=[{'id':key,'hinge_error_mm9b':by_old[key]['native_hinge_line_error_mm'],
              'hinge_error_mm27b':by_new[key]['native_hinge_line_error_mm']}
            for key in by_old.keys()&by_new.keys() if by_old[key]['geometry_accepted'] and by_new[key]['geometry_accepted']]
    raw9=read(root/'raw-hinge9b-private.json',[]);raw27=read(root/'raw-hinge27b-private.json',[])
    model_totals={};tool_totals={};counts=[];case_summaries=[]
    for phase in ['study','matched27b','fresh-qwen9b','fresh-qwen27b']:
        for path in sorted((root/phase).glob('**/call*/timing.json')):
            timing=read(path,{});request=read(path.with_name('request.json'),{});model=request.get('model','unknown');usage=timing.get('usage') or {};context=timing.get('context_budget') or {}
            total=model_totals.setdefault(model,{'requests':0,'seconds':0.,'output_tokens':0,'input_tokens':0,'context_admission_seconds':0.,'length_finishes':0,'errors':0})
            total['requests']+=1;total['seconds']+=timing.get('wall_seconds',0);total['output_tokens']+=usage.get('completion_tokens',0);total['input_tokens']+=usage.get('prompt_tokens',0);total['context_admission_seconds']+=context.get('tokenization_seconds',0);total['length_finishes']+=timing.get('finish_reason')=='length';total['errors']+=bool(timing.get('error'))
            counts.append({'file':str(path.relative_to(root)),'measured_prompt_tokens':context.get('prompt_tokens'),
                'actual_prompt_tokens':usage.get('prompt_tokens'),'matches':context.get('prompt_tokens')==usage.get('prompt_tokens') if usage.get('prompt_tokens') is not None else None,
                'requested_output_tokens':request.get('max_tokens')})
        for path in sorted((root/phase).glob('*/tool-events.jsonl')):
            for line in path.read_text().splitlines():
                e=json.loads(line);name=e['call']['function']['name'];total=tool_totals.setdefault(name,{'calls':0,'seconds':0.,'errors':0});total['calls']+=1;total['seconds']+=e['seconds'];total['errors']+=e['result'].get('ok') is False
        if phase!='study':
            for path in sorted((root/phase).glob('*/summary.json')):
                case=read(path,{});case_summaries.append({'phase':phase,'id':case.get('id'),
                    **{k:case.get(k) for k in ['actor_wall_seconds','model_seconds','tool_seconds','geometry_seconds','private_evaluation_seconds','ground_calls','accepted_fits','termination_reason','error']}})
    gpu={}
    for name in ['server27b','server9b']:
        exit_record=read(root/name/'server-exit.json')
        if exit_record:gpu[name]=exit_record
        elif (root/name/'gpu-memory.jsonl').exists():
            records=[json.loads(s) for s in (root/name/'gpu-memory.jsonl').read_text().splitlines() if s.strip()];gpu[name]={'peak_owned_gpu_bytes_so_far':max(r['server_gpu_bytes'] for r in records)}
        if name in gpu:
            measured=gpu[name].get('peak_server_gpu_bytes',gpu[name].get('peak_owned_gpu_bytes_so_far'))
            gpu[name]['peak_owned_gpu_gib']=measured/2**30
            gpu[name]['interpretation']='Configured serving allocation including cache and runtime, not minimum model memory.'
    result={'complete_offline_comparison':complete,'one_step':{'valid27b':sum(r['contract_valid'] for r in rows),'cases':len(rows),
            'geometry_evaluated_cases27b':len(geo),'fits9b':len(old_errors),'fits27b':len(new_errors) if geo else None,'median_hinge_error_mm9b':old_median,'median_hinge_error_mm27b':new_median,
            'hinge_errors_mm27b':new_errors,'handle_recoveries9b':recovered(old),'handle_recoveries27b':recovered(geo),
            'paired_admitted_fits':sorted(paired,key=lambda r:r['id']),
            'raw_hinge_line_median_error_px9b':median([r['mean_distance_to_projected_native_hinge_line_px'] for r in raw9]),
            'raw_hinge_line_median_error_px27b':median([r['mean_distance_to_projected_native_hinge_line_px'] for r in raw27]),
            'accepted_median_caveat':'Admitted-fit subsets differ; use the paired records and raw image-line measurements as well.',
            'recovery_count_caveat':'Counts among admitted fits; denominators differ if fit acceptance differs. The registered manipulation gate conservatively uses counts.'},
        'matched_tool_comparison':matched_pairs,'fresh_tool_comparison':fresh_summary,'model_totals':model_totals,
        'tool_totals':tool_totals,'case_time_breakdown':case_summaries,'gpu_memory':gpu,
        'offline_timing_scope':{'new_model_requests_only':True,'grasp_planning_seconds':0,'ik_seconds':0,
            'arc_planning_seconds':0,'physical_execution_seconds':0,
            'one_step_observer_geometry_seconds':sum(g.get('geometry_seconds',0) for g in geo),
            'tool_grounding_seconds':sum(c.get('geometry_seconds') or 0 for c in case_summaries),
            'tool_case_private_scoring_seconds':sum(c.get('private_evaluation_seconds') or 0 for c in case_summaries)},
        'physical_gate':{'decision':'skip' if failed_early else 'pass' if complete and all(gates.values()) else 'skip' if complete else 'pending','checks':gates,
            'interpretation':'A screening gate for one development simulator episode, not evidence of task success or a general grounding tolerance.'},
        'all_reported_context_counts_match':all(r['matches'] is True for r in counts),
        'all_requests_allow_32k':all((r['requested_output_tokens'] or 0)>=32768 for r in counts),
        'scope':'Same-asset development comparison. Native answers remain observer-only. No memory promotion or matched Luna comparison. Latency is descriptive on shared hardware.'}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(root/'context-count-audit.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps({'one_step':result['one_step'],'models':model_totals,'physical_gate':result['physical_gate']['decision']}))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);summarize(p.parse_args().run.resolve())
