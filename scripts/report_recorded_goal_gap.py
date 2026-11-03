"""Display model-chosen landmarks, measured gaps and separate evaluator evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    """Read a saved result without invoking a model or simulator."""
    return json.loads(path.read_text())


def resource_audit(study: Path) -> dict:
    """Preserve compact monitoring evidence from both GPU phases."""
    phases = []
    for name, folder in [('selection', study), ('tracking', study/'tracking-resources'),
                         ('all_recorded_rgb', study/'dense-tracking-resources')]:
        location = read(folder/'resource-monitor.json')
        rows = [json.loads(line) for line in Path(location['samples']).read_text().splitlines()]
        probes = {r['login_probe']['checked_at']:r['login_probe'] for r in rows if 'login_probe' in r}
        gpu = read(folder/'gpu-summary.json')
        phases.append({'phase':name,'samples':len(rows),
            'peak_job_memory_gib':max(r['job_memory_peak_gib'] for r in rows),
            'peak_anon_and_shmem_gib':max(r['job_anon_and_shmem_gib'] for r in rows),
            'minimum_available_host_memory_gib':min(r['host_available_memory_gib'] for r in rows),
            'maximum_io_full_avg10':max(r['io_full_avg10'] for r in rows),
            'oom_kills':max(r['job_memory_oom_kills'] for r in rows),
            'ssh_checks':len(probes),'ssh_successes':sum(p['healthy'] for p in probes.values()),
            'peak_gpu_memory_mib':gpu['peak_gpu_memory_mib'],
            'supervisor_status':read(folder/'supervisor-state.json')['status']})
        keys=['time','job_memory_gib','job_anon_and_shmem_gib','host_available_memory_gib',
              'io_full_avg10','memory_full_avg10','job_memory_oom_kills','login_probe']
        (study/(name+'-telemetry.jsonl')).write_text(''.join(
            json.dumps({k:r.get(k) for k in keys})+'\n' for r in rows))
    return {'phases':phases,'postprocessing_repaired_without_rerunning_tracker':True,
        'explanation':'The GPU tracker completed both cases. The first reader pass failed on depth files removed by prior post-episode retention. Corrected CPU postprocessing used the saved tracks and reports absent depth explicitly.',
        'memory_limit_gib':32,'swap_limit_bytes':0}


def condition_summary(study: Path, folder: Path, state: int, angles: dict, label: str) -> dict:
    """Keep the two tracking-density conditions separate and score only after measurement."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gap=read(folder/'gap.json')
    validation=[]
    for sample in gap['samples']:
        validation.append({'observation_id':sample['observation_id'],'control_step':sample['control_step'],
            'native_open_deg':angles[sample['control_step']],'gap_m':sample['gap_m']})
    (folder/'evaluator-validation.json').write_text(json.dumps({
        'native_labels_used_only_after_reference_freeze_and_gap_measurement':True,
        'rows':validation},indent=2)+'\n')
    supported=[r for r in gap['samples'] if r['supported']]
    causes={role:dict(Counter(r[role]['reason'] for r in gap['samples']))
            for role in ['moving','fixed_closed']}
    height_difference=None
    raw=gap.get('reference_samples',{})
    if raw.get('moving',{}).get('raw_world_m') and raw.get('fixed_closed',{}).get('raw_world_m'):
        height_difference=abs(raw['moving']['raw_world_m'][2]-raw['fixed_closed']['raw_world_m'][2])
    if gap['samples']:
        fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
        times=[r['control_step'] for r in gap['samples']]
        values=[r['gap_m']*1000 if r['gap_m'] is not None else np.nan for r in gap['samples']]
        axs[0].plot(times,values,'o-',color='#1462ad',label='Depth-supported tracked pair')
        unsupported=[r for r in gap['samples'] if not r['supported'] and r.get('raw_sample_distance_m') is not None]
        if unsupported:
            axs[0].scatter([r['control_step'] for r in unsupported],
                [r['raw_sample_distance_m']*1000 for r in unsupported],marker='x',color='#888888',
                label='Raw depth at unconfirmed track (not a supported gap)')
        axs[0].set(ylabel='Selected-point distance (mm)',title=f'State {state} · {label}: {len(supported)}/{len(times)} supported captures')
        axs[0].legend(loc='best');axs[0].grid(alpha=.2)
        axs[1].plot(list(angles),list(angles.values()),color='#a15617',label='Evaluator only; withheld from models')
        axs[1].set(ylabel='True door opening (degrees)',xlabel='Control step in the earlier recording (20 steps = 1 second)',xlim=(60,600))
        axs[1].grid(alpha=.2);axs[1].legend(loc='best')
        fig.savefig(folder/'gap.png',dpi=130);plt.close(fig)
    assessment=(f"Supported measurements: {len(supported)}/{len(gap['samples'])}. "
                f"Depth retained at {gap.get('recorded_depth_capture_count',0)} captures. "
                f"Moving landmark marked visible by CoTracker in {sum(r['moving']['tracker_visible'] for r in gap['samples'])} captures. ")
    if height_difference is not None:
        assessment+=f"Selected-reference height difference: {height_difference*1000:.1f} mm; these are not verified mating points."
    disagreements=[]
    for first,last in zip(supported,supported[1:]):
        angle_change=angles[last['control_step']]-angles[first['control_step']]
        gap_change=last['gap_m']-first['gap_m']
        if angle_change*gap_change<0:
            disagreements.append({'start_control':first['control_step'],'end_control':last['control_step'],
                'native_opening_change_deg':angle_change,'gap_change_mm':gap_change*1000,
                'interpretation':'The selected-pair distance changes opposite to the intended closing-progress cue.'})
    if disagreements:
        mismatch=disagreements[0]
        assessment+=(f" Controls {mismatch['start_control']}–{mismatch['end_control']}: native opening changed "
                     f"{mismatch['native_opening_change_deg']:+.2f}°, but gap changed {mismatch['gap_change_mm']:+.1f} mm.")
    initial=raw.get('moving',{}).get('raw_world_m')
    vertical_drift=max((abs(r['moving']['raw_world_m'][2]-initial[2]) for r in supported),default=0) if initial else None
    return {'label':label,'gap':gap,'plot':str(folder.relative_to(study)/'gap.png') if gap['samples'] else None,
            'gap_path':str(folder.relative_to(study)/'gap.json'),'assessment':assessment,
            'measurement_reasons':causes,'reference_height_difference_m':height_difference,
            'evaluator_validation':validation,'trend_disagreements':disagreements,
            'maximum_moving_point_height_drift_m':vertical_drift}


def render(study: Path) -> None:
    """Publish the frozen choices and both measured conditions without substituting results."""
    if not all((study/name).exists() for name in ['measurement-complete.json','dense-measurement-complete.json']):
        raise RuntimeError('Complete both numerical conditions before rendering their comparison')
    cases=[]
    for state in [17,49]:
        folder=study/f'state{state:03d}'
        run=ROOT/(f'results/luna-molmo2-integration-v3-20261103/trials/'
                  f'baseline-r1-close_microwave-state{state:03d}-600')
        native=read(run/'attempt01/initial/evaluation_contact_diagnostics.json')
        if native.get('excluded_from_policy_input') is not True:raise ValueError('Use evaluator-only export')
        angles={r['step']:abs(math.degrees(next(j['position'] for j in r['evaluation_only_articulation_state']
                                               if j['joint_name']=='microwave_1_microjoint'))) for r in native['steps']}
        conditions={'sampled64':condition_summary(study,folder,state,angles,'64 RGB frames'),
                    'all_recorded':condition_summary(study,study/'dense_tracking'/folder.name,state,angles,'all recorded RGB')}
        calls=[]
        for line in (folder/'tool-events.jsonl').read_text().splitlines():
            e=json.loads(line)
            if e['request']['operation']!='point':continue
            v=e['response'].get('result',{})
            calls.append({'event':e['sequence'],'camera':e['request']['arguments']['camera'],
                          'query':e['request']['arguments']['query'],'raw_text':v.get('raw_text'),
                          'points':v.get('points',[]),'images':v.get('images',[]),'seconds':e['wall_seconds']})
        cases.append({'state':state,'folder':folder.name,'references':read(folder/'references.json'),
            **conditions['all_recorded'],'conditions':conditions,'point_calls':calls,
            'selection_seconds':read(folder/'agent-exit.json')['wall_seconds'],
            'input_audit':read(folder/'input-audit.json'),
            'video':'../'+str((run/'attempt01/initial/rollout.mp4').relative_to(ROOT/'results'))})
    resources=resource_audit(study)
    peak=max(p['peak_job_memory_gib'] for p in resources['phases'])
    count=sum(p['ssh_checks'] for p in resources['phases']);passed=sum(p['ssh_successes'] for p in resources['phases'])
    gpu=max(p['peak_gpu_memory_mib'] for p in resources['phases'])/1024
    resource_text=f'Peak job RAM including file cache: {peak:.2f} GiB; peak GPU: {gpu:.2f} GiB. SSH checks: {passed}/{count}. No new robot episode. All model/tracker workers have exited.'
    data={'cases':cases,'protocol':read(study/'protocol.json'),'resources':resources,
          'dense_protocol':read(study/'dense-protocol.json'),
          'resource_summary':resource_text,
          'summary':'All-recorded-frame condition: '+' '.join(f'State {c["state"]}: {c["gap"].get("supported_capture_count",0)}/{c["gap"].get("capture_count",0)} depth-supported captures.' for c in cases)+' More frames improved coverage slightly, but a state-17 interval reports a smaller gap while the door opens. This is not reliable closing-progress feedback. Both density conditions and their unchanged references are shown.'}
    (study/'summary.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (study/'resource-audit.json').write_text(json.dumps(resources,indent=2)+'\n')
    template=(ROOT/'templates/recorded_goal_gap.html').read_text()
    (study/'watch.html').write_text(template.replace('DATA_PLACEHOLDER',json.dumps(data).replace('<','\\u003c')))
    shutil.copyfile(ROOT/'templates/luna_documents.js',study/'document-viewer.js')
    print([(r['state'],r['gap'].get('supported_capture_count')) for r in cases])


def main() -> None:
    """Render the requested recorded-data experiment."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',required=True,type=Path)
    args=parser.parse_args()
    render(args.study.resolve())


if __name__=='__main__':main()
