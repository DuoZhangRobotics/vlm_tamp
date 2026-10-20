"""Audit native outcomes, paired inputs, frozen sources, spending and report links."""

import hashlib
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit,unquote

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in {'href','src'} and v)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',default='results/articulated-libero-report-v2')
    args=parser.parse_args()
    report=ROOT/args.report
    roots={'strict':ROOT/'results/articulated-strict-final','diagnostic':ROOT/'results/articulated-physics-pilot'}
    protocols={mode:read(base/'protocol.json') for mode,base in roots.items()}
    sources=protocols['strict']['sources_sha256']
    changed_current=[name for name,h in sources.items() if sha(ROOT/name)!=h]
    bad_snapshots=[]
    episodes=[]
    required={'robot0_link1_collision','robot0_link2_collision','robot0_link3_collision',
        'robot0_link4_collision','robot0_link5_collision','robot0_link6_collision','gripper0_hand_collision',
        'gripper0_finger1_collision','gripper0_finger2_collision','gripper0_finger1_pad_collision','gripper0_finger2_pad_collision'}
    paired_checks=[]
    for mode,base in roots.items():
        for name,h in sources.items():
            if sha(base/'source-snapshot'/name)!=h:bad_snapshots.append(f'{mode}/{name}')
        for item in read(base/'manifest.json')['planned_episodes']:
            folder=base/f'{item["task"]}-state{item["state"]:03d}'
            summary=read(folder/'summary.json')
            actions=read(folder/'initial/actions.json')
            native=any(row['evaluation_task_success'] for row in actions)
            if native!=summary['official_task_success'] or bool(actions[-1]['evaluation_task_success'])!=summary['final_task_success']:
                raise ValueError('Native task flags disagree with action logs')
            if len(actions)!=summary['control_steps'] or len(actions)>600:
                raise ValueError('Episode count/control budget mismatch')
            metadata=read(folder/'opened/metadata.json')
            names={v['robot_geom_name'] for v in metadata['robot_collision_volumes_world'] if not v.get('fixed')}
            if names!=required:raise ValueError('Known robot geometry mapping does not cover all expected moving parts')
            if metadata['no_simulator_inventory'] is not True:raise ValueError('Simulator inventory must be excluded')
            for camera in metadata['cameras'].values():
                prior=camera['robot_depth_prior']
                if prior['scene_geometry_used'] or not all(name.startswith(('robot0_','gripper0_')) for name in prior['rendered_robot_geometry_names']):
                    raise ValueError('Self-depth prior contains non-robot geometry')
            if summary['assisted_grasping'] or not summary['physical_contacts_enabled']:
                raise ValueError('Physical grasp protocol changed')
            frame=read(folder/'curobo-frame-validation.json')
            if not frame['accepted']:raise ValueError('Robot/TCP frame parity failed')
            evidence=read(folder/'grasp-cache-evidence.json')
            if not evidence['accepted']:raise ValueError('Paired grasp geometry cache mismatch')
            paired_checks.append({'mode':mode,'task':item['task'],
                'candidate_sha256':sha(folder/'grasps/grasp_candidates.npz')})
            episodes.append({'mode':mode,'task':item['task'],'native_success':native,
                             'control_steps':len(actions),'failure_stage':summary['failure_stage']})
    by_task={}
    for row in paired_checks:by_task.setdefault(row['task'],[]).append(row['candidate_sha256'])
    identical_candidates=all(len(hashes)==2 and len(set(hashes))==1 for hashes in by_task.values())
    before=read(ROOT/'results/contact-policy-v8-development/preserved-before.json')
    changed_prior=[name for name,h in before.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=h]
    old_sources=read(ROOT/'results/physical-tasks-curobo-eval/protocol.json')['sources_sha256']
    changed_legacy=[name for name,h in old_sources.items() if sha(ROOT/name)!=h]
    ledger=read(ROOT/'results/articulated-api-v1/ledger.json')
    provenance=read(ROOT/'results/articulated-api-v1/prior-spend.json')
    if sha(ROOT/provenance['prior_ledger'])!=provenance['prior_ledger_sha256']:
        raise ValueError('Previous spending ledger changed')
    request_policy=[]
    for entry in ledger['requests'].values():
        args=read(ROOT/'results/articulated-api-v1'/f'request-{entry["request_sha256"]}.json')
        request_policy.append(args['model']=='gpt-6-luna' and args['reasoning']=={'effort':'high'}
            and args['store'] is False and args['max_output_tokens']==6000
            and {c['type'] for c in args['input'][0]['content']}=={'input_text','input_image'})
    suites=ET.parse(ROOT/'results/articulated-development/regression-tests-final.xml').getroot().findall('testsuite')
    tests={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
    parser=Links();parser.feed((report/'index.html').read_text())
    local=[unquote(urlsplit(link).path) for link in parser.links if not urlsplit(link).scheme and not link.startswith('#')]
    missing=[link for link in local if not (report/link).is_file()]
    checks={'all_ten_planned_episodes_present':len(episodes)==10,
        'native_success_recomputed':True,'all_control_budgets_passed':True,
        'paired_frozen_sources_identical':protocols['strict']['sources_sha256']==protocols['diagnostic']['sources_sha256'],
        'current_sources_match_frozen':not changed_current,'source_snapshots_match':not bad_snapshots,
        'all_prior_files_preserved':not changed_prior,'legacy_curobo_sources_preserved':not changed_legacy,
        'paired_grasp_candidates_identical':identical_candidates,'robot_geometry_name_mapping_verified':True,
        'no_simulator_inventory_or_scene_geometry_in_robot_prior':True,'all_gpu_tcp_parity_checks_passed':True,
        'all_api_requests_luna_high_rgb_only':all(request_policy),'api_cap_respected':ledger['cumulative_conservative_spend_usd']<=2.,
        'report_links_valid':not missing,'regression_tests_passed':tests['tests']>0 and tests['failures']==tests['errors']==0}
    result={'checks':checks,'all_checks_passed':all(checks.values()),'episodes':episodes,
        'source_files_frozen':len(sources),'previous_files_preserved':len(before),'legacy_source_files_preserved':len(old_sources),
        'changed_current_sources':changed_current,'changed_prior_files':changed_prior,'missing_links':missing,
        'local_report_links_checked':len(local),'api_attempts_including_prototypes':ledger['api_attempts'],
        'new_api_spend_usd':ledger['cumulative_conservative_spend_usd']-provenance['inherited_cumulative_spend_usd'],
        'cumulative_api_spend_usd':ledger['cumulative_conservative_spend_usd'],'remaining_api_budget_usd':ledger['remaining_usd'],
        'regression_tests':tests,'validator_source_sha256':sha(Path(__file__))}
    (report/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'}))
    if not result['all_checks_passed']:raise ValueError('Pilot validation failed')


if __name__=='__main__':
    main()
