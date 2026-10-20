"""Same-case paired retest: shared completion evidence, abstract episodic memory."""
import argparse,ast,copy,hashlib,html,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_subscription_memory_challenge import run,diagnostic_counts,CONDITIONS,DONE
from run_subscription_dino_study import read,write,sha
from vlm_tamp.episodic_memory import store_hashes
from abstract_subscription_memory import validate_abstract

GROUPS={'known_challenge':'Previously difficult scenes','fresh_transfer':'Repeated transfer scenes'}
PROMPT='''Perform the instruction returned by reset through physical interaction using
tools.txt. Carry out the instruction to its full intended extent. Aim for the
intended end state; partial progress alone does not count as completion.
Read the documented tools, the approved method-only example, and the
fixed guide in experience/guidelines.json. There is ONE physical attempt, seed
and scene supplied by the experiment, and 600 recorded control steps including
reset's initial hold. Call reset once, inspect current images, and call
memory.recall with the returned instruction and robot Panda before movement.
The library may be empty or contain coordinate-free prior experience. Inspect
its applicability and evidence. No historical geometry or paths are available.

Before moving, use declare_completion to state an observable quantitative
terminal condition implied by the instruction and current scene. State your
assumptions; do not equate slight progress with the full requested result. You
choose the target range and rationale; the service supplies no task-specific
target or hidden success threshold. Establish fresh reference measurements.
Use measure_surface and compare_surfaces to measure current material patches.
The comparison requires consistent image/depth correspondences and can measure
full rigid rotation or material displacement. A plane normal alone cannot observe
rotation within its plane. Choose an appropriate measured quantity. Inspect
overlays and association support to ensure they track the intended material, and
check the final observations quantitatively with check_completion. Explain
visibility, fit support and ambiguity; use another selection or view when
needed. Optional DINO or your own analysis can supply current matched pixels.
If long motions lose overlap, comparisons between intermediate observations can
establish a chain of supported material matches. These tools do not recognize
objects or certify the native goal. No old coordinates or numeric goals are supplied.

Use fresh plans via robot.py. You may write analysis code, use optional DINO,
investigate planning failures and adapt between complete trajectories. Never
reset for another attempt. Every issued trajectory executes fully. A failed
planning call does not automatically terminate the episode. If observations
contradict your completion claim, investigate or continue while appropriate;
do not lower the goal or count planned angles/TCP motion as object progress.
Call finish with an explicit completed/incomplete/uncertain claim and reason.
Completed requires a fresh supported measured check. Incomplete and uncertain
remain legitimate outcomes if you cannot establish the intended result.

Learn reusable methods for similar interactions from the observations and tool
outcomes within the existing attempt and control budgets. On later decisions,
reuse a supported lesson when its preconditions hold; revise it when new evidence
contradicts it. Extract conditional principles rather than a scene-specific replay.

Write findings.txt as the factual episode record: intended result, observations,
failed attempts, competing explanations, changes tried, observed consequences,
completion evidence and unresolved questions. Separate observation, hypothesis
and conclusion. A correction followed by progress alone does not establish its
cause. Cite accessible current-session request/reply files or saved observations;
identify inherited evidence separately. Preserve failures and counterexamples.

Write guidelines.json as candidate procedural memory with a principles list.
For each useful lesson include:
- principle: a concise conditional rule or methodology derived from the evidence;
- applicability: observable preconditions, relevant mechanism/contact properties,
  and conditions where the lesson may not apply;
- procedure: choices of tools or actions, what observations inform each choice,
  and what expected outcome would support or contradict the method;
- failure_modes: observed symptoms, possible causes, evidence that distinguishes
  them, and candidate responses labeled as tried or untested;
- evidence: references, support actually observed, counterexamples, limitations,
  and status such as hypothesis, supported in this episode or transfer-tested;
- reobserve: quantities and relationships that must be measured in each new scene;
- transfer_test: a proposed change of scene or related task that would test the
  claimed scope, including what outcome would falsify the proposed lesson;
- attribution: inherited guidance, documented tool behavior, or your own finding.
Keep this concise; an empty principles list is acceptable when nothing is
supported. Record failures even without a successful correction. One successful
attempt supports a candidate method, not universal reliability. Express reusable
methods through current observations and relative relationships, without storing
historical coordinates, pixel regions, poses, trajectories or fixed action values
as instructions for future scenes. Detailed measurements belong in findings.

Write memory-use.json with retrieved ids, which prior principles applied, which
were rejected or revised and why, and candidate additions. These outputs propose
future memory updates; do not modify the supplied read-only library or common
guide. Learning from this run and testing transfer on later frozen runs are
separate stages. Do not claim a proposed transfer test has already been run.
Use only this Luna/high ChatGPT subscription session. No external model APIs,
implementation source, evaluator state, other project results or agent history.
Do not ask the human to debug or provide strategies. Distinguish inherited
tools and examples from your own decisions.\n'''

def numerical_policy_audit(prior):
    old=read(prior/'frozen-source-hashes.json')
    allowed={'scripts/serve_subscription_robot.py','scripts/prepare_subscription_pilot.py',
        'scripts/run_subscription_agent_pilot.py','scripts/run_subscription_memory_challenge.py',
        'scripts/run_subscription_remembered_agent.py','scripts/run_subscription_dino_study.py','configs/subscription_memory.json'}
    drift=[p for p,h in old.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    unexpected=sorted(set(drift)-allowed)
    if unexpected:raise RuntimeError('Unexpected old numerical source drift: '+str(unexpected))
    reference=prior/'runs/known-nominal-047-without_memory/source-snapshot/scripts/serve_subscription_robot.py'
    def methods(path):
        tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RobotService')
        return {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    before=methods(reference);after=methods(ROOT/'scripts/serve_subscription_robot.py')
    protected=['ground','rebuild','export_geometry','start_tracking','measure_tracking','apply_tracking',
               '_apply_tracking_pose','set_model','keep_plan','plan_grasp','plan_arc','plan_joint',
               'ik_solutions','execute_plan','gripper','issue','finish_attempt']
    checks={name:ast.dump(before[name])==ast.dump(after[name]) for name in protected}
    if not all(checks.values()):raise RuntimeError('Numerical or physical methods changed: '+str(checks))
    def config_updates(method):
        return [ast.dump(n) for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                and isinstance(n.func.value,ast.Attribute) and n.func.value.attr=='config' and n.func.attr=='update']
    config_same=config_updates(before['__init__'])==config_updates(after['__init__'])
    if not config_same:raise RuntimeError('Numerical service configuration changed')
    return {'old_files_changed':drift,'unexpected_changes':unexpected,'protected_methods_equal':checks,
            'numerical_config_equal':config_same,'scope':'Only agent memory/context, completion measurement and reporting interfaces change.'}

def freeze(study):
    if (study/'protocol.json').exists():raise ValueError('Already frozen')
    prior=ROOT/'results/subscription-memory-challenge-20261019'
    if not read(prior/'completion.json')['completed']:raise ValueError('Prior comparison incomplete')
    profile=read(ROOT/'configs/subscription_memory.json');library=ROOT/profile['public_library']
    if profile['version']!=2 or profile.get('completion_protocol_version')!=2:raise ValueError('Register the reviewed abstract memory and revised completion protocol first')
    hashes=store_hashes(library)
    if hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()!=profile['library_manifest_sha256']:raise ValueError('Memory changed')
    for key in ['prompt','common_guide']:
        if sha(ROOT/profile[key])!=profile[key+'_sha256']:raise ValueError('Registered input changed')
    if (ROOT/profile['prompt']).read_text()!=PROMPT:raise ValueError('Wrong completion prompt')
    validate_abstract(read(library/'episode-0001/procedure.json'));validate_abstract(read(ROOT/profile['common_guide']))
    expected={'episode-0001/entry.json','episode-0001/procedure.json','episode-0001/evidence/events.json'}
    if set(hashes)!=expected:raise ValueError('Unexpected accessible memory payload')
    audit=numerical_policy_audit(prior);write(study/'numerical-policy-audit.json',audit)
    manifest=read(prior/'manifest.json');manifest['repeated_previously_evaluated_scenarios']=True
    write(study/'manifest.json',manifest);(study/'empty-library').mkdir()
    (study/'prompt.txt').write_text(PROMPT);write(study/'guidelines.json',read(ROOT/profile['common_guide']))
    # Preserve the actual registered bytes, not a JSON reserialization's whitespace.
    (study/'guidelines.json').write_bytes((ROOT/profile['common_guide']).read_bytes())
    protocol=copy.deepcopy(read(prior/'protocol.json'))
    protocol.update(version=2,previous_study=str(prior),memory_store=str(library),library_hashes=hashes,
        prompt_sha256=sha(study/'prompt.txt'),guidelines_sha256=sha(study/'guidelines.json'),
        same_prompt_guide_example_tools_as_previous_study=False,
        completion_checks=True,completion_protocol_version=2,abstract_experience=True,coordinates_available_from_history=False,
        previous_evaluated_source_drift=audit['old_files_changed'],new_task_specific_rules=False,
        numerical_planning_grasp_ik_contact_control_unchanged=True,
        completion_target_source='Acting agent, chosen before action; no native target supplied',
        completion_geometry='Generic RGB-D patch fitting, mutual SIFT or supplied point matches, robust rigid transforms, and supported material-association chains. Semantic identity and goals remain agent judged.',
        diagnostic_predecessor=str(ROOT/'results/subscription-completion-memory-20261019'),
        diagnostic_predecessor_status='Stopped after two preserved diagnostic episodes exposed missing material-association validation; no diagnostic outcomes enter memory.',
        diagnostic_predecessors=[str(ROOT/'results/subscription-completion-memory-20261019'),str(ROOT/'results/subscription-completion-memory-v2-20261019')],
        diagnostic_serialization_fix='Convert matched pixel coordinates to JSON-native integers; exact failed request replay and full response regression pass. Prior affected outcomes are retained separately.',
        human_example_representation='Same fitting method as prose; historical arrays/images/numbers omitted',
        memory_updated_from_evaluation=False,groups=GROUPS,
        interpretation='Same cases as the preceding comparison, now reused for retesting. Both arms get new shared completion checks and abstract common context. Memory arm additionally retrieves the abstract old episode. Old/new changes are bundled; they do not isolate individual improvements. One session per condition per scene and one asset.')
    files=list((ROOT/'src/vlm_tamp').rglob('*.py'))+list((ROOT/'src/vlm_tamp/native').glob('*.cc'))+list((ROOT/'scripts').glob('*.py'))
    files +=[ROOT/'configs/articulated_libero.json',ROOT/'configs/subscription_memory.json',ROOT/'requirements-rekep-libero.txt',ROOT/'requirements-lock.txt']
    write(study/'frozen-source-hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
    write(study/'protocol.json',protocol);write(study/'previous-summary.json',read(prior/'summary.json'))

def diagnose(folder):
    row=diagnostic_counts(folder);p=folder/'tool-events.jsonl'
    events=[json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
    row['planning_failures']=sum(e['request']['operation'].startswith('plan_') and
        (not e['response']['ok'] or e['response'].get('result',{}).get('planned') is False) for e in events)
    row['completion_checks']=sum(e['request']['operation']=='check_completion' for e in events)
    row['unsupported_finish_claims']=sum(e['request']['operation']=='finish' and e['response'].get('result',{}).get('finished') is False for e in events)
    return row

def report(study,manifest,records):
    scores={};table=[];rows=[];videos=[]
    old=read(study/'previous-summary.json');pairs={};cross={}
    for group,label in GROUPS.items():
        scores[group]={}
        for condition in CONDITIONS:
            cases=[c for c in manifest['cases'] if c['group']==group and c['condition']==condition]
            done=[records[c['case_id']] for c in cases if records.get(c['case_id'],{}).get('status') in DONE]
            score={'planned':len(cases),'completed':len(done),'native_goal_reached':sum(r['first_attempt_success'] for r in done),
                   'native_goal_at_end':sum(r['final_task_success'] for r in done)}
            scores[group][condition]=score
            table.append(f'<tr><td>{label}</td><td>{"Abstract memory" if condition=="with_memory" else "Guide only"}</td><td>{old["scores"][group][condition]["native_goal_reached"]}/3</td><td>{score["native_goal_reached"]}/3</td><td>{score["native_goal_at_end"]}/3</td><td>{len(done)}/3</td></tr>')
    for scenario in manifest['scenarios']:
        a=records.get(scenario['scenario_id']+'-without_memory',{}).get('initial_arrays',{})
        b=records.get(scenario['scenario_id']+'-with_memory',{}).get('initial_arrays',{})
        if a and b:pairs[scenario['scenario_id']]={'same_keys':a.keys()==b.keys(),'equal_arrays':a==b}
    for c in manifest['cases']:
        cid=c['case_id'];r=records.get(cid,{});folder=study/'runs'/cid
        initial=r.get('initial_arrays')
        if initial:cross[cid]=initial==old['records'][cid]['initial_arrays']
        eps=r.get('attempts',[]);claim=read(folder/'agent-workspace/robot_data/attempt01/completion/finish-claim.json',{})
        link=f'<a href="runs/{cid}/report/index.html">{cid}</a>' if (folder/'report/index.html').exists() else cid
        rows.append(f'<tr><td>{link}</td><td>{r.get("status","planned")}</td><td>{r.get("first_attempt_success","—")}</td><td>{eps[0]["control_steps"] if eps else "—"}</td><td>{html.escape(claim.get("claim","—"))}</td><td>{r.get("diagnostics",{}).get("planning_failures","—")}</td></tr>')
        if eps:videos.append(f'<details><summary>{link}</summary><video controls preload="none" src="runs/{cid}/attempt01/initial/rollout.mp4"></video></details>')
    summary={'scores':scores,'records':records,'initial_sensor_pairs':pairs,'initial_arrays_equal_to_prior_study':cross,
        'completed_cases':sum(r.get('status') in DONE for r in records.values()),'planned_cases':12,
        'all_planned_cases_in_denominator':True,'separately_billed_api_calls':sum(r.get('api_calls',0) for r in records.values())}
    write(study/'summary.json',summary)
    (study/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Completion checks and abstract memory</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:35px auto;padding:20px;background:#f4f7fa;color:#193146}}article{{background:white;border:1px solid #ccd5df;padding:20px;border-radius:12px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #ccd5df}}video{{width:100%;max-height:550px}}details{{margin:15px 0}}pre{{white-space:pre-wrap}}</style>
<h1>Completion checks and coordinate-free memory</h1><p>Completed {summary['completed_cases']}/12 planned sessions. Pending cases have no outcome yet.</p>
<article><p>Same six scenes, order, seed, single-attempt budget, Luna/high subscription, optional DINO, grasping, IK, collision planning and controller. Both groups now declare their own observable goal and use current RGB-D measurements to support completion. Native success and native target thresholds stay hidden. The memory arm receives the old learned procedure without historical coordinates, trajectories, images or raw arrays. The common guide and human example also omit historical numerical examples.</p>
<table><tr><th>Scene group</th><th>Condition</th><th>Previous native success</th><th>New native success</th><th>New goal true at end</th><th>Finished</th></tr>{''.join(table)}</table>
<p>These are repeated test scenes, not newly held-out cases. Old/new changes include completion checking and context abstraction together; this comparison does not isolate those individual changes. Within this retest both groups receive the same tools and common context.</p></article>
<article><h2>Memory representation</h2><p>A read-only JSON library retrieved by task-word overlap and robot match. Each entry contains a task description, conditional procedure, failed/corrected episode summaries, applicability, limitations and categorical event provenance. It is loaded into context, not trained into model weights. The original full episode is archived outside agent access. Evaluation results do not update memory between cases.</p></article>
<article><h2>Completion checks</h2><p>Luna declares the desired observed condition before movement, selects current material surfaces, and inspects fit overlays. Image/depth matches must support a rigid material association. The tool measures full rigid rotation or material displacement as well as plane-normal angle, and checks current fit bounds against the agent-chosen condition. It does not recognize objects or certify native task completion. An unsupported completed claim returns feedback; the agent may continue or explicitly finish incomplete/uncertain. All issued trajectories execute fully.</p><p>Two diagnostic versions were stopped before this final comparison: the first after two episodes exposed wrong-surface and in-plane-rotation limitations; the second after a reply-serialization bug, with the next launched session stopped immediately after reset. All four diagnostic attempts are preserved separately. The full twelve-case comparison below uses fresh sessions after the fixes. <a href="../subscription-completion-memory-20261019/index.html">Measurement-design diagnostic</a> · <a href="../subscription-completion-memory-v2-20261019/index.html">Serialization diagnostic</a>.</p></article>
<article><h2>Runs</h2><table><tr><th>Case</th><th>Status</th><th>Native goal reached</th><th>Steps</th><th>Agent claim</th><th>Planning failures</th></tr>{''.join(rows)}</table></article>
<article><h2>Videos</h2>{''.join(videos)}</article>
<p><a href="protocol.json">Protocol</a> · <a href="summary.json">Results</a> · <a href="numerical-policy-audit.json">Unchanged numerical policy audit</a> · <a href="../../data/agent_memory/luna-microwave-v2/library/episode-0001/procedure.json">Abstract memory</a></p>''')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',required=True);parser.add_argument('--freeze',action='store_true');a=parser.parse_args()
    study=(ROOT/a.study).resolve()
    if study==ROOT/'results' or not study.is_relative_to(ROOT/'results'):raise ValueError('Use a new results directory')
    study.mkdir(exist_ok=True)
    if a.freeze:freeze(study)
    run(study,extra_arguments=['--completion-checks','--abstract-experience'],reporter=report,diagnostic_reader=diagnose)
