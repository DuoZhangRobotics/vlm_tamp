"""Prepare the explicitly supplied example and public numerical interface."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.robot_contracts import GROUND_HELP
from vlm_tamp.subscription_tasks import DEFAULT_TASK, TASKS, task_spec

CLIENT=r'''"""Client for the numerical service; transport coordination only."""
import json
from pathlib import Path
import time
import uuid
import sys
import fcntl
import re


def pending(root):
    result=[]
    for path in (root/'requests').glob('*.json'):
        if not re.fullmatch(r'[0-9a-f]{32}\.json',path.name) or (root/'replies'/path.name).exists():continue
        try:data=json.loads(path.read_text())
        except ValueError:continue
        result.append({'request_id':path.stem,'operation':data.get('operation')})
    return result


def await_reply(root,request_id):
    if not isinstance(request_id,str) or not re.fullmatch(r'[0-9a-f]{32}',request_id):
        return {'ok':False,'error':'Use a request_id returned by the transport'}
    response=root/'replies'/(request_id+'.json');deadline=time.monotonic()+900
    while not response.exists():
        if time.monotonic()>deadline:raise TimeoutError('Request still pending; inspect episode_status or await_request, do not resubmit blindly')
        time.sleep(.2)
    return json.loads(response.read_text())


def _call(operation, **arguments):
    root=Path(__file__).resolve().parent
    if operation=='episode_status':
        if arguments:return {'ok':False,'error_type':'ValueError','error':'episode_status takes no arguments'}
        path=root/'robot_data/episode-status.json'
        result=json.loads(path.read_text()) if path.exists() else {'needs_initial_reset':True,'control_steps':0,'remaining_control_steps':600,'task_success':False}
        deadline=root/'robot_data/episode-deadline.json'
        if deadline.exists():result['remaining_wall_seconds']=max(0.,json.loads(deadline.read_text())['deadline_monotonic']-time.monotonic())
        result['in_flight_requests']=pending(root)
        if not result.get('task_success') and result.get('remaining_wall_seconds',1)==0:
            result.update(episode_done=True,termination_reason='wall_budget_exhausted')
        result['control_state_is_last_completed_request']=True
        return {'ok':True,'result':result}
    if operation=='await_request':
        if set(arguments)!={'request_id'}:return {'ok':False,'error_type':'ValueError','error':'await_request requires only request_id'}
        return await_reply(root,arguments['request_id'])
    with (root/'rpc.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return {'ok':False,'error_type':'RequestInFlight','error':'Another request is running. Wait for that client or use await_request; do not resubmit.','in_flight_requests':pending(root)}
        active=pending(root)
        if active:
            return {'ok':False,'error_type':'RequestInFlight','error':'A submitted request still has no reply. Use await_request to collect it before another request.','in_flight_requests':active}
        token=uuid.uuid4().hex;request=root/'requests'/(token+'.json')
        temporary=request.with_suffix('.pending')
        temporary.write_text(json.dumps({'operation':operation,'arguments':arguments}))
        temporary.rename(request)
        return await_reply(root,token)


def call(operation, **arguments):
    begin=time.monotonic()
    try:
        result=_call(operation,**arguments)
    except Exception as error:
        result={'ok':False,'error_type':type(error).__name__,'error':str(error)}
    root=Path(__file__).resolve().parent
    with (root/'robot_data/transport-events.jsonl').open('a') as log:
        log.write(json.dumps({'operation':operation,'ok':result.get('ok'),
            'error_type':result.get('error_type'),'wall_seconds':time.monotonic()-begin,
            'local_status_query':operation=='episode_status',
            'existing_reply_query':operation=='await_request'})+'\n')
    return result


def brief(operation, **arguments):
    response=call(operation,**arguments)
    root=Path(__file__).resolve().parent
    if (root/'robot_interface.py').exists():
        from robot_interface import present
        return present(root,operation,arguments,response)
    return response


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    if (root/'robot_interface.py').exists():
        from robot_interface import main
        main(call,root)
    else:
        print(json.dumps(call(sys.argv[1],**(json.loads(sys.argv[2]) if len(sys.argv)>2 else {})),indent=2))
'''

DOC='''Your task is to open the microwave using physical robot interaction.
This is a development pilot on one case, with at most two resets and 600 control
steps per attempt. You may write and run arbitrary analysis/experiment code in
this workspace. Use python3 for analysis; numpy, scipy and Pillow are checked
before the session starts. Versions are in analysis-environment.json. Inspect images with
your image-viewing tool; save diagnostic images with Pillow if useful.

The local numerical service is called through robot.py:
    from robot import call
    result = call("observe")
or: python3 robot.py observe
Each response is {"ok": true, "result": ...} or {"ok": false, "error": ...}.
Failed planning does not end the experiment. You decide what to do next.
Do not launch tool calls in parallel: the physical robot is one stateful process.
The client admits only one in-flight request. If a call is still running, wait
for its existing process rather than sending it again. episode_status is a
read-only local snapshot that remains available during a long request. It lists
pending request_ids and the last completed request_id. await_request(request_id) retrieves that existing reply
without issuing another robot/planner command. Snapshot control steps reflect
the last completed request; wall time is updated when queried.

Tool interfaces (all coordinate conventions are explicit):
- reset(): reset the same development case; return instruction, current images,
  RGB-D arrays, camera calibration, robot joints, TCP pose and remaining steps.
- observe(): current observations without moving. Images are per-camera PNGs.
  sensors.npz contains <camera>_rgb, _depth (metres), _world_points (HxWx3, metres),
  and _known_robot_depth_prior. Calibration maps optical camera coordinates into
  world. Robot TCP poses are [x,y,z,qx,qy,qz,qw]. Joint angles are radians.
''' + GROUND_HELP + '''
- start_tracking(): select RGB features on current selected moving geometry and
  save a reference. This does not move the robot or refine its model.
- measure_tracking(): match the reference against the latest observation and
  return measured 3D point pairs, robust upright rigid-motion fit and an estimated
  hinge if observable. Does NOT update the planner's model or moving cloud.
  You can also write your own estimator using the saved correspondences.
  Requires start_tracking BEFORE the motion being measured. The automatic
  per-execution motion fit uses a separate reference and does not initialize
  this explicit tracker. Starting a reference now cannot measure past motion.
- apply_tracking(hinge_world_m=null): transport selected geometry by the latest
  measured rigid motion. Optionally set the pivot to the value YOU supply.
  The pivot otherwise stays unchanged. Repeat start_tracking when you choose
  to establish a new reference; no automatic reference renewal occurs.
- set_model(hinge_world_m=null, axis_world=null): set your numerical hypothesis.
  Updating geometry makes previously planned trajectories stale.
- refresh_contact(camera, contact_box_xyxy, moving_panel_box_xyxy): refit the
  currently visible handle and panel from original-pixel boxes while retaining
  the established fixed-world hinge and signed direction. Use this when the part
  moved or contact geometry needs refresh; no hinge reannotation is required.
  Ordinary ground creates a new joint hypothesis instead. Neither certifies a grasp.
- inspect_scene(camera=null): compare the current predicted moving/contact
  points with current calibrated depth and save a coloured overlay. Green points
  have depth support in at least one view; red points conflict with measured
  free space in at least two views; gray points are occluded or unresolved.
  This is geometric evidence, not a semantic or contact guarantee. It changes
  neither the model nor the robot. Failed arc queries also report depth-support
  summaries. Use the images and diagnostics when investigating a scene mismatch.
- motion_settings(): read the effective control budget, speed settings,
  numerical tolerances and environment-collision policy. No motion.
- plan_grasp(probe_degrees=3, max_candidates=24, seed=0, speed=0.6,
  hold=6, close_steps=40, budget_seconds=30, collision_policy="advisory"):
  select a reachable grasp approach by scheduled control cost across IK/posture
  alternatives. Direct and short standoff approaches are considered. Returns an
  approach plan ID; no object motion or grasp retention is assumed. Manipulation
  is planned after actual closure. The search returns its best feasible candidate
  when its computation budget expires; incomplete search does not prove infeasibility.
- execute(plan_id, gripper=1, speed=0.35, hold=6): execute that complete joint
  trajectory; gripper=-1 opens, 0 holds, +1 closes. Speed is maximum joint rad/s.
  For an approach you can request open fingers. Returns new sensor observation.
- gripper(close=true, steps=40): physically close/open while holding the current
  arm joints. Steps are a positive count within the episode budget. If measured
  joints have overshot their bounds, a bounded inward reference recovers them.
  No virtual attachment or simulator state assignment is made.
- plan_arc(degrees, grasp_constraint="point", position_tolerance_m=null,
  orientation_tolerance_deg=null, max_wrist_reorientation_deg=null): plan a relative
  signed nonzero angle around your current pivot/axis. Larger requests use
  multiple planning reference frames internally, without wrapping progress.
  "point" keeps the contact midpoint on its arc with bounded jaw tilt.
  "rigid" transports the measured hand orientation with the moving handle.
  "jaw_axis" transports the line between the fingers, allowing rotation about
  that line. These are different grasp assumptions, not guarantees of retention.
  Position tolerance defaults to 6 mm and is explicit in the reply. It bounds
  contact-path error, not collision clearance. Wrist rotation is free unless you
  explicitly request a limit; physical joint bounds and path checks still apply.
  Rigid/jaw-axis orientation error defaults to 3 degrees; this differs from total
  wrist reorientation. You can request another tolerance (0 uses legacy exact).
  The legacy refined argument is ignored; it estimates no hinge.
  Returns a plan id and joint path, or numerical failure; makes no motion.
  The requested arc is planned in full or fails; it is not silently shortened.
  Failed arc requests include planning_diagnostic with endpoint-projection,
  collision and search-budget evidence plus a readable full_diagnostic JSON.
  These describe the attempted numerical search, not proof of infeasibility.
  plan_arc accepts budget_seconds to change its search allowance (default20s),
  capped by the remaining episode wall time. Changing it does not change geometry.
  After any motion with a grounded articulated scene, accepted RGB-D rigid
  tracking updates the moving cloud for the next query. For contact arcs only,
  visual motion and hand motion are reported separately. Unresolved visual
  motion never becomes an observed part pose through a hand-motion fallback:
  scene_pose_observation reports unconfirmed motion without changing the model.
  scene_pose_update labels an applied estimate's source and fit. It does not
  refine the hinge or certify attachment. Explicit tracking/model tools remain
  available. No update vetoes an executing trajectory.
- ik(pose_world_xyzw): return robot-valid IK proposals for a chosen TCP pose.
- plan_joint(joints, collision_policy="advisory"): free-motion plan to seven
  arm joint angles. Environment predictions are advisory in simulation by default;
  veto can be explicitly requested. Real joint limits and self checks remain.
- plan_sequence(degrees, max_candidates=24, seed=0, grasp_constraint="jaw_axis",
  speed=0.35, approach_speed=0.6, hold=6, close_steps=40, budget_seconds=30,
  collision_policy="advisory"): plan and price a low-cost approach and retain
  the signed manipulation intent. The exact manipulation cost remains unknown
  until real closure is observed. Both motion directions and requests above 90
  degrees are supported; physical reachability and the episode budget still apply.
- execute_sequence(plan_id, close_steps=40, speed=0.35, approach_speed=null,
  hold=6, manipulation_gripper=1, feedback_degrees=15, calibrate_joint=true, open_steps=20): approach with open
  fingers (opening and replanning first when needed), hold the approach reference
  while closing, then replan manipulation
  with the measured aperture and current robot geometry. speed controls
  manipulation; approach_speed defaults to the approach plan's setting.
  With calibrate_joint=true, small initial increments collect RGB-D motion.
  A fitted joint can replace the previous hypothesis only after support from a
  separate later interval and improved prediction error. The revision is reported.
  The signed motion is executed in complete feedback increments, with fresh
  planning between them. Native success terminates immediately. A collapsed pad
  gap or a part that does not follow returns actionable recovery evidence and the
  remaining budget; the episode remains available for local regrasp or push.
  Physical grasp retention is uncertain and reported separately from plan success.
Successful arc/sequence responses are concise. planning_details names a JSON
file with the full numerical evidence when deeper inspection is useful.
Cloud synchronization applies to both continuing-close and neutral-hold
commands during an arc. The neutral command does not disable scene updates.
- finish(): end the current physical attempt and retain the experiment evidence.

The planners check the observed point cloud. Intended finger contact with the
selected part is allowed; arm/palm and other surfaces are checked. Every issued
trajectory executes fully. Observation/model revisions occur BETWEEN trajectories.
The numerical tools inherit existing robot calibration, upright geometry priors,
grasp proposal/filtering and contact/collision implementations. These are supplied
tools, not discoveries by you. You cannot inspect their source or prior solutions.
There is no run-the-entire-task or recovery-policy tool.

The example/ folder is the one human-supplied worked example. It demonstrates
sensor-based hinge refinement. Do not assume its metric coordinates apply here.
No hidden simulator articulation, native object poses or evaluation scores are
available. Determine your progress from images and measurements; the experiment
will be scored independently with the native simulator predicate afterward.

When finished, write findings.txt and guidelines.json: observations, hypotheses,
experiments, observed results, limitations, and reusable conditional guidelines.
For each guideline, include its supporting tool-event numbers or filenames.
Clearly distinguish the supplied example from anything you discovered. Preserve
useful analysis code and images. Avoid storing repeated full point-cloud copies.
Do not request another language model or use network services. Reason within
this actor session; additional model services are unavailable to its tools.
'''

EXAMPLE='''Worked example supplied by the human researcher: refine a hinge from motion.

An RGB-selected hinge edge gives an initial estimate. After a planned motion,
measured door landmarks can disagree with that estimate. Hand motion alone does
not establish door motion because contact can slip or push the wrong surface.

In this earlier development example, we tracked material image corners before
and after a short physical motion, and lifted them through metric depth and
camera calibration. The supplied matches.npz contains reference_world and
current_world arrays from those sensor correspondences. The two PNGs show the
camera before and after the corresponding closure motion.

For an upright mechanism, fit p_after_xy = R p_before_xy + t to consistent moving
points. A fixed pivot h satisfies (I-R)h=t. Outliers, stationary background and
insufficient rotation make this unreliable. Check residuals, spatial support,
rotation magnitude and stability under resampling before trusting h. A reliable
rigid transform can still update the moving cloud when its pivot is uncertain.

The old experiment used observations to revise the model, then planned and
executed subsequent motions. This is an example of a diagnostic process, not a
provided trajectory for the new case. It supplies an upright-axis assumption
and the rigid-motion fitting idea. Your task is to decide whether/when that idea
is useful, choose your own regions and motions, and record what you observe.

There are no simulator joint measurements or true hinge coordinates in these
example files. The sensor-fit.json is a numerical estimate from the correspondences.
'''

DINO_DOC='''
Optional perception tools (available on demand; you may ignore them):
- dino_landmarks(camera="frontview", region_xyxy=null, count=8): local DINOv3
  ViT-S/16 appearance features propose 1..32 numbered image points within your
  optional pixel box. Returns a numbered image, original pixels and measured
  depth-derived world positions. These are unsupervised candidates, not verified
  hinges, grasps or object labels. No MobileSAM or simulator segmentation is used.
- dino_match(reference_observation, pixels_xy, reference_camera="frontview",
  camera="frontview", region_xyxy=null): use a previous observation_id from this
  session and 1..24 of its pixels. Returns up to three current correspondences
  per pixel, annotated images, measured 3D positions, cosine scores, a nonlocal
  score margin and a reciprocal-match check. Ambiguity flags are heuristic and
  scores are not probabilities. Partial occlusion or repeated surfaces can
  produce missing or misleading matches. Inspect the evidence yourself.
These tools do not move the robot, change the planner model, or replace the
optical-flow tracking reference. You decide how to use the returned evidence.
DINO receives RGB only; measured depth is used separately for 3D positions and
robot-pixel filtering. Model loading, inference and tool time are recorded.
'''

ABSTRACT_EXAMPLE='''Human-supplied method: estimate rigid motion from observations.
Track material landmarks before and after a physical motion. Lift corresponding
pixels through current metric depth and calibration. For an upright mechanism,
fit p_after_xy = R p_before_xy + t robustly. A fixed pivot h satisfies (I-R)h=t.
Check residuals, spatial support, rotation and stability under resampling.
Stationary background, outliers and small motion can make the pivot unreliable.
A reliable rigid transform can still transport geometry when the pivot is
uncertain. Hand motion alone does not establish object motion. Use observations
to decide whether to revise a model and obtain new plans. This supplies the same
upright-axis and fitting idea as the approved worked example. Historical images,
coordinates, fit statistics and arrays are omitted; measure the present scene.
'''

COMPLETION_DOC='''
Observation-based completion protocol (no task-specific target is supplied):
Before your first physical action, translate the instruction into an observable
terminal condition. State assumptions if the instruction is ambiguous. Partial
progress is not automatically the intended final state. Choose a measurable goal
range from the instruction and scene, not from old example numbers. Declare it
once; do not weaken it after difficulty. This does not change motion execution.

- declare_completion(goal, criteria): criteria is a nonempty list of objects
  {id: a unique name, quantity: a measured quantity, minimum: chosen lower bound,
   maximum: chosen upper bound, rationale: why this follows from the instruction}.
  At least one bound is required. Declare before moving. No builtin object,
  task name, success angle, grasp rule or motion sequence is encoded here.
  Supported quantities are rigid_rotation_deg, material_displacement_m and
  plane_angle_deg. Choose a quantity that actually observes the requested motion.
- measure_surface(observation_id, camera, polygon_xy): select a planar material
  patch using three or more pixel vertices in a saved current-session image.
  Fits a robust plane to measured world points, removing visible robot pixels.
  Returns a measurement_id, world normal/centroid, support spans, residuals,
  inliers, stability samples and a colored overlay. Inspect the overlay: a good
  plane fit can still be the wrong surface. Green points are inliers; red are
  rejected points. Avoid background, occlusion and narrow edge-only selections.
- compare_surfaces(first_id, second_id, pixels_first_xy=null, pixels_second_xy=null):
  compare selected material patches using mutual image-feature matches and a
  robust three-dimensional rigid transform. Both arrays may optionally supply
  six to sixty-four matched current-session pixels, for example from your DINO
  tool or own analysis; each point must lie on its selected measured surface.
  Matches must have spatial support and consistent metric depth motion. Returns
  association evidence, rigid_rotation_deg, material_displacement_m at the initial
  selected patch centroid, plane_angle_deg, and fit-stability bounds when available.
  Full rigid rotation can observe rotation WITHIN a plane, which its normal
  cannot. Rigid rotation magnitude is unsigned and limited to a half-turn.
  Plane angle is UNSIGNED and limited to zero through ninety degrees: it cannot
  distinguish rotation beyond a right angle. Choose initial and current patches
  of the same material surface; the tool does not identify objects. It never
  substitutes planned angles or TCP motion for observed object geometry.
  A plane fit alone is never association evidence. Check quantity_supported.
  If a large view change loses matches, you can measure intermediate saved
  observations and compare successive material patches. Previously supported
  correspondences can form an identity chain back to the initial patch, with
  accumulated rotation-stability margins. This adds no motion or model update.
  Unsupported matches require better evidence or an uncertain outcome, not
  assuming that similarly oriented or differently oriented patches are one object.
- check_completion(evidence, interpretation): evidence maps each criterion id
  to a comparison measurement_id. Explain surface identity, visual corroboration
  and uncertainty. The tool compares fit-stability bounds with your chosen range
  and checks freshness. A change in control steps makes an old check stale.
  Fit stability is not calibrated confidence; check another view or selection
  when uncertain. A supported check is evidence for YOUR declared conditions,
  not proof of native benchmark success or semantic identity.
- finish(claim, reason, check_id=null): claim is completed, incomplete or uncertain.
  completed requires a fresh supported check. An unsupported claim leaves the
  episode open and returns feedback. You may improve measurements or continue
  if appropriate. You can always stop explicitly as incomplete or uncertain
  with a reason; neither is automatically converted into success or failure.
  Native task scoring remains separate. Every issued trajectory executes fully.

Use fresh initial measurements and final measurements from this episode. A
planned motion, gripper closure, apparent gap, or plausible view alone is not a
  quantitative completion check. Save your completion reasoning and evidence in
findings.txt. These tools measure planar orientation; they do not provide a
universal completion detector for arbitrary semantic tasks.
'''


def completion_document(advisory=False):
    if not advisory:return COMPLETION_DOC
    interfaces=COMPLETION_DOC.split('- declare_completion',1)[1].split('- finish(',1)[0]
    return '''
Optional observation-based completion measurements (advisory):
Interpret the instruction and judge progress from fresh observations. Record the
evidence and uncertainty behind your judgment. Geometry tools can supplement
visual assessment; unavailable feature matches do not imply physical failure.
If you choose quantitative criteria, declare them before moving as below.
- declare_completion'''+interfaces+'''- finish(claim, reason, check_id=null): end this physical attempt and record
your judgment as completed, incomplete or uncertain with an observed reason.
A supported quantitative check is optional. An accepted claim does not establish
native task success: the independent evaluator scores that separately. Preserve
measurement uncertainty instead of inventing a verified value. This ends the
attempt, not the practice session; reset for another while budget remains.
Every issued trajectory executes fully. Measurements do not veto execution or
require you to end practice. Write your evidence and limitations in findings.txt.
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--output',required=True)
    p.add_argument('--control-steps',type=int,default=600)
    p.add_argument('--reference-mode',choices=['measured_delta','slew_limited'],default='slew_limited')
    p.add_argument('--state',type=int,default=21);p.add_argument('--dino-tools',action='store_true')
    p.add_argument('--task',choices=sorted(TASKS),default=DEFAULT_TASK)
    p.add_argument('--landmark-evidence',action='store_true')
    p.add_argument('--multiview-grounding',action='store_true')
    p.add_argument('--part-motion-tools',action='store_true')
    p.add_argument('--guidelines-source');p.add_argument('--evaluation-case',action='store_true')
    p.add_argument('--attempts',type=int,choices=range(1,21),default=2)
    p.add_argument('--learning-practice',action='store_true')
    p.add_argument('--practice-cases')
    p.add_argument('--initial-attempts',type=int)
    p.add_argument('--stop-on-success',action='store_true')
    p.add_argument('--persistent-episode',action='store_true')
    p.add_argument('--surface-contact-tools',action='store_true')
    p.add_argument('--compact-interface',action='store_true')
    p.add_argument('--visual-inspection',action='store_true',help='Local retrieval/cropping of already captured public observations')
    p.add_argument('--successful-method-file')
    p.add_argument('--memory-store')
    p.add_argument('--completion-checks',action='store_true')
    p.add_argument('--abstract-experience',action='store_true')
    p.add_argument('--no-example',action='store_true',help='Supply tool contracts without a worked example or example images')
    a=p.parse_args();work=Path(a.workspace);out=Path(a.output)
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from vlm_tamp.subscription_agent import validate_attempt_budget
    validate_attempt_budget(a.attempts,learning_practice=a.learning_practice,evaluation_case=a.evaluation_case,initial_attempts=a.initial_attempts)
    work.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    from vlm_tamp.public_analysis import prepare_workspace
    prepare_workspace(work)
    selected_task = task_spec(a.task)
    part_motion_introduction=None
    document=DOC.replace('Your task is to open the microwave', 'Your task is to '+selected_task['instruction'])+(DINO_DOC if a.dino_tools else '')
    if a.no_example:
        document=document.replace('The example/ folder is the one human-supplied worked example. It demonstrates\nsensor-based hinge refinement. Do not assume its metric coordinates apply here.',
            'No worked example or prior solution is supplied.')
        document=document.replace('Clearly distinguish the supplied example from anything you discovered.',
            'Distinguish supplied tool contracts from anything you discovered.')
    document+='\nUse the current instruction returned by reset. Earlier examples or experience may describe a different task or motion direction.\n'
    if a.multiview_grounding or a.part_motion_tools:
        from vlm_tamp.robot_contracts import MULTIVIEW_GROUND_HELP
        document+='\nOptional calibrated multiview geometry:\n'+MULTIVIEW_GROUND_HELP
    if a.part_motion_tools:
        from vlm_tamp.robot_contracts import PART_MOTION_HELP
        part_motion_introduction=(ROOT/'prompts/part_motion_introduction.txt').read_text()
        (work/'pipeline-introduction.txt').write_text(part_motion_introduction)
        (out/'pipeline-introduction.txt').write_text(part_motion_introduction)
        document=part_motion_introduction+'\n'+document
        document+='\nOptional part masks, cross-view correspondence and measured articulation:\n'+PART_MOTION_HELP
    if a.landmark_evidence:
        document+='''
Optional measured landmark feedback:
Motion fits include landmark_displacements from the same public RGB-D matches
BEFORE the hinge estimator filters near-stationary points. It reports finite
pair count, count at or below 1.5 mm XY displacement, displacement quantiles,
vertical-drift count and the pairs entering the existing hinge fit. These are
point measurements, not a motion class or contact certificate. Selection may
include background and matches can be wrong. Compare with the images and fit
provenance. This feedback does not change fitting, planning or execution.
'''
    if a.completion_checks:
        if a.stop_on_success:
            # Keep the ordinary native-success/persistence finish contract.
            # Measurements add evidence, not a second termination authority.
            document+=completion_document(advisory=True).split('- finish(',1)[0]
        else:
            document=document.replace('- finish(): end the current physical attempt and retain the experiment evidence.',
                '- finish: use the observation-based completion interface documented below.')
            document+=completion_document(advisory=a.learning_practice)
        document+='''\n- compare_surface_geometry(first_id, second_id): compare two current measure_surface
  results, which may name DIFFERENT parts. Returns relative plane angle and
  centroid-to-plane offsets without assuming material correspondence or object
  motion. Both selections must be from the current physical state. Use the
  values, current images and your goal interpretation to assess progress;
  no native goal threshold is provided. A false task_success means not yet
  complete; it does not mean the last action failed or moved in the wrong direction.\n'''
        document+='''Choose material surface patches for depth measurements. Depth
through a window or opening can belong to background rather than the intended
surface. Inspect the returned support, residuals and overlays; a good plane fit
alone does not establish part identity.\n'''
    if a.attempts==1:
        document=document.replace('at most two resets','exactly one reset')
        document+='\nThis run permits ONE physical attempt only. No reset retry is available.\n'
    elif a.learning_practice:
        document=document.replace('at most two resets',f'at most {a.attempts} resets')
        document+=f'''\nThis is learning practice: at most {a.attempts} physical attempts,
each with 600 recorded control steps. Each reset restores the same supplied
development scene and begins a fresh attempt. Your reasoning session, analysis
files and notes persist. finish ends one attempt, not your learning session;
reset starts the next while budget remains. Save per-attempt findings before
resetting. Re-measure the scene and obtain fresh plans for each attempt.
Use repeated attempts to test explanations and check whether a correction works
again from reset. These are practice results, not held-out evaluation results.\n'''
    if a.initial_attempts is not None:
        document+=f'''\nAdaptive attempt budget: initially {a.initial_attempts} attempts are
granted, with a hard maximum of {a.attempts}. When a granted block is exhausted,
finish the active attempt, review your notes and either end practice or call:
- request_attempts(learned, next_experiments): learned is a nonempty summary of
  findings and uncertainty; next_experiments is a list of one to five concrete
  tests or repeatability checks. Grants up to five more, bounded by the hard cap.
  This checks the budget and request format, not whether your hypothesis is true.
- practice_status(): read granted, used and remaining attempt budgets.
Budget extensions use your stated tests and the budget; phase changes use the
reported success booleans. You may stop early with a documented
reason. Report repeatable success only with observed evidence from fresh resets.\n'''
        document+='''\nStaged learning: practice_status and reset replies provide a learning_phase
with current instructions. The phase is success_first until two consecutive
simulator successes occur in this session. Only then are efficiency instructions
released. Continue following the current phase; recorded failures remain failures.\n'''
    if a.memory_store:
        import sys
        sys.path.insert(0,str(ROOT/'src'))
        from vlm_tamp.episodic_memory import client_source,store_hashes
        store=Path(a.memory_store).resolve()
        staged_memory=work/'episodic_library'
        shutil.copytree(store,staged_memory)
        if store_hashes(staged_memory)!=store_hashes(store):raise ValueError('Memory staging mismatch')
        (work/'memory.py').write_text(client_source(staged_memory))
        document=document.replace('You cannot inspect their source or prior solutions.',
            'Implementation source remains unavailable. Episodes returned by the memory interface are explicitly available.')
        document+='''\nExperience retrieval: from memory import recall; recall(instruction, robot="Panda")
returns matching saved episodes, their procedure and public evidence paths, or an
empty list. Call this after the first observation and before physical movement.
The library is read-only. It contains past observations and agent-authored claims,
not current hidden state. Inspect applicability to the current scene. Past pixel
coordinates, joint paths and world poses are evidence, not automatic guarantees
of a match. Reuse useful decisions; obtain fresh valid plans through robot.py.
Memory procedures may describe other attempt budgets; the current documented
attempt limit takes precedence. Record memory-use.json with retrieved entry ids,
what you reused or changed, evidence for applicability, and limitations.\n'''
    if a.evaluation_case:document=document.replace('This is a development pilot on one case','This is a frozen initial-state evaluation on one case')
    if a.abstract_experience and not a.no_example:
        document+='''\nThis condition supplies only abstract prior context. The common guide and
approved example contain method descriptions without historical numerical
examples. If episodic memory is present, only its conditional prose and
categorical event provenance are available. Original images, raw sensor arrays,
pixel coordinates, world poses, recorded trajectories, tool arguments and
historical analysis code are deliberately excluded. Obtain geometry anew.\n'''
    if a.successful_method_file:
        document+='''\nRead experience/successful-method.json before acting. It contains the
previous successful agent's own methodology, its observed repeatability and
limitations. This newer verified method is the starting point for reproduction;
older retrieved lessons are background. It includes previously tested relative
control choices but excludes historical pixel regions, world coordinates, joint
paths and raw tool replies. Re-measure geometry and generate fresh plans.
This supplied method is read-only. Preserve changes in your own findings.\n'''
    if a.stop_on_success:
        document=document.replace('Every issued\ntrajectory executes fully.',
            'Every issued trajectory executes until complete or simulator success ends the episode.')
        document=document.replace('Every issued trajectory executes fully.',
            'Every issued trajectory executes until complete or simulator success ends the episode.')
        document+='''\nSimulator-success termination: after each control step, the simulator
checks the official task predicate. On success the remaining trajectory is
discarded, the episode is closed and temporary simulation data are compacted.
Execution replies include task_success and episode_done booleans plus a final
observation. No release, retreat or return-home step is required after success.
Further motion in that episode is blocked. observe returns the saved final view;
finish is idempotent after success. Reset to test the method in a fresh attempt.
The success boolean is authorized learning feedback. Object identities, object
poses, native joint axes and thresholds remain private. Geometric completion
tools are optional diagnostics; use task_success for the benchmark outcome.
While task_success is false, continue within budget when a useful action remains.
finish(claim="completed") cannot override the simulator's result. You may end a
failed attempt as incomplete or uncertain and then reset to test a correction.\n'''
    if a.persistent_episode:
        document=document.replace('- finish(): end the current physical attempt and retain the experiment evidence.',
            '- finish(): finalize only after simulator success or an exhausted episode budget; an early request is rejected.')
        document=document.replace('You may end a\nfailed attempt as incomplete or uncertain and then reset to test a correction.',
            'An incomplete episode must continue until success or an actual budget limit. No reset retry is allowed.')
        document+='''\nPersistence policy:
- episode_status(): read native success, remaining control steps, remaining wall
  time and whether the episode is terminal. The limit is shared across all
  reasoning turns of this episode.
Do not voluntarily stop while success is false and budget remains. A failed
plan, uncertain perception or unsuccessful contact is feedback for another
decision, not episode termination. If you end a reasoning turn early, the same
agent context will be resumed with a budget reminder. No reset is performed.
Use the remaining budget purposefully; do not issue idle or meaningless actions
merely to consume control steps. Runtime/account failures are reported separately
from simulator outcomes and do not become fabricated budget exhaustion.\n'''
    if a.surface_contact_tools:
        document+='\n'+(ROOT/'prompts/contact_skills_introduction.txt').read_text()+'\n'
        document+='''\nGeneral multi-stage manipulation:
A task may require several stages and different contacts. A continuous grasp is
not mandatory. Grasping, releasing, regrasping, pushing, pulling and sliding are
possible choices when justified by current observations and the available tools.
You choose the contact surface, contact mode, geometry and stage sequence.
Past successful recipes are hypotheses, not restrictions to one contact mode.
No specific object surface, direction, trajectory or recovery sequence is supplied.

Additional generic contact interfaces:
- select_surface_contact(camera, region_xyxy, radius_m=0.02): select a bounded
  patch of the CURRENT observed surface. Radius is 0.005..0.04 metres. Returns a
  measured point, approximate surface normal toward the camera, point clouds,
  and configured finger-pad centres in the TCP frame. It builds a fresh
  observed scene and permits configured-finger contact with only that patch.
  Arm, palm and other observed surfaces remain checked. No grasp, closure,
  cylinder fit or hinge model is required. No object motion is predicted.
  This switches to a free-motion surface scene. It does not move the robot.
- plan_pose(pose_world_xyzw): find IK and a collision-aware joint path to YOUR
  requested TCP pose in the current scene. It returns a plan_id for execute.
  No gripper closure or grasp sequence is imposed. plan_joint is also available.
  For a chosen contact landmark c in the TCP frame and orientation R, the TCP
  translation placing it at observed point p is p - R*c. Choose orientations,
  offsets and motion stages from current geometry; the tool supplies no policy.
Choose the gripper setting before selecting/planning with its geometry. After a
plan, execute with gripper=0 to keep that aperture command; changing the gripper
during the motion changes the geometry for which the path was checked. After a
physical motion, refresh select_surface_contact from current observations before
planning again in this mode. This prevents reusing a pre-motion static snapshot.
An articulated arc requires a new ground call; a surface patch is not a hinge
estimate. Planned finger contact does not certify physical contact or task
progress. The simulator's success signal remains the stopping criterion.\n'''
    if a.persistent_episode and a.learning_practice:
        document=document.replace('No reset retry is allowed.',
            'Reset for another granted practice attempt only after the current episode is terminal.')
        document+=f'''\nPersistent adaptive practice: an active physical episode must reach
simulator success or its 600-step / 30-minute budget before another reset.
Each reset renews the episode budget, within the total session limit. After a
terminal episode, practice_status, request_attempts and reset remain available.
A final response before practice completion resumes this same learning context.
Complete at least the initial {a.initial_attempts or a.attempts} attempts. After two consecutive successes,
you may stop with a documented method and limitations; efficiency experiments
are optional in this recovery study. If repeatability is not established, extend
in blocks up to the {a.attempts}-attempt cap. A final practice-summary.json must include
best_supported_method and stop_reason. These are learning claims, not native
success evidence. All earlier failures remain in the study.\n'''
    from vlm_tamp.execution_limits import checked_control_steps
    checked_control_steps(a.control_steps)
    if a.practice_cases:
        if not a.learning_practice or a.evaluation_case:raise ValueError('Curriculum is practice-only')
        import re
        document=re.sub(r'Each reset restores the same supplied[^.]*\.',
            'Each reset selects a declared training case; task and starting configuration may change.',document)
        document='Practice curriculum: follow the instruction and current observations returned by every reset. Notes persist across cases; success in one case does not establish success in another. Complete the initial grant and record applicability separately.\n\n'+document
    document+=f'\nController reference mode: {a.reference_mode}. motion_settings reports the effective limits.\n'
    document=document.replace('600 control',f'{a.control_steps} control').replace('600 recorded control',f'{a.control_steps} recorded control').replace('600-step',f'{a.control_steps}-step')
    client=CLIENT.replace("'remaining_control_steps':600",f"'remaining_control_steps':{a.control_steps}")
    (work/'robot.py').write_text(client);(work/'tools.txt').write_text(document)
    interface=None
    if a.compact_interface:
        from vlm_tamp.compact_robot_interface import install
        interface=install(work,document,attempts=a.attempts,
            initial_attempts=a.initial_attempts or a.attempts,practice_mode=a.learning_practice,control_steps=a.control_steps)
    else:
        import re
        from vlm_tamp.robot_contracts import export_schemas
        names=re.findall(r'(?m)^- ([a-z][a-z_]+)(?:\(|:)',document)
        registry=export_schemas(work/'tools/schemas',[*names,'episode_status','await_request'])
        (work/'tools/schema-index.json').write_text(json.dumps(registry,indent=2)+'\n')
    if a.landmark_evidence:
        with (work/'tools.txt').open('a') as index:
            index.write('\nMotion-fit feedback includes landmark_displacements: measured tracked-point counts and displacement quantiles before hinge-fit filtering. These do not certify object identity or retained contact.\n')
    if interface and part_motion_introduction is not None:
        index=work/'tools.txt'
        index.write_text(part_motion_introduction+'\n'+index.read_text())
        interface['index_bytes']=index.stat().st_size
    visual_interface=None
    if a.visual_inspection:
        from vlm_tamp.visual_inspection import install as install_visual_inspection
        visual_interface=install_visual_inspection(work)
    if a.guidelines_source:
        experience=work/'experience';experience.mkdir(exist_ok=True)
        shutil.copyfile(a.guidelines_source,experience/'guidelines.json')
        with (work/'tools.txt').open('a') as index:
            index.write('\nFrozen learner experience is available at experience/guidelines.json. Read it before acting, check applicability to the current observations, and re-estimate geometry. Its claims are prior experience, not current state or guaranteed outcomes. Do not update this frozen input during evaluation.\n')
    if a.successful_method_file:
        experience=work/'experience';experience.mkdir(exist_ok=True)
        shutil.copyfile(a.successful_method_file,experience/'successful-method.json')
    example=work/'example'
    if a.no_example:
        if example.exists():raise ValueError('No-example preparation requires a workspace without an existing example directory')
    else:
        example.mkdir(exist_ok=True)
        (example/'readme.txt').write_text(ABSTRACT_EXAMPLE if a.abstract_experience else EXAMPLE)
    source=ROOT/'results/state11-observed-contact-recovery-20261015/open_microwave-state011'
    mapping={'closure-landmark-matches.npz':'matches.npz','closure-rigid-tracking.json':'sensor-fit.json',
        'opened/frontview.png':'before.png','closed/frontview.png':'after.png'}
    provenance={}
    for name,dest in mapping.items():
        if a.abstract_experience or a.no_example:continue
        origin=source/name
        if origin.exists():
            shutil.copyfile(origin,example/dest)
            provenance[dest]={'source':str(origin),'sha256':hashlib.sha256(origin.read_bytes()).hexdigest()}
    write={'model':'gpt-6-luna','reasoning_effort':'high','authentication':'ChatGPT subscription',
        'separately_billed_model_calls':False,'development_only':not a.evaluation_case,
        'case':{'task':a.task,'task_name':selected_task['task_name'],'suite':selected_task['suite'],'state_index':a.state,'seed':0},
        'case_previously_used_in_project':None if a.evaluation_case else a.state==21,
        'held_out_initial_state_evaluation':bool(a.evaluation_case),'held_out_generalization_claim':False,
        'optional_dino_tools_available':bool(a.dino_tools),
        'frozen_agent_guidelines_sha256':hashlib.sha256(Path(a.guidelines_source).read_bytes()).hexdigest() if a.guidelines_source else None,
        'maximum_attempts':a.attempts,'maximum_control_steps_per_attempt':a.control_steps,'controller_reference_mode':a.reference_mode,
        'human_supplied_example':provenance,'agent_source_access':False,
        'agent_recovery_policy_supplied':False,'inherited_numerical_tool_implementations':True,
        'workspace':str(work.resolve()),'current_conversation_inherited':False}
    write['completion_checks']=bool(a.completion_checks)
    write['learning_practice']=bool(a.learning_practice)
    if a.practice_cases:write['private_practice_curriculum']=json.loads(Path(a.practice_cases).read_text())
    write['same_agent_session_across_attempts']=bool(a.learning_practice)
    write['completion_policy']='advisory' if (a.learning_practice or a.stop_on_success) and a.completion_checks else 'required_measurement' if a.completion_checks else 'not_enabled'
    write['initial_attempts_granted']=a.initial_attempts if a.initial_attempts is not None else a.attempts
    write['adaptive_attempt_budget']=a.initial_attempts is not None
    write['stop_on_simulator_success']=bool(a.stop_on_success)
    write['persistent_episode']=bool(a.persistent_episode)
    write['surface_contact_tools']=bool(a.surface_contact_tools)
    write['public_interface']=interface or {'mode':'full'}
    write['visual_inspection']=visual_interface or {'enabled':False}
    write['landmark_evidence']=bool(a.landmark_evidence)
    write['multiview_grounding']=bool(a.multiview_grounding or a.part_motion_tools)
    write['part_motion_tools']=bool(a.part_motion_tools)
    write['part_motion_introduction']={
        'file':'pipeline-introduction.txt','included_in':'tools.txt',
        'sha256':hashlib.sha256(part_motion_introduction.encode()).hexdigest(),
        'word_count':len(part_motion_introduction.split()),
        'scope':'Generic tool onboarding; not task-specific answers or learner-authored memory',
    } if part_motion_introduction is not None else None
    write['task_outcome_feedback']='success_boolean' if a.stop_on_success else 'none'
    write['successful_method_sha256']=hashlib.sha256(Path(a.successful_method_file).read_bytes()).hexdigest() if a.successful_method_file else None
    write['delayed_efficiency_instructions']=bool(a.learning_practice and a.initial_attempts is not None)
    write['abstract_experience']=bool(a.abstract_experience)
    write['example_supplied']=not a.no_example
    write['human_example_representation']='none' if a.no_example else 'method-only prose' if a.abstract_experience else 'worked RGB-D example'
    if a.memory_store:
        write['episodic_memory']={'store':str(store),'files_sha256':store_hashes(store),
            'staged_path':str(staged_memory),'read_only':True,'automatic_cross_case_updates':False}
        write['agent_authored_procedure_supplied']=bool(list(store.glob('*/entry.json')))
        write['held_out_initial_state_evaluation']=False
        write['evaluation_scope']='See study manifest: same-case repetitions and transfer scenes are scored separately.'
    (out/'protocol.json').write_text(json.dumps(write,indent=2)+'\n')
    initial=out/'public-inputs';initial.mkdir(exist_ok=True)
    for name in ['robot.py','tools.txt']:
        shutil.copyfile(work/name,initial/name)
    if interface:
        shutil.copyfile(work/'robot_interface.py',initial/'robot_interface.py')
        shutil.copytree(work/'tools',initial/'tools',dirs_exist_ok=True)
    if (work/'memory.py').exists():shutil.copyfile(work/'memory.py',initial/'memory.py')
    if visual_interface:
        shutil.copyfile(work/'visual.py',initial/'visual.py')
        if not interface:
            (initial/'tools').mkdir(exist_ok=True)
            shutil.copyfile(work/'tools/visual_inspection.txt',initial/'tools/visual_inspection.txt')
    if example.exists():shutil.copytree(example,initial/'example',dirs_exist_ok=True)
    if (work/'experience').exists():shutil.copytree(work/'experience',initial/'experience',dirs_exist_ok=True)
    sources=out/'source-snapshot';sources.mkdir(exist_ok=True)
    for name in ['scripts/prepare_subscription_pilot.py','scripts/serve_subscription_robot.py','src/vlm_tamp/subscription_agent.py']:
        dest=sources/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    print(json.dumps({'workspace':str(work),'example_files':list(provenance)}))


if __name__=='__main__':main()
