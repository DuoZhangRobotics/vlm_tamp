"""Shared public robot request schemas and inexpensive input validation.

No policy, scene identity, target coordinates, or simulator geometry lives here.
The service enforces these contracts for every caller. Local actors can expose
the same schemas as named functions without inventing an operation envelope.
"""
import json
import math
from numbers import Real

VERSION = 'robot-request-contract-v11'
GROUND_FIELDS = ('camera', 'contact_box_xyxy', 'moving_panel_box_xyxy',
                 'hinge_pixels_xy', 'initial_image_direction_xy')
GROUND_HELP = '''- ground(regions): fit an upright door and vertical handle from current RGB-D.
  Pass ONE regions object with camera, contact_box_xyxy, moving_panel_box_xyxy,
  hinge_pixels_xy, and initial_image_direction_xy; all five fields are required.
  Coordinates use ORIGINAL image pixels: origin top-left, x right, y down.
  See image_sizes in the observation. For width W and height H, use
  0 <= x < W and 0 <= y < H. Normalized 0..1 or 0..1000 coordinates are not accepted
  as a different convention; convert explicitly before calling. No auto-scaling.
  Boxes are [left,top,right,bottom] with positive width/height. contact_box_xyxy
  selects the handle; moving_panel_box_xyxy selects the door. hinge_pixels_xy
  contains two distinct points along the SAME hypothesized hinge edge.
  initial_image_direction_xy is the intended INITIAL movement [dx,dy] of the
  selected contact point in this image, not camera orientation. Its scale is
  irrelevant; a supplied vector must be finite and nonzero. Use null when the
  image direction is unresolved; never use [0,0] as an unknown direction.
  A resolved image-motion hint must agree with the adjacent-depth direction
  hypothesis; disagreement is rejected, not silently overridden. With null,
  adjacent stationary depth may supply a direction hypothesis. Without that
  depth evidence, an unresolved hint is an error. Returned
  motion_direction_check identifies the source; a fit is not proof of the
  hinge hypothesis, retained contact, or task success. No robot motion occurs.
'''

MULTIVIEW_GROUND_HELP = '''- ground_multiview(observation_id, regions, comparison_regions, check_view):
  Fit an upright door from YOUR selections in three CURRENT camera views.
  regions and comparison_regions each use the five-field ground(regions)
  format. Select the same handle, panel and hypothesized hinge edge in both.
  check_view is {"camera": name, "hinge_pixels_xy": [[x,y],[x,y]]} from a THIRD
  view of that same edge. All coordinates are original image pixels. Use the
  current observation_id; do not mix captures. All three cameras must differ.
  Existing depth and camera calibration perform the 3D calculations for you.
  The first two edges triangulate an upright hinge; the third checks its
  projection. Independent depth fits check handle and panel agreement.
  The reply gives accepted, failed_checks, measurements and overlay image paths.
  Inspect the overlays to check part identity and revise your selections or
  choose clearer views when needed. Consistent edges can still name the wrong
  physical part; passing is not proof of an exact hidden hinge or stable grasp.
  A passing fit installs a fresh planning model and invalidates old plans.
  A failed check leaves the current model unchanged. No robot motion occurs.
  This optional tool assumes an upright hinge and vertical handle; it does not
  identify objects automatically or estimate camera calibration.
'''

PART_MOTION_HELP = '''- segment_parts(observation_id, camera, points_xy=null, point_labels=null, box_xyxy=null, label=null):
  Use a coarse original-pixel box or clicked points to request up to three
  MobileSAM masks. Give one point_labels value per point: 1 includes, 0 excludes.
  label is YOUR descriptive name, not a text prompt or an automatic category.
  Choose a returned part_id after inspecting its overlay. Raw mask scores do
  not certify semantic identity. Invalid depth and known robot pixels are
  excluded from usable masks. No hinge annotation or hinge guess is required.
- associate_views(part_id, cameras, comparison_part_id=null):
  Project that part's observed surface into 1..4 cameras of the SAME capture.
  Depth agreement, occlusion and free-space conflicts are reported separately.
  Optionally compare another part_id from one of those views; projected_seed_coverage
  measures overlap. Overlap can mean containment, not necessarily identical parts.
  Returns an association_id and projected-mask overlays; no unseen geometry is filled.
- track_part_motion(part_id, observation_ids, backend="lk", association_id=null, cameras=null):
  Track the selected part through 2..16 ordered saved observations CONTAINING
  its original capture. The part may be selected in any frame, including the
  last; tracking runs forward/backward around that reference. Keep the IDs in
  chronological order. Use lk or optional cotracker3. A supplied association_id
  enables its other camera masks. Select at most four cameras. Without it, use
  the original camera. Returns a track_id, visible-depth support and per-interval
  motion diagnostics, displacement summaries and visible-track overlays.
  Each interval reports fit_correspondences after motion/depth filtering,
  fit_inliers supporting the fitted rigid motion, fit_rms_m on those inliers,
  and fit_failed_checks. A low residual can describe a small subset of the
  matches; these counts and residuals are evidence, not confidence probabilities.
  If tracks disappear, inspect another camera or re-segment a later visible
  reference frame before treating the whole recording as unobservable.
  World coordinates use each frame's depth/calibration.
  Invisible predictions and robot pixels are not measured object motion.
- track_video_motion(part_id, observation_ids, association_id=null, max_rgb_frames=64):
  Use recorded agentview RGB between the first and last supplied depth captures
  with CoTracker3. The ordered 2..16 observation IDs must contain the part's
  reference. Select the part in agentview or associate its mask into that camera
  first. Other cameras use track_part_motion instead.
  Up to max_rgb_frames (2..64) RGB frames are spread across the interval while
  retaining every depth timestamp. No future frames are used. The reply gives
  exact RGB/depth counts, control-step alignment and a saved video_manifest.
  Only measured depth anchors become 3D points; no intermediate depth is invented.
  This reads existing camera JPEGs without rendering or moving the simulator.
- estimate_and_verify_joint(track_id, fit_observations, verify_observations=null):
  Supply a pair of observation IDs to fit an upright revolute-axis hypothesis.
  No old hinge prior is used. An optional later pair tests the fixed hypothesis
  on another motion interval; it may share the fit interval's last frame.
  Returns a numerical world-frame estimate with motion_supported or a specific
  failure. A static part, tiny motion or narrow feature spread can be unobservable.
  verification_failed means observed motion disagreed with the proposed axis.
  It does not isolate an axis error from correspondence errors.
  verification_inconclusive means the checking interval lacked sufficient
  observable motion or supporting points; it neither confirms nor disproves
  the candidate. Both leave motion_supported=false and never install a model.
  This independent fit has no minimum visible-point radius prior. It retains
  measured-motion gates (at least6 moving points,6cm XY span,1.5..70deg rotation,
  rigid RMS below2mm, bootstrap spread below15mm, median point radius below70cm).
  Rejection checks are returned explicitly. These are estimator assumptions,
  not robot execution tolerances or calibrated probabilities of correctness.
  The returned unsigned axis does not specify the desired task direction.
  This estimates an upright hinge only; it is not a general joint-type classifier.
  It does not update the planner model; use apply_joint_estimate explicitly.
- apply_joint_estimate(estimate_id, allow_unverified=false, reuse_fixed_joint=false,
  current_part_id=null, motion_sign=null):
  Install a registered estimate into the current grounded articulated model.
  Without a handle model, current_part_id supplies current observed part geometry
  and motion_sign (+1 or -1) explicitly orients the unsigned fitted axis. This
  enables joint-linked pushing; it does not fabricate a pinch-grasp handle.
  A supported estimate is the default. An admitted but unverified/inconclusive
  fit requires allow_unverified=true and remains labelled provisional. A fit
  that disagreed with its checking motion cannot be applied through this tool.
  Evidence must belong to this episode. By default it must reach the current
  step. reuse_fixed_joint=true explicitly reuses an older fixed-world joint
  hypothesis; it supplies no new current-pose or contact evidence.
  The tool retains the signed task direction and contact geometry, changes the
  hinge parameters as a new model version, and invalidates old plans. Inspect scene support and obtain
  fresh plans afterwards. Use refresh_contact to update visible contact/panel
  geometry while retaining the joint. Ordinary ground replaces the joint too.
  No robot motion is issued.
- apply_part_motion(track_id, reference_observation_id, current_observation_id):
  Apply measured selected-part rigid motion to the existing moving/contact
  geometry. The reference capture must match scene_reference_control_step; the
  final capture must be current. This changes part pose, not the fixed hinge.
  No hand-motion substitution or robot motion occurs.
  The update is rejected if it would move the fixed hinge beyond the configured
  position allowance or put more than 10% of contact samples in observed free
  space in at least two views (5 mm depth tolerance). Rejection changes no model.

The five perception operations read recorded sensors and issue no robot motion;
apply_joint_estimate is the explicit planner-model update. A mask
proposal, same-time correspondence, motion estimate, later-motion support and
physical task success are different claims. Inspect the evidence; none alone
certifies part identity, contact retention or collision-free execution. Offline
CoTracker sees the whole supplied clip, so later-interval support is not a claim
of statistically independent tracking data.
'''


class RequestValidationError(ValueError):
    def __init__(self, operation, field, reason, message):
        super().__init__(f'{operation}: {field}: {message}')
        self.request_diagnostic = {'status': 'invalid_request', 'operation': operation,
            'field': field, 'reason': reason, 'validation_stage': 'before_operation',
            'physical_motion_issued': False, 'expensive_work_started': False,
            'interpretation': 'Request validation failed; geometry and motion feasibility were not tested.'}


def number(**extra):
    return {'type': 'number', **extra}


def array(size, item=None, **extra):
    return {'type': 'array', 'items': item or number(), 'minItems': size, 'maxItems': size, **extra}


def nullable(schema):
    return {'anyOf': [schema, {'type': 'null'}]}


def parameters(properties=None, required=()):
    return {'type': 'object', 'properties': properties or {}, 'required': list(required), 'additionalProperties': False}


def ground_parameters(image_sizes=None):
    image_sizes = image_sizes or {}
    pixel = number(minimum=0)
    if image_sizes:
        pixel['maximum'] = max(max(size['width'], size['height'])-1 for size in image_sizes.values())
    point = array(2, pixel)
    camera = {'type': 'string', 'minLength': 1, 'description': 'Camera name from the current observation; coordinates must refer to this camera.'}
    if image_sizes:
        camera['enum'] = sorted(image_sizes)
    regions = parameters({
        'camera': camera,
        'contact_box_xyxy': array(4, pixel, description='Handle box [left,top,right,bottom], ORIGINAL image pixels.'),
        'moving_panel_box_xyxy': array(4, pixel, description='Moving door box [left,top,right,bottom], ORIGINAL image pixels.'),
        'hinge_pixels_xy': array(2, point, description='Two distinct original-pixel points along the same hypothesized hinge edge.'),
        'initial_image_direction_xy': {**nullable(array(2, number(), **{'not': {'const': [0, 0]}})),
            'description': 'Intended initial CONTACT-POINT motion in image axes (x right, y down), not camera orientation. Finite nonzero vector, or null if unresolved. Adjacent stationary depth geometry chooses the final direction when available.'},
    }, GROUND_FIELDS)
    return parameters({'regions': regions}, ['regions'])


def multiview_ground_parameters(image_sizes=None):
    region=ground_parameters(image_sizes)['properties']['regions']
    check=parameters({k:region['properties'][k] for k in ['camera','hinge_pixels_xy']},
                     ['camera','hinge_pixels_xy'])
    return parameters({'observation_id':{'type':'integer','minimum':0},
        'regions':region,'comparison_regions':region,'check_view':check},
        ['observation_id','regions','comparison_regions','check_view'])


def operation_schemas(image_sizes=None):
    """Schemas reflect public arguments; enabling an operation remains separate."""
    text = {'type': 'string', 'minLength': 1}
    integer = lambda lo, hi: {'type': 'integer', 'minimum': lo, 'maximum': hi}
    angle = {**number(), 'not':{'const':0}}
    speed = number(exclusiveMinimum=0, maximum=1.)
    grasp = {'type': 'string', 'enum': ['point', 'rigid', 'jaw_axis']}
    pose = array(7, description='TCP [x,y,z,qx,qy,qz,qw] in world coordinates; metres and unit quaternion.')
    seed = integer(0, 2**31-1)
    candidates = {'type':'integer','minimum':1}
    box = array(4, number(minimum=0))
    point = array(2, number(minimum=0))
    coordinates = {'type': 'array', 'items': point, 'minItems': 1}
    contact_arguments={'observation_id':integer(0,2**31-1),'camera':text,'region_xyxy':box,
        'target_region_xyxy':nullable(box),'direction_world':nullable(array(3)),
        'distance_m':number(exclusiveMinimum=0),
        'joint_degrees':nullable(angle),
        'joint_model_epoch':nullable(integer(0,2**31-1)),
        'contact_speed_m_s':number(exclusiveMinimum=0),
        'approach_speed_rad_s':speed,'stroke_length_m':nullable(number(exclusiveMinimum=0)),
        'search_seconds':nullable(number(exclusiveMinimum=0)),'stroke_plan_seconds':nullable(number(exclusiveMinimum=0)),
        'max_ineffective_strokes':{'type':'integer','minimum':1},
        'collision_policy':{'type':'string','enum':['advisory','veto']},
        'preview':{'type':'boolean'},'budget_seconds':number(minimum=.001,maximum=1800)}
    policy={'type':'string','enum':['advisory','veto']}
    definitions = {
        'compare_surface_geometry': ('Compare two currently measured surfaces, including different parts. Returns relative plane angles and offsets; no motion or native goal threshold is supplied.',
            parameters({'first_id':text,'second_id':text},['first_id','second_id'])),
        'push': ('Plan and execute simulation pushing with RGB-D feedback. Environment collision predictions are advisory through approach, contact and withdrawal by default; physical contacts remain active. preview=true issues no motion.',
            parameters({**contact_arguments,'mode':{'type':'string','enum':['controlled','push_and_release']},
                'contact_surface':{'type':'string','enum':['fingertips','palm']},'withdrawal_distance_m':number(exclusiveMinimum=0)},['observation_id','camera','region_xyxy'])),
        'hook_pull': ('Catch an observed retaining feature without pinching; test engagement and pull with feedback. A flat surface is not a hook.',
            parameters(contact_arguments,['observation_id','camera','region_xyxy'])),
        'reset': ('Reset the supplied episode and return current sensor observations.', parameters()),
        'observe': ('Read current RGB-D and robot state without motion.', parameters()),
        'motion_settings': ('Read effective motion, collision, numerical and control-budget settings without motion.',parameters()),
        'episode_status': ('Read current success and time/control budgets without motion.', parameters()),
        'await_request': ('Retrieve an existing pending request; never resubmit its action.', parameters({'request_id': {'type': 'string', 'pattern': '^[0-9a-f]{32}$'}}, ['request_id'])),
        'ground': ('Fit observed handle/door geometry from your original-pixel annotations. No motion.', ground_parameters(image_sizes)),
        'refresh_contact': ('Refresh current handle/panel geometry while retaining the fixed-world hinge and signed direction. No motion.',parameters({
            'camera':text,'contact_box_xyxy':box,'moving_panel_box_xyxy':box},['camera','contact_box_xyxy','moving_panel_box_xyxy'])),
        'ground_multiview': ('Check your three-view upright-door hypothesis using calibration and depth; install only a consistent fit. No motion.', multiview_ground_parameters(image_sizes)),
        'segment_parts': ('Propose RGB part masks from your coarse clicks/box; no hinge guess or motion.', parameters({
            'observation_id':integer(0,2**31-1),'camera':text,
            'points_xy':nullable({'type':'array','items':point,'minItems':1,'maxItems':12}),
            'point_labels':nullable({'type':'array','items':{'type':'integer','enum':[0,1]},'minItems':1,'maxItems':12}),
            'box_xyxy':nullable(box),'label':nullable({'type':'string','minLength':1,'maxLength':80})},['observation_id','camera'])),
        'associate_views': ('Project a selected part into other calibrated views with measured depth checks.', parameters({
            'part_id':text,'cameras':{'type':'array','items':text,'minItems':1,'maxItems':4,'uniqueItems':True},
            'comparison_part_id':nullable(text)},['part_id','cameras'])),
        'track_part_motion': ('Track a selected part through recorded RGB-D observations; no hand-motion substitution.', parameters({
            'part_id':text,'observation_ids':{'type':'array','items':integer(0,2**31-1),'minItems':2,'maxItems':16,'uniqueItems':True},
            'backend':{'type':'string','enum':['lk','cotracker3']},'association_id':nullable(text),
            'cameras':nullable({'type':'array','items':text,'minItems':1,'maxItems':4,'uniqueItems':True})},['part_id','observation_ids'])),
        'estimate_and_verify_joint': ('Independently fit an upright hinge from measured motion; optionally test a later interval.',parameters({
            'track_id':text,'fit_observations':array(2,integer(0,2**31-1)),
            'verify_observations':nullable(array(2,integer(0,2**31-1)))},['track_id','fit_observations'])),
        'track_video_motion': ('Track bounded recorded agentview RGB, lifting only supplied measured depth timestamps.',parameters({
            'part_id':text,'observation_ids':{'type':'array','items':integer(0,2**31-1),'minItems':2,'maxItems':16,'uniqueItems':True},
            'association_id':nullable(text),'max_rgb_frames':integer(2,64)},['part_id','observation_ids'])),
        'apply_joint_estimate': ('Explicitly apply a current registered hinge hypothesis; preserve signed task direction and invalidate plans.',parameters({
            'estimate_id':text,'allow_unverified':{'type':'boolean'},'reuse_fixed_joint':{'type':'boolean'},
            'current_part_id':nullable(text),'motion_sign':nullable({'type':'integer','enum':[-1,1]})},['estimate_id'])),
        'apply_part_motion': ('Apply measured part pose between matching captured states; preserve the fixed joint and issue no motion.',parameters({
            'track_id':text,'reference_observation_id':integer(0,2**31-1),'current_observation_id':integer(0,2**31-1)},
            ['track_id','reference_observation_id','current_observation_id'])),
        'inspect_scene': ('Compare selected geometry with current depth; this does not establish contact.', parameters({'camera': nullable(text)})),
        'start_tracking': ('Start tracking selected observed moving geometry; no motion.', parameters()),
        'measure_tracking': ('Measure tracked RGB-D motion without changing the planning model.', parameters()),
        'apply_tracking': ('Apply measured geometry motion, optionally with your updated hinge hypothesis.', parameters({'hinge_world_m': nullable(array(3))})),
        'set_model': ('Set a hinge/axis hypothesis; existing plans become stale.', parameters({'hinge_world_m': nullable(array(3)), 'axis_world': nullable(array(3))})),
        'plan_grasp': ('Select a reachable grasp approach by control cost, using the intended timing. Manipulation is planned after actual closure.', parameters({'probe_degrees': angle, 'max_candidates': candidates, 'seed': seed,
            'speed':speed,'approach_speed':speed,'hold':{'type':'integer','minimum':0},'close_steps':{'type':'integer','minimum':1},'budget_seconds':number(exclusiveMinimum=0,maximum=1800),'collision_policy':policy})),
        'plan_arc': ('Plan the complete requested relative contact arc with explicit error tolerances; no robot motion.', parameters({
            'degrees':angle,'refined':{'type':'boolean'},'grasp_constraint':nullable(grasp),
            'position_tolerance_m':nullable({'type':'number','exclusiveMinimum':0}),
            'orientation_tolerance_deg':nullable({'type':'number','minimum':0,'maximum':180}),
            'max_wrist_reorientation_deg':nullable({'type':'number','exclusiveMinimum':0,'maximum':180}),'collision_policy':policy},['degrees'])),
        'plan_joint': ('Plan simulation free motion under the declared environment prediction policy to seven joint angles in radians.', parameters({'joints': array(7),'collision_policy':policy}, ['joints'])),
        'ik': ('Compute IK proposals for your world TCP pose; no robot motion.', parameters({'pose_world_xyzw': pose}, ['pose_world_xyzw'])),
        'plan_pose': ('Plan simulation free motion under the declared environment prediction policy to your world TCP pose.', parameters({'pose_world_xyzw': pose,'collision_policy':policy}, ['pose_world_xyzw'])),
        'execute': ('Execute a fresh complete plan. gripper=-1 opens, 0 holds, +1 closes; default +1. Choose explicitly for a grasp approach.', parameters({'plan_id': text, 'gripper': {'type': 'integer', 'enum': [-1, 0, 1]}, 'speed': speed, 'hold': {'type':'integer','minimum':0}}, ['plan_id'])),
        'gripper': ('Physically open/close while holding the arm. If measured arm joints have drifted beyond planner limits, command one inward joint reference increment and report it; simulator state is never assigned.', parameters({'close': {'type': 'boolean'}, 'steps': {'type':'integer','minimum':1}})),
        'plan_sequence': ('Plan and price a cost-selected approach; retain the signed manipulation request for fresh planning after measured closure.', parameters({'degrees': angle, 'max_candidates': candidates, 'seed': seed, 'grasp_constraint': grasp,
            'speed':speed,'approach_speed':speed,'hold':{'type':'integer','minimum':0},'close_steps':{'type':'integer','minimum':1},'budget_seconds':number(exclusiveMinimum=0,maximum=1800),'collision_policy':policy}, ['degrees'])),
        'execute_sequence': ('Execute a prepared sequence with the documented gripper stages.', parameters({'plan_id': text, 'close_steps': {'type':'integer','minimum':1}, 'speed': speed, 'approach_speed':speed, 'hold': {'type':'integer','minimum':0}, 'manipulation_gripper': {'type': 'integer', 'enum': [0, 1]},'feedback_degrees':number(exclusiveMinimum=0),'calibrate_joint':{'type':'boolean'},'open_steps':{'type':'integer','minimum':1}}, ['plan_id'])),
        'select_surface_contact': ('Select a current observed surface patch; no grasp or hinge assumption.', parameters({'camera': text, 'region_xyxy': box, 'radius_m': number(minimum=.005, maximum=.04)}, ['camera', 'region_xyxy'])),
        'practice_status': ('Read the granted practice budget and learning phase.', parameters()),
        'request_attempts': ('Request more practice within the existing hard cap with your findings and next tests.', parameters({'learned': text, 'next_experiments': {'type': 'array', 'items': text, 'minItems': 1, 'maxItems': 5}}, ['learned', 'next_experiments'])),
        'dino_landmarks': ('Find optional image-feature candidates; no semantic or motion guarantee.', parameters({'camera': text, 'region_xyxy': nullable(box), 'count': integer(1, 32)})),
        'dino_match': ('Match supplied reference pixels against the current image.', parameters({'reference_observation': {'type': 'integer'}, 'pixels_xy': coordinates, 'reference_camera': text, 'camera': text, 'region_xyxy': nullable(box)}, ['reference_observation', 'pixels_xy'])),
        'declare_completion': ('Declare your observation-based goal before motion.', parameters({'goal': text, 'criteria': {'type': 'array', 'minItems': 1, 'items': parameters({'id': text, 'quantity': {'enum': ['rigid_rotation_deg', 'material_displacement_m', 'plane_angle_deg']}, 'minimum': number(), 'maximum': number(), 'rationale': text}, ['id', 'quantity', 'rationale'])}}, ['goal', 'criteria'])),
        'measure_surface': ('Measure a planar material patch in a saved session observation.', parameters({'observation_id': {'type': 'integer'}, 'camera': text, 'polygon_xy': {'type': 'array', 'items': point, 'minItems': 3, 'maxItems': 32}}, ['observation_id', 'camera', 'polygon_xy'])),
        'compare_surfaces': ('Compare two selected material patches using measured correspondences.', parameters({'first_id': text, 'second_id': text, 'pixels_first_xy': nullable(coordinates), 'pixels_second_xy': nullable(coordinates)}, ['first_id', 'second_id'])),
        'check_completion': ('Check your declared goal against supplied measured evidence.', parameters({'evidence': {'type': 'object', 'additionalProperties': {'type': 'string'}}, 'interpretation': text}, ['evidence', 'interpretation'])),
        'finish': ('Finish only when the active episode policy permits; never fabricates success.', parameters({'claim': {'enum': ['completed', 'incomplete', 'uncertain']}, 'reason': {'type': 'string'}, 'check_id': nullable(text)})),
    }
    return {name: {'type': 'function', 'function': {'name': name, 'description': description, 'parameters': schema}}
            for name, (description, schema) in definitions.items()}


def reject_nonfinite(value, operation, field='arguments'):
    if isinstance(value, dict):
        for key, item in value.items():
            reject_nonfinite(item, operation, f'{field}.{key}')
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            reject_nonfinite(item, operation, f'{field}[{i}]')
    elif isinstance(value, Real) and not math.isfinite(value):
        raise RequestValidationError(operation, field, 'nonfinite_number', 'Use a finite number; NaN and infinity are invalid.')


def validate_operation(operation, arguments, image_sizes=None):
    import jsonschema
    schemas = operation_schemas(image_sizes)
    if not isinstance(operation, str) or operation not in schemas:
        raise RequestValidationError(str(operation), 'operation', 'unknown_operation', 'Use a documented operation name without parentheses; see tools.txt.')
    reject_nonfinite(arguments, operation)
    errors = list(jsonschema.Draft202012Validator(schemas[operation]['function']['parameters']).iter_errors(arguments))
    if errors:
        error = errors[0]
        field = 'arguments' + ''.join(f'[{p}]' if isinstance(p, int) else '.'+p for p in error.absolute_path)
        message = error.message
        if operation == 'ground' and list(error.absolute_path) == ['regions', 'initial_image_direction_xy']:
            message = 'Use a finite nonzero contact-point image-motion vector [dx,dy], or null when unresolved; [0,0] is not an unknown direction.'
        raise RequestValidationError(operation, field, 'schema_mismatch', message[:1000])
    if operation == 'ground':
        validate_ground_regions(arguments['regions'], image_sizes=image_sizes, schema_checked=True)
    if operation == 'refresh_contact':
        if image_sizes and arguments['camera'] not in image_sizes:
            raise RequestValidationError(operation,'camera','unknown_camera','Choose a camera in the current capture')
        for key in ['contact_box_xyxy','moving_panel_box_xyxy']:
            left,top,right,bottom=arguments[key]
            if not(left<right and top<bottom):
                raise RequestValidationError(operation,key,'inverted_or_empty_box','Use a positive-area box in original pixels')
            if image_sizes:
                size=image_sizes[arguments['camera']]
                if not(0<=left<right<size['width'] and 0<=top<bottom<size['height']):
                    raise RequestValidationError(operation,key,'pixel_out_of_bounds','Use original pixels inside the current image')
    if operation == 'ground_multiview':
        for name in ['regions','comparison_regions']:
            try:validate_ground_regions(arguments[name],image_sizes=image_sizes,schema_checked=True)
            except RequestValidationError as error:
                field=error.request_diagnostic['field'].replace('regions.',name+'.',1)
                raise RequestValidationError(operation,field,error.request_diagnostic['reason'],str(error)) from error
        views=[arguments[k] for k in ['regions','comparison_regions','check_view']]
        if len({v['camera'] for v in views})!=3:
            raise RequestValidationError(operation,'cameras','distinct_views_required','Choose three different cameras from the same capture.')
        check=arguments['check_view'];points=check['hinge_pixels_xy']
        if points[0]==points[1]:
            raise RequestValidationError(operation,'check_view.hinge_pixels_xy','coincident_hinge_points','Choose two distinct points on the same hypothesized edge.')
        if image_sizes:
            size=image_sizes[check['camera']]
            if any(not(0<=x<size['width'] and 0<=y<size['height']) for x,y in points):
                raise RequestValidationError(operation,'check_view.hinge_pixels_xy','pixel_out_of_bounds',f"Use original pixels inside {size['width']}x{size['height']}.")
    return arguments


def validate_ground_regions(regions, *, image_sizes=None, schema_checked=False):
    if not schema_checked:
        validate_operation('ground', {'regions': regions}, image_sizes)
        return regions
    for key in ['contact_box_xyxy', 'moving_panel_box_xyxy']:
        left, top, right, bottom = regions[key]
        if not (left < right and top < bottom):
            raise RequestValidationError('ground', 'regions.'+key, 'inverted_or_empty_box', 'Use [left,top,right,bottom] with positive width and height.')
    if regions['hinge_pixels_xy'][0] == regions['hinge_pixels_xy'][1]:
        raise RequestValidationError('ground', 'regions.hinge_pixels_xy', 'coincident_hinge_points', 'Select two distinct points along the same hypothesized hinge edge.')
    direction = regions['initial_image_direction_xy']
    if direction is not None and not any(direction):
        raise RequestValidationError('ground', 'regions.initial_image_direction_xy', 'unresolved_direction', 'Use a nonzero intended contact-point motion vector, or null when unresolved.')
    if image_sizes:
        size = image_sizes[regions['camera']]
        points = [regions[key][i:i+2] for key in ['contact_box_xyxy', 'moving_panel_box_xyxy'] for i in [0, 2]] + regions['hinge_pixels_xy']
        if any(not (0 <= x < size['width'] and 0 <= y < size['height']) for x, y in points):
            raise RequestValidationError('ground', 'regions', 'pixel_out_of_bounds', f"Use original pixels inside the selected {size['width']}x{size['height']} image; no automatic rescaling occurs.")
    return regions


def observed_image_sizes(cameras):
    sizes = {}
    for name, camera in cameras.items():
        view = camera.get('rgb')
        if view is None:
            view = camera['world_points']
        height, width = view.shape[:2]
        sizes[name] = {'width': int(width), 'height': int(height)}
    return sizes


def export_schemas(folder, operations):
    folder.mkdir(parents=True, exist_ok=True)
    schemas = operation_schemas()
    registered = {}
    for name in sorted(set(operations)):
        if name not in schemas:
            raise ValueError('Documented robot operation lacks a request schema: ' + name)
        target = folder / (name + '.json')
        target.write_text(json.dumps(schemas[name], indent=2) + '\n')
        registered[name] = str(target.name)
    return {'version': VERSION, 'operations': registered}
