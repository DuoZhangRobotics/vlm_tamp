"""Luna-high RGB role/primitive choices with geometry measured from depth."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from .api import DurableLedger, request_arguments

ROOT = Path(__file__).resolve().parents[3]
CAMERAS = ('agentview', 'frontview')


class ArticulatedLedger(DurableLedger):
    """Carry forward all previous spend; leave historical ledgers untouched."""

    def __init__(self, directory):
        prior_path = ROOT/'results/rekep-libero-api/ledger.json'
        prior = json.loads(prior_path.read_text())
        if prior['budget_usd'] != 2. or prior['cumulative_conservative_spend_usd'] > 2.:
            raise ValueError('Invalid inherited two-dollar budget')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory/'ledger.json'
        self.prior_spend, self.budget, self.max_calls = prior['cumulative_conservative_spend_usd'], 2., 30
        provenance = {'prior_ledger': str(prior_path.relative_to(ROOT)),
            'prior_ledger_sha256': hashlib.sha256(prior_path.read_bytes()).hexdigest(),
            'inherited_cumulative_spend_usd': self.prior_spend,
            'user_budget_usd': 2., 'reason_for_new_request_scope': 'user requested additional articulated tasks'}
        provenance_path = self.directory/'prior-spend.json'
        if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
            raise ValueError('Inherited spending changed; do not silently reset the budget')
        provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')


PROMPT = '''Ground this robot instruction from the RGB images only: {instruction}

The image is a 2-by-2 sheet. Top row: agentview, raw and DINOv2-numbered.
Bottom row: frontview, raw and DINOv2-numbered. Every individual view is 512x512.
All pixel coordinates below are LOCAL to the selected individual camera view,
not the whole sheet. Numbered points are visual proposals, not object identities.
Choose the view with the clearest intended contact part and moving panel.
The robot grasps a HANDLE or CONTROL, not the entire cabinet or appliance.
For drawers choose the requested ordinal drawer, not a neighboring handle.
For a microwave door identify the actual hinge-side edge of the moving door.
For closing, the door may already be swung open: identify its hinge where it
attaches to the appliance, rather than the free handle edge.
Do not invent world coordinates, joint locations, or an object inventory.
Metric positions and rotations will be computed separately from measured depth.

Return ONLY one JSON object filling this template:
{{
 "camera": "agentview" OR "frontview",
 "grasp_keypoint": an integer visible in that camera, or -1 if no proposal is on the small handle/control,
 "grasp_pixel_xy": [x,y] on the intended handle/control surface,
 "contact_box_xyxy": [left,top,right,bottom] tightly around that handle/control,
 "moving_panel_box_xyxy": [left,top,right,bottom] around ONLY the drawer front, microwave DOOR, or knob,
 "joint_family": "prismatic" OR "revolute",
 "axis_source": "panel_normal" OR "vertical_hinge",
 "motion": "open" OR "close" OR "turn_on",
 "hinge_pixels_xy": [[x,y],[x,y]] at the top and bottom ends of the hinge edge, or null for a drawer/knob,
 "initial_image_direction_xy": [dx,dy], each -1,0,1, describing where the selected contact point should initially move in the chosen image,
 "turn_direction_view": "clockwise" OR "counterclockwise" OR null,
 "travel_m": one of 0.10,0.15,0.18 (drawers), or 0 for revolute,
 "travel_deg": one of 45,60,75,90 (revolute), or 0 for drawers,
 "confidence": "high" OR "medium" OR "low",
 "explanation": one short sentence
}}
Use vertical_hinge for a conventional upright hinged door and panel_normal for
a drawer or knob. A handle fixed to a hinged door rotates WITH the door, so its
contact point follows a circle and the gripper orientation must rotate too.
For a stove control, infer whether its visible mechanism translates or rotates.
State uncertainty with low confidence; do not claim hidden joint measurements.
'''


def response_text(raw):
    return '\n'.join(c.get('text', '') for item in raw.get('output', [])
                     for c in item.get('content', []) if c.get('type') == 'output_text')


def parse_choices(text):
    stripped = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    value = json.loads(stripped)
    if value['camera'] not in CAMERAS or value['joint_family'] not in {'prismatic', 'revolute'}:
        raise ValueError('Invalid camera or mechanism choice')
    if value['axis_source'] not in {'panel_normal', 'vertical_hinge'} or value['motion'] not in {'open','close','turn_on'}:
        raise ValueError('Invalid axis or action choice')
    for key in ['contact_box_xyxy', 'moving_panel_box_xyxy']:
        a = np.asarray(value[key], float)
        if a.shape != (4,) or not np.isfinite(a).all() or not (a >= 0).all() or not (a <= 511).all() or a[2]<=a[0] or a[3]<=a[1]:
            raise ValueError(f'Invalid RGB box: {key}')
    point = np.asarray(value['grasp_pixel_xy'], float)
    if point.shape != (2,) or not np.isfinite(point).all() or (point<0).any() or (point>511).any():
        raise ValueError('Invalid RGB contact pixel')
    if value['joint_family']=='revolute' and value['travel_deg'] not in {45,60,75,90}:
        raise ValueError('Revolute target must be a bounded choice')
    if value['joint_family']=='prismatic' and value['travel_m'] not in {.10,.15,.18}:
        raise ValueError('Prismatic target must be a bounded choice')
    if value['axis_source']=='vertical_hinge':
        hinge = np.asarray(value['hinge_pixels_xy'], float)
        if hinge.shape != (2,2) or not np.isfinite(hinge).all() or (hinge<0).any() or (hinge>511).any():
            raise ValueError('A hinged door needs two visible hinge-edge points')
    return value


def ground(perception, observation, metadata, output, scene_id, *, ledger=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if ledger is None:
        from ..run_api_budget import ledger_for_output
        relative = output.resolve().relative_to(ROOT/'results')
        if len(relative.parts) < 2:
            raise BudgetError('Grounding must belong to an experiment output directory')
        ledger = ledger_for_output(ROOT/'results'/relative.parts[0])
    bundles = {}
    sheet = Image.new('RGB', (1024, 1072), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(CAMERAS):
        bundles[name] = perception.prepare(Path(observation)/'obs.npz', {**metadata, 'main_camera': name}, output/name)
        for column, kind in enumerate(['rgb', 'numbered']):
            draw.text((column*512+10, row*536+5), f'{name}: {kind}', fill='black')
            sheet.paste(Image.open(output/name/f'{kind}.png'), (column*512, row*536+24))
    sheet.save(output/'role-choice-sheet.png')
    prompt = PROMPT.format(instruction=metadata['instruction'])
    (output/'prompt.txt').write_text(prompt)
    arguments = request_arguments(output/'role-choice-sheet.png', prompt, 6000)
    from vlm_tamp.cli import load_credentials
    load_credentials('/home/duo/.codex/keys/vlm-tamp.env')
    raw = ledger.perform(arguments, scene_id, allow_paid=True)
    (output/'response.json').write_text(json.dumps(raw, indent=2)+'\n')
    text = response_text(raw)
    (output/'response.txt').write_text(text)
    if raw.get('status') != 'completed':
        raise ValueError('Luna response did not complete; no automatic retry')
    choices = parse_choices(text)
    bundle = bundles[choices['camera']]
    camera = bundle['cameras'][choices['camera']]
    pixel = np.rint(choices['grasp_pixel_xy']).astype(int)
    x,y = pixel
    source = choices['grasp_keypoint']
    box = choices['contact_box_xyxy']
    synthetic = True
    if isinstance(source, int) and 0 <= source < len(bundle['points_world']):
        sy,sx = bundle['pixels_yx'][source]
        synthetic = not (box[0]<=sx<=box[2] and box[1]<=sy<=box[3])
    if synthetic:
        point = camera['world_points'][y,x]
        if not np.isfinite(point).all():
            raise ValueError('Selected RGB point has no metric depth')
        source = len(bundle['points_world'])
        bundle['points_world'] = np.vstack([bundle['points_world'], point])
        bundle['pixels_yx'] = np.vstack([bundle['pixels_yx'], [y,x]])
    cloud, mask, record = perception.selected_cloud(bundle, source, articulated=True, out_dir=output/'contact-region')
    # The RGB box disambiguates a thin handle from its whole-appliance SAM mask.
    yy,xx = np.indices(mask.shape)
    mask &= (xx>=box[0]) & (xx<=box[2]) & (yy>=box[1]) & (yy<=box[3])
    reference = camera['world_points'][mask]
    if len(reference)<30:
        raise ValueError('Predicted contact box/mask lacks depth support')
    from scipy.spatial import cKDTree
    cloud = cloud[cKDTree(reference).query(cloud)[0] <= .008]
    if len(cloud)<30:
        raise ValueError('Fused predicted contact region is empty')
    np.save(output/'contact-cloud.npy', cloud)
    np.save(output/'contact-mask.npy', mask)
    record.update(supplemental_vlm_pixel_used=synthetic, selected_keypoint=int(source),
                  source_box_from_rgb_vlm=True, points_after_rgb_box_filter=len(cloud))
    result = {'choices': choices, 'source': record, 'provided_inventory': False,
              'simulator_scene_state_used': False, 'model': 'gpt-6-luna', 'reasoning_effort': 'high'}
    (output/'grounding.json').write_text(json.dumps(result, indent=2)+'\n')
    return bundle, choices, cloud, mask
