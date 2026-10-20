"""Versioned visual choices, independent of benchmark labels and state indices."""
import json
import re
import numpy as np


class InteractionError(ValueError):
    pass


def _keys(value, names, name):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise InteractionError(f'{name}: expected fields {names}')


def _choice(value, options, name):
    if not isinstance(value, str) or value not in options.split():
        raise InteractionError(f'{name}: unsupported choice')


def _pixels(value, shape, name, *, nullable=False):
    if nullable and value is None:
        return
    try:
        a = np.asarray(value, float)
    except (TypeError, ValueError) as error:
        raise InteractionError(f'{name}: numeric pixels required') from error
    if a.shape != shape or not np.isfinite(a).all() or (a < 0).any() or (a > 511).any():
        raise InteractionError(f'{name}: finite local 0..511 coordinates required')
    if shape == (4,) and (a[2] <= a[0] or a[3] <= a[1]):
        raise InteractionError(f'{name}: empty box')
    if shape == (2, 2) and np.linalg.norm(a[1] - a[0]) < 2:
        raise InteractionError(f'{name}: distinct endpoints required')


def parse_interaction(text):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    try:
        v = json.loads(text)
    except ValueError as error:
        raise InteractionError('Invalid interaction JSON') from error
    _keys(v, 'version camera contact moving mechanism goal contact_roles confidence explanation', 'root')
    if v['version'] != 'interaction-v1':
        raise InteractionError('Unsupported interaction version')
    _choice(v['camera'], 'agentview frontview', 'camera')
    _choice(v['confidence'], 'high medium low', 'confidence')
    if not isinstance(v['explanation'], str) or len(v['explanation']) > 1500:
        raise InteractionError('A bounded explanation is required')
    c, m, k, g = (v[n] for n in ['contact', 'moving', 'mechanism', 'goal'])
    _keys(c, 'shape box_xyxy pixel_xy axis_pixels_xy', 'contact')
    _keys(m, 'shape box_xyxy', 'moving')
    _keys(k, 'family axis_source axis_pixels_xy support_box_xyxy motion initial_image_direction_xy turn_direction_view', 'mechanism')
    _keys(g, 'relation reference_box_xyxy', 'goal')
    _choice(c['shape'], 'straight_bar round_control rim_or_edge surface_patch unknown', 'contact.shape')
    _choice(m['shape'], 'planar_panel compact_control free_body unknown', 'moving.shape')
    _choice(k['family'], 'free prismatic revolute unknown', 'mechanism.family')
    _choice(k['axis_source'], 'none panel_normal hinge_edge control_face_normal unknown', 'axis_source')
    _choice(k['motion'], 'place open close turn_on', 'motion')
    _choice(g['relation'], 'on_support aligned_with_reference_plane quarter_turn hand_clearance unknown', 'goal.relation')
    for value, name in [(c['box_xyxy'], 'contact'), (m['box_xyxy'], 'moving')]:
        _pixels(value, (4,), name)
    _pixels(c['pixel_xy'], (2,), 'contact.pixel')
    _pixels(c['axis_pixels_xy'], (2, 2), 'contact.axis', nullable=True)
    _pixels(k['axis_pixels_xy'], (2, 2), 'mechanism.axis', nullable=True)
    _pixels(k['support_box_xyxy'], (4,), 'support', nullable=True)
    _pixels(g['reference_box_xyxy'], (4,), 'goal.reference', nullable=True)
    x, y = c['pixel_xy']; left, top, right, bottom = c['box_xyxy']
    if not left <= x <= right or not top <= y <= bottom:
        raise InteractionError('Contact pixel is outside its selected region')
    direction = np.asarray(k['initial_image_direction_xy'])
    if direction.shape != (2,) or not np.isin(direction, [-1, 0, 1]).all():
        raise InteractionError('Image direction must have two ternary components')
    roles = v['contact_roles']
    allowed = {'grasp_selected_part', 'fingers_on_local_moving_surface', 'object_on_selected_support'}
    if not isinstance(roles, list) or any(not isinstance(r, str) for r in roles) or len(set(roles)) != len(roles) or not set(roles) <= allowed or 'grasp_selected_part' not in roles:
        raise InteractionError('Invalid contact roles')
    if k['family'] == 'free':
        if k['motion'] != 'place' or k['axis_source'] != 'none' or g['relation'] not in {'on_support', 'unknown'}:
            raise InteractionError('Free-body placement choices conflict')
        if 'object_on_selected_support' not in roles:
            raise InteractionError('Placement needs selected support contact')
    elif k['family'] == 'prismatic':
        if k['axis_source'] != 'panel_normal' or k['motion'] not in {'open', 'close'}:
            raise InteractionError('Sliding choices conflict')
        if g['relation'] not in {'hand_clearance', 'aligned_with_reference_plane', 'unknown'}:
            raise InteractionError('Sliding goal must use a translational relation')
    elif k['family'] == 'revolute':
        if k['axis_source'] not in {'hinge_edge', 'control_face_normal'} or k['motion'] == 'place':
            raise InteractionError('Rotation choices conflict')
        if g['relation'] not in {'quarter_turn', 'aligned_with_reference_plane', 'unknown'}:
            raise InteractionError('Rotation goal must use an angular relation')
    if k['axis_source'] == 'hinge_edge' and k['axis_pixels_xy'] is None:
        raise InteractionError('Hinge edge needs visible endpoint references')
    if k['family'] in {'prismatic', 'revolute'} and k['support_box_xyxy'] is None:
        raise InteractionError('Articulation needs an explicit visible face patch')
    if c['shape'] == 'straight_bar' and c['axis_pixels_xy'] is None:
        raise InteractionError('Bar needs visible endpoint references')
    if k['axis_source'] == 'control_face_normal':
        if k['support_box_xyxy'] is None or k['turn_direction_view'] not in {'clockwise', 'counterclockwise'}:
            raise InteractionError('Rotary control needs support plane and turn direction')
    elif k['family'] not in {'free', 'unknown'} and not np.any(direction):
        raise InteractionError('Articulation needs an image motion direction')
    if g['relation'] in {'on_support', 'aligned_with_reference_plane'} and g['reference_box_xyxy'] is None:
        raise InteractionError('Relational goal needs a visible reference region')
    if k['motion'] == 'close' and g['relation'] not in {'aligned_with_reference_plane', 'unknown'}:
        raise InteractionError('Closing requires a measured reference or unknown')
    return v


def supported_selection(v):
    if any(v[part][field] == 'unknown' for part, field in [('contact', 'shape'), ('moving', 'shape'), ('mechanism', 'family'), ('mechanism', 'axis_source'), ('goal', 'relation')]):
        raise InteractionError('VLM reported unresolved interaction evidence')
    return v


def backend_for(v):
    """Selection is driven by predicted physical properties, never task names."""
    supported_selection(v)
    if v['mechanism']['family'] == 'free':
        return 'placement'
    if v['mechanism']['axis_source'] == 'hinge_edge':
        return 'hinge'
    return 'mechanism'
