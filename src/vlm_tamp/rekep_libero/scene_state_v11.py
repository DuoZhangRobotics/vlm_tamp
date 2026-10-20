"""Observed geometry with local source/destination-support pickup contacts."""

from .contact_policy_v11 import ContactPolicy
from .scene_state_v10 import ObservedScene as SceneV10


class ObservedScene(SceneV10):
    def __init__(self, cameras, metadata, spec, source_cloud, source_mask, main_camera,
                 parent_mask=None, target_cloud=None, config=None):
        super().__init__(cameras, metadata, spec, source_cloud, source_mask, main_camera,
                         parent_mask, target_cloud, config)
        self.policy = ContactPolicy(metadata['robot_collision_volumes_world'], spec, self.config)

    def record(self):
        result = super().record()
        result.update(schema_version='observed-scene-11.0', contact_policy_version='11.0')
        return result
