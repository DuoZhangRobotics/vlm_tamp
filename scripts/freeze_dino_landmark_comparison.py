from vlm_tamp.rgbd_landmark_trial import freeze


protocol = freeze('configs/rgbd_dino_comparison.json')
print({'frozen_requests': len(protocol['requests']), 'model_scene_pairs': len(protocol['offline_proposal_coverage']),
       'preserved_files': len(protocol['preserved_artifacts'])})
