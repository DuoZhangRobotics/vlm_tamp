from vlm_tamp.rgbd_landmark_trial import freeze


protocol = freeze('configs/rgbd_dinov2_choices_trial.json')
print({'frozen_requests': len(protocol['requests']), 'unique_scenes': len(protocol['offline_proposal_coverage']),
       'preserved_files': len(protocol['preserved_artifacts'])})
