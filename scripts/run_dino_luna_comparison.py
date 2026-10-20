import argparse

from vlm_tamp.cli import load_credentials
from vlm_tamp.rgbd_luna_landmark_trial import run


parser = argparse.ArgumentParser(description='GPT-6 Luna on matched DINOv2/v3 landmarks with the authorized two-dollar ceiling')
parser.add_argument('--config', default='configs/rgbd_dino_luna_comparison.json')
parser.add_argument('--output', required=True)
parser.add_argument('--allow-paid', action='store_true')
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--env-file')
args = parser.parse_args()
if args.env_file and not args.dry_run:
    load_credentials(args.env_file)
manifest = run(args.config, args.output, allow_paid=args.allow_paid, dry_run=args.dry_run)
print(manifest['ledger'])
