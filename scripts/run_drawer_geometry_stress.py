import argparse
from vlm_tamp.cli import load_credentials
from vlm_tamp.drawer_geometry_stress import run

parser=argparse.ArgumentParser(description='Two-point Luna-high grounding on changed views and partial-open rotated drawers')
parser.add_argument('--config',default='configs/drawer_geometry_stress.json')
parser.add_argument('--output',default='results/drawer-geometry-stress-luna')
parser.add_argument('--env-file')
parser.add_argument('--allow-paid',action='store_true')
parser.add_argument('--dry-run',action='store_true')
args=parser.parse_args()
if args.env_file and not args.dry_run:load_credentials(args.env_file)
print(run(args.config,args.output,allow_paid=args.allow_paid,dry_run=args.dry_run)['ledger'])
