import argparse
from vlm_tamp.cli import load_credentials
from vlm_tamp.task_grounding_scope import run

parser=argparse.ArgumentParser(description='Ordinary pick-place without extra path constraints; middle-drawer pull constraints')
parser.add_argument('--config',default='configs/task_grounding_scope.json')
parser.add_argument('--output',required=True)
parser.add_argument('--env-file')
parser.add_argument('--allow-paid',action='store_true')
parser.add_argument('--dry-run',action='store_true')
args=parser.parse_args()
if args.env_file and not args.dry_run:
    load_credentials(args.env_file)
manifest=run(args.config,args.output,allow_paid=args.allow_paid,dry_run=args.dry_run)
print(manifest['ledger'])
