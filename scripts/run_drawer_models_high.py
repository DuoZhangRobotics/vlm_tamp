import argparse
from vlm_tamp.cli import load_credentials
from vlm_tamp.drawer_model_comparison import run

parser=argparse.ArgumentParser(description='Matched Luna/Astra high-reasoning drawer selection with one shared budget')
parser.add_argument('--config',default='configs/drawer_models_high.json')
parser.add_argument('--output',required=True)
parser.add_argument('--env-file')
parser.add_argument('--allow-paid',action='store_true')
parser.add_argument('--dry-run',action='store_true')
args=parser.parse_args()
if args.env_file and not args.dry_run:
    load_credentials(args.env_file)
manifest=run(args.config,args.output,allow_paid=args.allow_paid,dry_run=args.dry_run)
print(manifest['ledger'])
