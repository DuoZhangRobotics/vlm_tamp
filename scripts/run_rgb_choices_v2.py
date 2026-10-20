import argparse
import json

from vlm_tamp.cli import load_credentials
from vlm_tamp.rgb_choice_v2_trial import run_choice_trial


parser = argparse.ArgumentParser(description="RGB-derived visual choices and final-state blanks")
parser.add_argument("--dataset", default="data/rgb-choices-v2")
parser.add_argument("--config", default="configs/rgb_choices_v2_dev.json")
parser.add_argument("--output", required=True)
parser.add_argument("--allow-paid", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--env-file")
args = parser.parse_args()
if args.env_file:
    load_credentials(args.env_file)
manifest = run_choice_trial(args.dataset, args.config, args.output, args.allow_paid, args.dry_run)
print(
    json.dumps(
        {"model": manifest["model"], "ledger": manifest.get("ledger"), "inventory_provided": False}, indent=2
    )
)
