import argparse
from pathlib import Path

from vlm_tamp.robopoint_baseline import prepare_protocol, run


parser = argparse.ArgumentParser(description="Released RoboPoint on the fixed RGB role benchmark")
parser.add_argument("--protocol", default="configs/frozen_robopoint_native_cap.json")
parser.add_argument("--checkpoint")
parser.add_argument("--output", default="results/robopoint-native-cap")
parser.add_argument("--prepare-only", action="store_true")
args = parser.parse_args()
if args.prepare_only:
    protocol = prepare_protocol(args.protocol)
    print("Frozen", len(protocol["requests"]), "tasks before local model inference")
else:
    checkpoint = args.checkpoint or Path("results/external-baseline-audit/robopoint-checkpoint.txt").read_text().strip()
    run(args.protocol, checkpoint, args.output)
