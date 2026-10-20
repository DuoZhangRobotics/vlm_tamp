import argparse

from vlm_tamp.rgbd_numeric_trial import prepare


parser = argparse.ArgumentParser(description="Capture aligned depth and freeze numerical blanks pilot")
parser.add_argument("--config", default="configs/rgbd_numeric_trial.json")
args = parser.parse_args()
protocol = prepare(args.config)
print({"captures": len(protocol["captures"]), "cases": len(protocol["requests"]),
       "preserved_files": len(protocol["preserved_artifacts"])})
