import argparse
from vlm_tamp.drawer_plane_trial import run

parser=argparse.ArgumentParser(description='Offline local RGB-D plane comparison using unchanged saved Luna-high seeds')
parser.add_argument('--config',default='configs/drawer_plane_ablation.json')
parser.add_argument('--output',default='results/drawer-plane-ablation')
args=parser.parse_args()
print(run(args.config,args.output))
