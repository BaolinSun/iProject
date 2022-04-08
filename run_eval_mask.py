import argparse

from config import cfg
from utils import run_eval_miou, run_eval_miou_simple

parser = argparse.ArgumentParser()
parser.add_argument('--gt-source', type=str, default=None)
parser.add_argument('--mask-output', type=str, default=None)
args = parser.parse_args()

if args.gt_source is not None and args.mask_output is not None:
    run_eval_miou_simple(args.gt_source, args.mask_output)
else:
    run_eval_miou(cfg.output)

print('success...')