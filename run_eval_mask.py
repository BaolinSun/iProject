import argparse

from utils import run_eval_miou

parser = argparse.ArgumentParser()
parser.add_argument('gt_source', type=str)
parser.add_argument('mask_output', type=str)
args = parser.parse_args()

run_eval_miou(args.gt_source, args.mask_output)

print('success...')