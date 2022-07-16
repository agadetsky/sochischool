import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir_save', type=str, default=None)
args = parser.parse_args()

args.checkpoint_dir_save = 'kek'
print(args.checkpoint_dir_save)