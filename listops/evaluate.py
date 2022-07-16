import argparse
import itertools
import listops.data as _data
import listops.model as _model
import listops.utils as _utils
import os
import torch
from listops.train import cycle

parser = argparse.ArgumentParser()
# Configs
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--archer', type=str, required=True)
parser.add_argument('--sampler', type=str, required=True)
parser.add_argument('--comp', type=str, required=True)
parser.add_argument('--dec', type=str, required=True)
parser.add_argument('--id', type=int, default=99)
# Eval specifics
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--last', action='store_true', default=False)
# Overwrite existing model + checkpoint
parser.add_argument('--overwrite', action='store_true', default=False)
parser.add_argument('--no_cuda', action='store_true', default=False)

def main():
    # Set-up
    args = parser.parse_args()
    configstring = _utils.get_configstring(args)
    chp_dir = './listops/checkpoints'
    model_dir = './listops/models'
    args.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    midfix = 'best' if not args.last else 'last'
    chp_path = os.path.join(chp_dir, '{}-{}.pkl'.format(configstring, midfix))
    chp = _utils.load_checkpoint(chp_path)
    model_path = os.path.join(model_dir, '{}-{}.pth'.format(configstring, midfix))

    # Eval_labels =
    eval_exists = 'test_loss' in chp.keys()
    if eval_exists and not args.overwrite:
        return # Exit immediately!

    # Data
    datasets = _data.get_datasets(args.data)
    loaders  = _data.get_dataloaders(datasets, 500)
    labels = ['train', 'val', 'test']
    # Model
    model = _model.get_model(args.archer, args.sampler, args.comp, args.dec, 1.0) # tau is inconsequential for evaluation
    model.load_weights(model_path)
    model.to(args.device)

    # Eval Logic
    for label, loader in zip(labels, loaders):
        losses = _utils.AverageMeter()
        accs = _utils.AverageMeter() #
        attchs = _utils.AverageMeter()
        parse_accs = _utils.AverageMeter()
        for _ in range(args.num_samples):
            loss, acc, attch, parse_acc = cycle(model, loader, args.device)
            losses.update(loss)
            accs.update(acc)
            attchs.update(attch)
            parse_accs.update(parse_acc)
        print('{}: \t Loss {:.3f} \t Acc {:.3f} \t Attch {:.3f} \t Parse Acc {:.3f}'.format(
                label, losses.avg, accs.avg, attchs.avg, parse_accs.avg))
        chp['{}-loss'.format(label)] = losses.avg
        chp['{}-accs'.format(label)] = accs.avg
        chp['{}-attchs'.format(label)] = attchs.avg
        chp['{}-parse_accs'.format(label)] = parse_accs.avg

    _utils.save_checkpoint(chp, chp_path)

if __name__ == '__main__':
    main()
