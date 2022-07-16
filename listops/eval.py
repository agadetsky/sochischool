import argparse
import itertools

import sys
sys.path.append('../')
sys.path.append('../../')

import numpy as np
import listops.data as _data
import listops.model as _model
import listops.utils as _utils
import listops.data_processing.python.loading as _loading
import os
import torch


parser = argparse.ArgumentParser()
# Configs
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--archer', type=str, required=True)
parser.add_argument('--sampler', type=str, required=True)
parser.add_argument('--comp', type=str, required=True)
parser.add_argument('--dec', type=str, required=True)
parser.add_argument('--critic', type=str, default='none')
parser.add_argument('--id', type=int, default=99)
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--reinforce', action='store_true')
parser.add_argument('--initial_lr_base', type=float, default=0.001)
parser.add_argument('--initial_lr_enc', type=float, default=0.001)
parser.add_argument('--wd_base', type=float, default=1e-5)
parser.add_argument('--wd_enc', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--batchsize', type=int, default=100)
# Eval specifics
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--last', action='store_true', default=False)
parser.add_argument('--keyword', type=str, default=None)
parser.add_argument('--early', action='store_true', default=False)
parser.add_argument('--dataset', type=str, default='val')
# Overwrite existing model + checkpoint

args = parser.parse_args()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda'
else:
    args.device = 'cpu'


def cycle(model, loader, device, args):
    cur_stats = {
        'loss' : _utils.AverageMeter(),
        'acc' : _utils.AverageMeter(),
        'iou' : _utils.AverageMeter(),
        'precision' : _utils.AverageMeter(),
        'recall' : _utils.AverageMeter(),
        'parse_acc' : _utils.AverageMeter(),
        'cliou' : _utils.AverageMeter(),
        'clprecision' : _utils.AverageMeter(),
        'clrecall' : _utils.AverageMeter(),
        'clparse_acc' : _utils.AverageMeter()
    }
    for batch_idx, (x, y, arcs, lengths, depths) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        arcs = arcs.to(device)
        lengths = lengths.to(device)
        bs = x.shape[0]
        with torch.no_grad():
            if not args.reinforce:
                pred_logits = model.forward(x, arcs, lengths)
            else:
                pred_logits, arc_logits, arb, arb_stat, z = _model.RModel.forward(model, x, arcs, lengths)

            loss = torch.nn.functional.cross_entropy(pred_logits, y)
            acc = (pred_logits.argmax(-1) == y).float().mean()
            sample = model.sample
            iou, cliou, precision, clprecision, recall, clrecall, parse_acc, clparse_acc = (
                compute_metrics(x, sample, arcs, lengths))

            bs = 1
            cur_stats['loss'].update(loss.item(), bs)
            cur_stats['acc'].update(acc.item(), bs)
            cur_stats['iou'].update(iou.item(), bs)
            cur_stats['cliou'].update(cliou.item(), bs)
            cur_stats['precision'].update(precision.item(), bs)
            cur_stats['clprecision'].update(clprecision.item(), bs)
            cur_stats['recall'].update(recall.item(), bs)
            cur_stats['clrecall'].update(clrecall.item(), bs)
            cur_stats['parse_acc'].update(parse_acc.item(), bs)
            cur_stats['clparse_acc'].update(clparse_acc.item(), bs)

    return cur_stats


def compute_metrics(x, sample, arcs, lengths):
    one = torch.tensor(1.0).cuda() if sample.is_cuda else torch.tensor(1.0)
    zero = torch.tensor(0.0).cuda() if sample.is_cuda else torch.tensor(0.0)
    # Compute true/false positives/negatives for metric calculations.
    maxlen = arcs.shape[-1]
    pad_tn = maxlen - lengths
    tp = torch.where(sample * arcs == 1.0, one, zero).sum((-1, -2))
    tn = torch.where(sample + arcs == 0.0, one, zero).sum((-1, -2)) - pad_tn
    fp = torch.where(sample - arcs == 1.0, one, zero).sum((-1, -2))
    fn = torch.where(sample - arcs == -1.0, one, zero).sum((-1, -2))

    # Calculate IoUs.
    iou = torch.mean((tp) / (tp + fp + fn)).cpu()
    # Calculate precision (attachment).
    precision = torch.mean(tp / (tp + fp)).cpu()
    # Calculate recall.
    recall = torch.mean(tp / (tp + fn)).cpu()
    # Calculate parse accuracy
    parse_acc = (sample == arcs).all(-1).all(-1).float().mean()

    # Clean computations
    # Compute clean attch_score which ignores "]" symbol (requires acces to x)
    close_ix = _loading.word_to_ix[']']
    clean_mask = (x != close_ix).unsqueeze(1).expand_as(arcs) # expands along 2nd dimension
    clean_mask = clean_mask & clean_mask.transpose(1, 2)

    cltp = torch.where((sample * arcs == 1.0) * clean_mask, one, zero).sum((-1, -2))
    cltn = torch.where((sample + arcs == 0.0) * clean_mask, one, zero).sum((-1, -2)) - pad_tn
    clfp = torch.where((sample - arcs == 1.0) * clean_mask, one, zero).sum((-1, -2))
    clfn = torch.where((sample - arcs == -1.0) * clean_mask, one, zero).sum((-1, -2))

    # Calculate IoUs.
    idx = (cltp + clfp + clfn) > 0
    clious = torch.ones_like(cltp)
    clious[idx] = cltp[idx] / (cltp + clfp + clfn)[idx]
    cliou = clious.mean().cpu()
    # Calculate precision (attachment).
    idx = (cltp + clfp) > 0
    clprecisions = torch.zeros_like(cltp)
    clprecisions[idx] = cltp[idx] / (cltp + clfp)[idx]
    clprecision = clprecisions.mean().cpu()
    # Calculate recall.
    idx = (cltp + clfn) > 0
    clrecalls = torch.ones_like(cltp)
    clrecalls[idx] = cltp[idx] / (cltp + clfn)[idx]
    clrecall = clrecalls.mean().cpu()
    # Calculate parse accuracy
    clparse_acc = (sample * clean_mask == arcs * clean_mask).all(-1).all(-1).float().mean()

    if torch.isnan(cliou) or torch.isnan(clprecision) or torch.isnan(clrecall) or torch.isnan(clparse_acc):
        print('Found NaN in cl metrics')

    return iou, cliou, precision, clprecision, recall, clrecall, parse_acc, clparse_acc


def eval_all():
    datasets = _data.get_datasets(args.data)
    train_loader, val_loader, test_loader  = _data.get_dataloaders(datasets, args.batchsize)


    folder = './listops/experiments'
    exps = os.listdir(path=folder)
    accs = {}
    precs = {}
    recs = {}

    for exp in exps:
#        if args.keyword in exp and 'seed' in exp and exp[-2:] == '42':
        if args.keyword in exp and 'seed' in exp:
#        if args.keyword in exp and 'seed' in exp and exp[-2:] != '42':
            experiment_folder = os.path.join(folder, exp)
            best_model = _model.get_model(args.archer, args.sampler, args.comp, args.dec, args.critic, args.tau, args.reinforce)
#            best_model.train()
            if args.early:
                if 'best_model_50.pth' in os.listdir(experiment_folder):
                    best_model.load_weights(
                        torch.load(os.path.join(experiment_folder, 'best_model_50.pth'))
                    )
                elif 'best_model50.pth' in os.listdir(experiment_folder):
                    best_model.load_weights(
                        torch.load(os.path.join(experiment_folder, 'best_model50.pth'))
                    )
                else:
                    continue
            else:
                if 'best_model_200.pth' in os.listdir(experiment_folder):
                    best_model.load_weights(
                        torch.load(os.path.join(experiment_folder, 'best_model_200.pth'))
                    )
                else:
                    continue

            best_model.to(args.device)
            #best_model.train()
            best_model.eval()

            if args.dataset == 'train':
                cur_stats = cycle(best_model, train_loader, args.device, args)
            elif args.dataset == 'val':
                cur_stats = cycle(best_model, val_loader, args.device, args)
            else:
                cur_stats = cycle(best_model, test_loader, args.device, args)
            accs[exp] = cur_stats['acc'].avg
            precs[exp] = cur_stats['clprecision'].avg
            recs[exp] = cur_stats['clrecall'].avg


    print(len(accs))
    top = None

    for key, value in accs.items():
        if top is None or value > accs[top]:
            top = key

    print('Top: {}, acc: {}'.format(top, accs[top]))

    accs = np.array(list(accs.values())) * 100
    precs = np.array(list(precs.values())) * 100
    recs = np.array(list(recs.values())) * 100

    print(len(accs))

    idx = np.argmax(accs)
    print('Accuracy: {:.4}+-{:.4}, Max: {:.4}'.format(accs.mean(), accs.std(), accs[idx]))
    print('Precision: {:.4}+-{:.4}, Max: {:.4}'.format(precs.mean(), precs.std(), precs[idx]))
    print('Recall: {:.4}+-{:.4}, Max: {:.4}'.format(recs.mean(), recs.std(), recs[idx]))


def eval_best():
    if args.reinforce:
        if args.estimator == 'relax':
            run_name = '{}_lrenc_{}_lrbase_{}_lrcritic_{}'.format(
                args.estimator,
                args.initial_lr_enc,
                args.initial_lr_base,
                args.initial_lr_critic
            )
        elif args.plus_samples > 0:
            run_name = '{}_lrenc_{}_lrbase_{}_samples_{}'.format(
                args.estimator,
                args.initial_lr_enc,
                args.initial_lr_base,
                args.plus_samples + 1
            )
        else:
            run_name = '{}_lrenc_{}_lrbase_{}'.format(
                args.estimator,
                args.initial_lr_enc,
                args.initial_lr_base
            )
    else:
        assert args.initial_lr_enc == args.initial_lr_base
        assert args.wd_enc == args.wd_base

        run_name = 'sst_lr_{}_tau_{}_wd_{}'.format(
            args.initial_lr_enc,
            args.tau,
            args.wd_enc,
        )

    experiment_folder = os.path.join('./listops/experiments', run_name)
    best_model = _model.get_model(args.archer, args.sampler, args.comp, args.dec, args.critic, args.tau, args.reinforce)
    best_model.load_weights(
        torch.load(os.path.join(experiment_folder, 'best_model.pth'))
    )
    best_model.to(args.device)
    best_model.eval()

    datasets = _data.get_datasets(args.data)
    train_loader, val_loader, test_loader  = _data.get_dataloaders(datasets, args.batchsize)

    loaders = [val_loader, test_loader]
    labels = ['val', 'test']

    stats = {}

    for label, loader in zip(labels, loaders):
        cur_stats = cycle(best_model, loader, args.device, args)
        cur_stats = {
            key : [value.avg] for key, value in cur_stats.items()
        }
        cur_stats['name'] = run_name

        stats[label] = cur_stats

    print('Val')
    print(stats['val'])
    print('Test')
    print(stats['test'])

    '''
    val_df = pd.DataFrame.from_dict(stats['val'])
    test_df = pd.DataFrame.from_dict(stats['test'])

    os.makedirs('metrics', exist_ok=True)
    val_df.to_csv('metrics/val_' + run_name + '.csv', index=False)
    test_df.to_csv('metrics/test_' + run_name + '.csv', index=False)
    '''

if __name__ == '__main__':
    #eval_best()
    eval_all()
