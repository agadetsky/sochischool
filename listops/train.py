import argparse
import itertools
import json
import os
import pickle
import time
import torch
import numpy as np
import sys
from functools import partial

sys.path.append('../')
sys.path.append('../../')
import listops.data as _data
import listops.data_processing.python.loading as _loading
import listops.model as _model
import listops.utils as _utils

from edmonds.estimators import reinforce, relax, reinforce_exp

import pandas as pd
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
# Configs
parser.add_argument('--data', type=str, required=True) #proc0
parser.add_argument('--archer', type=str, required=True)
parser.add_argument('--sampler', type=str, required=True)
parser.add_argument('--comp', type=str, required=True)
parser.add_argument('--dec', type=str, required=True)
parser.add_argument('--critic', type=str, default='none')
parser.add_argument('--id', type=int, default=99)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', type=eval, default=True, choices=[True, False])
# Training Parameters
parser.add_argument('--optimizer_base', type=str, default="adam",
                    choices=["sgd", "adam", "adamw"])
parser.add_argument('--optimizer_enc', type=str, default="adam",
                    choices=["sgd", "adam", "adamw"])
parser.add_argument('--optimizer_critic', type=str, default="adam",
                    choices=["sgd", "adam", "adamw"])
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--initial_lr_base', type=float, default=0.001)
parser.add_argument('--initial_lr_enc', type=float, default=0.001)
parser.add_argument('--initial_lr_critic', type=float, default=0.001)
parser.add_argument('--decay_factor_base', type=float, default=1.0)
parser.add_argument('--decay_factor_enc', type=float, default=1.0)
parser.add_argument('--decay_factor_critic', type=float, default=1.0)
parser.add_argument('--stop_decay_base', type=int, default=10) # decay for this many epochs
parser.add_argument('--stop_decay_enc', type=int, default=10) # decay for this many epochs
parser.add_argument('--stop_decay_critic', type=int, default=10) # decay for this many epochs
parser.add_argument('--momentum_base', type=float, default=0.0)
parser.add_argument('--momentum_enc', type=float, default=0.0)
parser.add_argument('--momentum_critic', type=float, default=0.0)
parser.add_argument('--wd_base', type=float, default=1e-5)
parser.add_argument('--wd_enc', type=float, default=1e-5)
parser.add_argument('--wd_critic', type=float, default=1e-5)
parser.add_argument('--tau', type=float, default=1.0)
# Logging related parameters.
parser.add_argument('--eval_every', type=int, default=None,
                    help='Number of training steps in-between evaluating.')
parser.add_argument('--add_timestamp', type=eval, default=True, choices=[True, False])
parser.add_argument('--experiments_folder', type=str, default=None)
parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument('--checkpoint_dir_load', type=str, default=None)
parser.add_argument('--checkpoint_dir_save', type=str, default=None)

parser.add_argument('--max_iter', type=int, default=-1)
parser.add_argument('--max_eval', type=int, default=-1)
parser.add_argument('--reinforce', action='store_true')
parser.add_argument('--plus_samples', type=int, default=0)
parser.add_argument('--mean_plus', action='store_true')
parser.add_argument('--fit_on_batch', action='store_true')
parser.add_argument('--estimator', type=str, default='reinforce')
parser.add_argument('--eval_only', action='store_true')

args = parser.parse_args()
print(args)


# Set-up
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and args.cuda:
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda'
else:
    args.device = 'cpu'

# Set up experiments folder.
os.makedirs('./listops/experiments', exist_ok=True)

'''
experiments_folder = args.experiments_folder
if experiments_folder is None:
    experiments_folder = _utils.get_configstring(args)
    if args.add_timestamp:
        experiments_folder += f'_{time.strftime("%Y%m%d")}'
os.makedirs(os.path.join('./listops/experiments', experiments_folder), exist_ok=True)
'''


if args.reinforce:
    if args.estimator == 'relax':
        assert args.initial_lr_enc == args.initial_lr_base
        assert args.wd_enc == args.wd_base
        run_name = '{}_lrb_{}_lrc_{}_wdb_{}_wdc_{}_seed_{}'.format(
            args.estimator,
            args.initial_lr_base,
            args.initial_lr_critic,
            args.wd_base,
            args.wd_critic,
            args.seed
        )
    elif args.plus_samples > 0:
        assert args.initial_lr_enc == args.initial_lr_base
        assert args.wd_enc == args.wd_base
        run_name = '{}_lr_{}_samples_{}_wd_{}_seed_{}'.format(
            args.estimator,
            args.initial_lr_enc,
            args.plus_samples + 1,
            args.wd_enc,
            args.seed
        )
    else:
        run_name = '{}_lrenc_{}_lrbase_{}_seed_{}'.format(
            args.estimator,
            args.initial_lr_enc,
            args.initial_lr_base,
            args.seed
        )
else:
    assert args.initial_lr_enc == args.initial_lr_base
    assert args.wd_enc == args.wd_base

    run_name = 'sst_lr_{}_tau_{}_wd_{}_seed_{}'.format(
        args.initial_lr_enc,
        args.tau,
        args.wd_enc,
        args.seed
    )


# Set up the experiment (specific hparams) folder.
#experiment_name = _utils.get_hparamconfigstring(args)
experiment_name = run_name
experiment_folder = os.path.join(
    './listops/experiments', experiment_name)
os.makedirs(experiment_folder, exist_ok=True)

# Save args in experiment folder.
with open(os.path.join(experiment_folder, 'train_config.json'), 'w') as f:
    config = {k: v for (k, v) in vars(args).items()}
    json.dump(config, f, indent=2)

# Data
datasets = _data.get_datasets(args.data)
train_loader, val_loader, test_loader  = _data.get_dataloaders(datasets, args.batchsize)

num_train = len(train_loader.dataset)
num_complete_batches, leftover = divmod(num_train, args.batchsize)
num_batches_per_epoch = num_complete_batches  # Drop the last incomplete batch.
total_num_steps = args.num_epochs * num_batches_per_epoch
# Eval every epoch if not specified.
args.eval_every = num_batches_per_epoch if args.eval_every is None else args.eval_every

# Model
estimator = None
if args.estimator == 'reinforce':
    estimator = reinforce
elif args.estimator == 'relax':
    estimator = relax
elif args.estimator == 'reinforce_exp':
    estimator = reinforce_exp

model = _model.get_model(args.archer, args.sampler, args.comp, args.dec, args.critic, args.tau, args.reinforce)
model.to(args.device)


# optimizer + LR scheduler
if args.optimizer_base == "sgd":
    opt_base = partial(torch.optim.SGD, momentum=args.momentum_base)
elif args.optimizer_base == "adam":
    opt_base = torch.optim.Adam
elif args.optimizer_base == "adamw":
    opt_base = partial(torch.optim.AdamW, weight_decay=args.wd_base)
else:
    raise NotImplementedError

optimizer_base = opt_base(model.base.parameters(), lr=args.initial_lr_base)

if args.optimizer_enc == "sgd":
    opt_enc = partial(torch.optim.SGD, momentum=args.momentum_enc)
elif args.optimizer_enc == "adam":
    opt_enc = torch.optim.Adam
elif args.optimizer_enc == "adamw":
    opt_enc = partial(torch.optim.AdamW, weight_decay=args.wd_enc)
else:
    raise NotImplementedError

optimizer_enc = opt_enc(model.archer.parameters(), lr=args.initial_lr_enc)



if model.critic is not None:
    if args.optimizer_critic == "sgd":
        opt_critic = partial(torch.optim.SGD, momentum=args.momentum_critic)
    elif args.optimizer_critic == "adam":
        opt_critic = torch.optim.Adam
    elif args.optimizer_critic == "adamw":
        opt_critic = partial(torch.optim.AdamW, weight_decay=args.wd_critic)
    else:
        raise NotImplementedError

    optimizer_critic = opt_critic(model.critic.parameters(), lr=args.initial_lr_critic)
else:
    optimizer_critic = None


# This is for checkpointing in Vaughan 2 cluster.
checkpoint_data = None
if args.checkpoint_dir_load:
    try:
        print('checkpoints/' + args.checkpoint_dir_load + '/checkpoint.pt')
        checkpoint_data = torch.load('checkpoints/' + args.checkpoint_dir_load + '/checkpoint.pt')
        model.load_weights(checkpoint_data['model_state_dict'])
        optimizer_base.load_state_dict(checkpoint_data['optimizer_base_state_dict'])
        optimizer_enc.load_state_dict(checkpoint_data['optimizer_enc_state_dict'])
        if model.critic is not None:
            optimizer_critic.load_state_dict(checkpoint_data['optimizer_critic_state_dict'])
    except:
        os.makedirs('checkpoints/' + args.checkpoint_dir_load, exist_ok=True)
        checkpoint_data = None

if not args.eval_only:
    writer = SummaryWriter('runs/' + run_name)


#@profile
def opt_step(model, optimizer_base, optimizer_enc, optimizer_critic, x, y, arcs, lengths, reinforce=False):
    critic_loss_item = None
    if not reinforce:
        with torch.set_grad_enabled(True):
            pred_logits = _model.Model.forward(model, x, arcs, lengths)
            loss = torch.nn.functional.cross_entropy(pred_logits, y)
            loss.backward()
            optimizer_base.step()
            optimizer_enc.step()

            acc = (pred_logits.argmax(-1) == y).float().mean()
    else:
        with torch.set_grad_enabled(True):
            if args.plus_samples == 0:
                pred_logits, arc_logits, arb, arb_stat, z = _model.RModel.forward(model, x, arcs, lengths)
                loss = torch.nn.functional.cross_entropy(pred_logits, y, reduction='none')
                v = torch.distributions.utils.clamp_probs(torch.rand_like(z))
                d_logits = estimator(fb=loss, b=(arb, arb_stat), logits=arc_logits,
                    lengths=lengths, z=z, c=model.critic, v=v, x=x, lengths_x=lengths)

                if model.critic is not None:
                    critic_loss = (d_logits ** 2).mean(dim=0).sum()
                    critic_loss.backward()
                    critic_loss_item = critic_loss.item()

                arc_logits.backward(d_logits.squeeze())
                loss = loss.mean()
                loss.backward()
                optimizer_base.step()
                optimizer_enc.step()

                if model.critic is not None:
                    optimizer_critic.step()

                acc = (pred_logits.argmax(-1) == y).float().mean()
            elif not args.mean_plus:
                pred_logits, arc_logits, arb, arb_stat, z, pred_logits_K = _model.RModel.forward_plus(model, x, arcs, lengths, K=args.plus_samples)
                loss = torch.nn.functional.cross_entropy(pred_logits, y, reduction='none')
                loss_K = torch.nn.functional.cross_entropy(pred_logits_K, y.repeat(args.plus_samples), reduction='none').view(args.plus_samples, -1)
                loss_K = loss_K.detach()

                v = torch.distributions.utils.clamp_probs(torch.rand_like(z))
                d_logits = estimator(fb=loss - loss_K.mean(0), b=(arb, arb_stat), logits=arc_logits,
                    lengths=lengths, z=z, c=model.critic, v=v)

                if model.critic is not None:
                    (d_logits ** 2).mean(dim=0).sum().backward()

                arc_logits.backward(d_logits.squeeze())
                loss = loss.mean()
                loss.backward()
                optimizer_base.step()
                optimizer_enc.step()

                acc = (pred_logits.argmax(-1) == y).float().mean()
            else:
                pred_logits, arc_logits, arc_logits_all, arb, arb_stat, z = _model.RModel.forward_plus_mean(model, x, arcs, lengths, K=args.plus_samples)
                y_all = y.repeat(args.plus_samples + 1)
                loss_all = torch.nn.functional.cross_entropy(pred_logits, y_all, reduction='none')
                loss_all_detached = loss_all.detach().view(args.plus_samples + 1, -1)
                lengths_all = lengths.repeat(args.plus_samples + 1)

                baselines = torch.zeros_like(loss_all_detached)
                for i in range(loss_all_detached.shape[0]):
                    mask = torch.cat((torch.arange(i), torch.arange(i + 1, loss_all_detached.shape[0])))
                    baselines[i] = loss_all_detached[mask].mean(dim=0)
                baselines = baselines.detach()

                v = torch.distributions.utils.clamp_probs(torch.rand_like(z))
                d_logits_all = estimator(fb=loss_all - baselines.view(-1), b=(arb, arb_stat), logits=arc_logits_all,
                    lengths=lengths_all, z=z, c=model.critic, v=v)

                d_logits = d_logits_all.view(args.plus_samples + 1, -1, d_logits_all.shape[1], d_logits_all.shape[2]).mean(0)

                if model.critic is not None:
                    (d_logits ** 2).mean(dim=0).sum().backward()

                arc_logits.backward(d_logits)
                loss = loss_all.mean()
                loss.backward()
                optimizer_base.step()
                optimizer_enc.step()
                acc = (pred_logits.argmax(-1) == y_all).float().mean()

    return loss, acc, critic_loss_item


#@profile
def train():
    # Start from valid staring point if checkpoint_dir is given.
    start_epoch = checkpoint_data['epoch'] if checkpoint_data else 0
    start_step = checkpoint_data['step'] if checkpoint_data else 0
    itercount = itertools.count(start_step)

    best_acc = checkpoint_data['best_acc'] if checkpoint_data else -np.inf

    measurements = (
        checkpoint_data['measurements'] if checkpoint_data else {
            'learning_rates_base': [],
            'learning_rates_enc' : [],
            'train_steps': [], 'loss_train': [], 'acc_train': [],
            'val_steps': [], 'loss_val': [], 'acc_val': [],
            'precision_val': [], 'recall_val': [], 'iou_val': [], 'parse_acc_val': [],
            'clprecision_val': [], 'clrecall_val': [], 'cliou_val': [], 'clparse_acc_val': [],
    })
    model.train()
    start_time = time.time()

    if args.fit_on_batch:
        x_, y_, arcs_, lengths_, depths_ = next(iter(train_loader))
        x_ = x_[:2]
        y_ = y_[:2]
        arcs_ = arcs_[:2]
        lengths_ = lengths_[:2]


    sum_update_time = 0
    n_updates = 0

    for epoch in range(start_epoch, args.num_epochs):
        print('epoch')
        print(epoch)
        for batch_idx, (x, y, arcs, lengths, depths) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print('batch')
                print(batch_idx)
            if args.fit_on_batch:
                x, y, arcs, lengths, depths = x_, y_, arcs_, lengths_, depths_

            i = next(itercount)
            if args.max_iter > 0 and i > args.max_iter:
                break
            x = x.to(args.device)
            y = y.to(args.device)
            arcs = arcs.to(args.device)
            lengths = lengths.to(args.device)

            optimizer_base.zero_grad()
            optimizer_enc.zero_grad()
            if optimizer_critic:
                optimizer_critic.zero_grad()

            bs = x.shape[0]

            update_time = time.perf_counter()
            loss, acc, critic_loss = opt_step(model, optimizer_base, optimizer_enc, optimizer_critic, x, y, arcs, lengths, args.reinforce)
            update_time = time.perf_counter() - update_time
            sum_update_time += update_time
            n_updates += 1


            # Take measurements.
            measurements["learning_rates_base"].append(get_curr_learning_rate(epoch, args, 'base'))
            measurements["learning_rates_enc"].append(get_curr_learning_rate(epoch, args, 'enc'))
            measurements["train_steps"].append(i)
            measurements["loss_train"].append(loss.item())
            measurements["acc_train"].append(acc.item())

            writer.add_scalar('train/loss', loss.item(), i)
            writer.add_scalar('train/acc', acc.item(), i)

            if critic_loss:
                writer.add_scalar('train/critic_loss', critic_loss, i)

            # Evaluate on validation set every 'eval_every' steps.
#            if False:
            if not args.fit_on_batch and i % args.eval_every == 0:
                train_time = time.time() - start_time
                start_time = time.time()
                eval_start_time = time.time()

                measurements["val_steps"].append(i)
                losses, accs, ious, precisions, recalls, parse_accs = [], [], [], [], [], []
                clious, clprecisions, clrecalls, clparse_accs = [], [], [], []

                model.eval()
                eval_i = 0
                for batch_idx, (x, y, arcs, lengths, depths) in enumerate(val_loader):
                    if args.max_eval > 0 and eval_i > args.max_eval:
                        break

                    x = x.to(args.device)
                    y = y.to(args.device)
                    arcs = arcs.to(args.device)
                    lengths = lengths.to(args.device)
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

                        losses.append(loss.item())
                        accs.append(acc.item())
                        ious.append(iou.item())
                        clious.append(cliou.item())
                        precisions.append(precision.item())
                        clprecisions.append(clprecision.item())
                        recalls.append(recall.item())
                        clrecalls.append(clrecall.item())
                        parse_accs.append(parse_acc.item())
                        clparse_accs.append(clparse_acc.item())
                    eval_i += 1

                # Log eval measurements.
                measurements["loss_val"].append(np.mean(losses))
                measurements["acc_val"].append(np.mean(accs))
                measurements["iou_val"].append(np.mean(ious))
                measurements["cliou_val"].append(np.mean(clious))
                measurements["precision_val"].append(np.mean(precisions))
                measurements["clprecision_val"].append(np.mean(clprecisions))
                measurements["recall_val"].append(np.mean(recalls))
                measurements["clrecall_val"].append(np.mean(clrecalls))
                measurements["parse_acc_val"].append(np.mean(parse_accs))
                measurements["clparse_acc_val"].append(np.mean(clparse_accs))

                writer.add_scalar('val/loss', measurements["loss_val"][-1], i)
                writer.add_scalar('val/acc', measurements["acc_val"][-1], i)
                writer.add_scalar('val/iou', measurements["iou_val"][-1], i)
                writer.add_scalar('val/cliou', measurements["cliou_val"][-1], i)
                writer.add_scalar('val/precision', measurements["precision_val"][-1], i)
                writer.add_scalar('val/clprecision', measurements["clprecision_val"][-1], i)
                writer.add_scalar('val/recall', measurements["recall_val"][-1], i)
                writer.add_scalar('val/clrecall', measurements["clrecall_val"][-1], i)
                writer.add_scalar('val/parse_acc', measurements["parse_acc_val"][-1], i)
                writer.add_scalar('val/clparse_acc', measurements["clparse_acc_val"][-1], i)

                eval_time = time.time() - eval_start_time
                print(
                    "{}/{} iterations in {:0.2f}s; ".format(
                        i, total_num_steps, train_time) +
                    "Eval in {:0.2f} sec".format(eval_time), flush=True)
                print(
                    'Iteration %s (Epoch %s) '%(i, epoch) +
                    'loss_train: {:.5f} '.format(measurements["loss_train"][-1]) +
                    'acc_train: {:.5f} '.format(measurements["acc_train"][-1]) +
                    'loss_val: {:.5f} '.format(measurements["loss_val"][-1]) +
                    'acc_val: {:.5f} '.format(measurements["acc_val"][-1]) +
                    'iou_val: {:.5f} '.format(measurements["iou_val"][-1]) +
                    'precision_val: {:.5f} '.format(measurements["precision_val"][-1]) +
                    'recall_val: {:.5f} '.format(measurements["recall_val"][-1]) +
                    'parse_acc_val: {:.5f} '.format(measurements["parse_acc_val"][-1]) +
                    'cl_iou_val: {:.5f} '.format(measurements["cliou_val"][-1]) +
                    'cl_precision_val: {:.5f} '.format(measurements["clprecision_val"][-1]) +
                    'cl_recall_val: {:.5f} '.format(measurements["clrecall_val"][-1]) +
                    'cl_parse_acc_val: {:.5f} '.format(measurements["clparse_acc_val"][-1])
                )
                if measurements["acc_val"][-1] > best_acc:
                    best_acc = measurements["acc_val"][-1]
                    model.save_weights(
                        os.path.join(experiment_folder, 'best_model_{}.pth'.format(args.num_epochs)))

                model.train()

        # New Learning rate
        adjust_learning_rate(optimizer_base, epoch, args, 'base')
        adjust_learning_rate(optimizer_enc, epoch, args, 'enc')
        if optimizer_critic:
            adjust_learning_rate(optimizer_critic, epoch, args, 'critic')

        # Optionally save checkpoint.
        args.checkpoint_dir_save = run_name + '_{}_epoch'.format(args.num_epochs)
        print('kekb')
        if args.checkpoint_dir_save:
            print('sesb')
            os.makedirs('checkpoints/' + args.checkpoint_dir_save, exist_ok=True)
            print(f"Saving checkpoint for epoch {epoch}.")

            if optimizer_critic:
                torch.save({
                    "epoch": epoch + 1, "step": i + 1,
                    "best_acc": best_acc,
                    "measurements": measurements,
                    "model_state_dict": model.state_dict(),
                    "optimizer_base_state_dict": optimizer_base.state_dict(),
                    "optimizer_enc_state_dict" : optimizer_enc.state_dict(),
                    "optimizer_critic_state_dict" : optimizer_critic.state_dict(),
                }, os.path.join('checkpoints/' + args.checkpoint_dir_save, "checkpoint.pt"))
            else:
                torch.save({
                    "epoch": epoch + 1, "step": i + 1,
                    "best_acc": best_acc,
                    "measurements": measurements,
                    "model_state_dict": model.state_dict(),
                    "optimizer_base_state_dict": optimizer_base.state_dict(),
                    "optimizer_enc_state_dict" : optimizer_enc.state_dict(),
                }, os.path.join('checkpoints/' + args.checkpoint_dir_save, "checkpoint.pt"))


    mean_update_time = sum_update_time / n_updates
    print('N updates:', n_updates)
    print('Mean update time:', mean_update_time)

    return measurements, best_acc


def eval_best():
    best_model = _model.get_model(args.archer, args.sampler, args.comp, args.dec, args.critic, args.tau, args.reinforce)
    best_model.load_weights(
        torch.load(os.path.join(experiment_folder, 'best_model_{}.pth'.format(args.num_epochs)))
    )
    best_model.to(args.device)
    best_model.eval()

    loaders = [val_loader, test_loader]
    labels = ['val', 'test']

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


    stats = {}

    for label, loader in zip(labels, loaders):
        cur_stats = cycle(best_model, loader, args.device, args)
        cur_stats = {
            key : [value.avg] for key, value in cur_stats.items()
        }
        cur_stats['name'] = run_name

        stats[label] = cur_stats

    val_df = pd.DataFrame.from_dict(stats['val'])
    test_df = pd.DataFrame.from_dict(stats['test'])

    os.makedirs('metrics', exist_ok=True)
    val_df.to_csv('metrics/val_' + run_name + '.csv', index=False)
    test_df.to_csv('metrics/test_' + run_name + '.csv', index=False)


def main():
    if not args.eval_only:
        '''
        try:
            measurements, best_acc = train()
            # Save measurements.
            with open(os.path.join(experiment_folder, "train_and_val_measurements.pkl"), "wb") as f:
                pickle.dump(measurements, f)
        except:
            print('Caught an exception. Evaluating best model...')

        '''

        measurements, best_acc = train()
        # Save measurements.
        with open(os.path.join(experiment_folder, "train_and_val_measurements.pkl"), "wb") as f:
            pickle.dump(measurements, f)
        model.save_weights(os.path.join(experiment_folder, 'last_model.pth'))

    with torch.no_grad():
        eval_best()


def get_curr_learning_rate(epoch, args, model):
    if model == 'base':
        final_lr = args.initial_lr_base * args.decay_factor_base
        diff = args.initial_lr_base - final_lr
        lr = args.initial_lr_base - diff * (epoch / args.stop_decay_base)
        lr = max(lr, final_lr)
    elif model == 'enc':
        final_lr = args.initial_lr_enc * args.decay_factor_enc
        diff = args.initial_lr_enc - final_lr
        lr = args.initial_lr_enc - diff * (epoch / args.stop_decay_enc)
        lr = max(lr, final_lr)
    elif model == 'critic':
        final_lr = args.initial_lr_critic * args.decay_factor_critic
        diff = args.initial_lr_critic - final_lr
        lr = args.initial_lr_critic - diff * (epoch / args.stop_decay_critic)
        lr = max(lr, final_lr)
    return lr


def adjust_learning_rate(optimizer, epoch, args, model):
    """Implements a linear learning rate schedule.
    The learning rate is annealed linearly to final_lr over the course
    of num=stop_decay epochs"""
    lr = get_curr_learning_rate(epoch, args, model)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            break

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
if __name__ == '__main__':
    main()
