import os
import pickle
import sys

def get_configstring(args):
    configstring = '{}-{}-{}-{}-{}-{}'.format(
                        args.data,
                        args.archer,
                        args.sampler,
                        args.comp,
                        args.dec,
                        args.critic,
                        args.id)
    return configstring

def get_hparamconfigstring(args):
    configstring = 'lr{}|{}|{}-f{}|{}-sd{}|{}-wd{}|{}-tau{}-{}'.format(
                        args.initial_lr_base,
                        args.initial_lr_enc,
                        args.initial_lr_critic,
                        args.decay_factor_base,
                        args.decay_factor_enc,
                        args.stop_decay_base,
                        args.stop_decay_enc,
                        args.wd_base,
                        args.wd_enc,
                        args.tau,
                        args.seed)
    return configstring

def save_args_to_checkpoint(args, path):
    # Load, if exists
    checkpoint = {}
    checkpoint.update(vars(args))
    # Save
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_checkpoint(path):
    try:
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
    except:
        raise FileNotFoundError
    return checkpoint

def save_checkpoint(checkpoint, path):
    # Reload checkpoint first to prevent (minimise chance of) overwriting
    current_checkpoint = load_checkpoint(path)
    for k, v in checkpoint.items():
        current_checkpoint[k] = v
        with open(path, 'wb') as f:
            pickle.dump(current_checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
