import os
import sys


def main():
    numtrain = 80000
    numval = 10000
    minlen = 10
    maxlen = 50
    seed = 0

    # Depth 1 (Generate lots)
    for nosm in [False, True]:
        folder_name = get_folder_name(1, maxlen, nosm)
        args = make_args(seed, 1, 200000, numval, 2, maxlen, folder_name)
        cmd = make_cmd(args, nosm)
        os.system(cmd)
        seed += 1

    # Depth 2
    for depth in range(2, 6):
        for nosm in [False, True]:
            folder_name = get_folder_name(depth, maxlen, nosm)
            args = make_args(seed, depth, numtrain, numval, minlen, maxlen, folder_name)
            cmd = make_cmd(args, nosm)
            os.system(cmd)
            seed += 1


def make_args(seed, depth, numtrain, numval, minlen, maxlen, folder_name):
    args = {}
    args['seed'] = depth
    args['min_depth'] = depth
    args['max_depth'] = depth
    args['train_data_points'] = numtrain
    args['valid_data_points'] = numval
    args['test_data_points'] = numval
    args['min_length'] = minlen
    args['train_max_length'] = maxlen
    args['valid_max_length'] = maxlen
    args['test_max_length'] = maxlen
    args['folder_name'] = folder_name
    return args

def make_cmd(args, nosm):
    cmd = 'python make_data.py '
#    cmd = 'python listops/data_processing/python/make_data.py '
    for k, v in args.items():
        cmd += '--{} {} '.format(k, v)
    if nosm:
        cmd += '--no_sm'
    return cmd

def get_folder_name(depth, maxlen, nosm):
    folder_name = 'd{}_ml{}'.format(depth, maxlen)
    if nosm:
        folder_name += '_nosm'
    return folder_name

if __name__ == '__main__':
    main()
