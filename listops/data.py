### Basic dataloading funcationality
import sys
sys.path.append('../')

import listops.data_processing.python.datasets as _datasets
import listops.data_processing.python.datasampler as _sampler
import numpy as np
import os
import torch

DATADIR = './data_processing/python/listops/data'

# If you create a new dataset, add its name and its path here.
# DATASETS = {
#     'proc0': {
#         'base_path': './listops/data_processing/processed_0',
#         'maxlens': [130, 300, np.inf],
#         #'train_maxlen': 130,
#         #'val_maxlen': 300,
#         #'test_maxlen': np.inf,
#     },
#     'd4_ml30': {
#         'base_path': './listops/data/d4_ml30',
#         'maxlens': [100, 150, np.inf],
#     },
#     'd2_ml30_nosm': {
#         # --train_data_points 80000 --valid_data_points 10000 --test_data_points 10000 --train_max_length 30 --valid_max_length 30 --test_max_length 30 --min_depth 2 --max_depth 2 --folder_name d2_ml30_nosm --min_length 10 --no_sm
#         'base_path': './listops/data/d2_ml30_nosm',
#         'maxlens': [30, 30, 30],
#     },
#     'd5_ml30_nosm': {
#         # --train_data_points 80000 --valid_data_points 10000 --test_data_points 10000 --train_max_length 30 --valid_max_length 30 --test_max_length 30 --min_depth 5 --max_depth 5 --folder_name d5_ml30_nosm --min_length 10 --no_sm
#         'base_path': './listops/data/d5_ml30_nosm',
#         'maxlens': [30, 30, 30],
#     },
#     'd10_ml50_nosm': {
#         # --train_data_points 80000 --valid_data_points 10000 --test_data_points 10000 --train_max_length 50 --valid_max_length 50 --test_max_length 50 --min_depth 10 --max_depth 10 --folder_name d10_ml50_nosm --min_length 10 --no_sm
#         'base_path': './listops/data/d10_ml50_nosm',
#         'maxlens': [50, 50, 50],
#     },
#     'd2_ml30': {
#         # --train_data_points 80000 --valid_data_points 10000 --test_data_points 10000 --train_max_length 30 --valid_max_length 30 --test_max_length 30 --min_depth 2 --max_depth 2 --folder_name d2_ml30 --min_length 10
#         'base_path': './listops/data/d2_ml30',
#         'maxlens': [30, 30, 30],
#     },
#     'd2to5_ml30_nosm':{
#         # --train_data_points 80000 --valid_data_points 10000 --test_data_points 10000 --train_max_length 30 --valid_max_length 30 --test_max_length 30 --min_depth 2 --max_depth 5 --folder_name d2to5_ml30_nosm --min_length 10 --no_sm
#         'base_path': './listops/data/d2to5_ml30_nosm',
#         'maxlens': [30, 30, 30],
#     },
# }

def parse_data_str(data_str):
    depth_type, max_depth, maxlen, sm, numd1 = data_str.split('_')
    assert depth_type in ['fix', 'var']
    assert max_depth.isnumeric()
    assert maxlen.isnumeric()
    assert sm in ['sm', 'nosm']
    assert numd1.isnumeric()
    is_fixed = (depth_type == 'fix')
    is_nosm = (sm == 'nosm')
    return is_fixed, int(max_depth), int(maxlen), is_nosm, int(numd1)

def get_datadir(depth, maxlen, is_nosm):
    dataname = 'd{}_ml{}'.format(depth, maxlen)
    if is_nosm:
        dataname += '_nosm'
    datadir = os.path.join(DATADIR, dataname)
    print(datadir)
    return datadir

def get_nums(is_fixed, max_depth, numd1, sizes):
    is_withd1 = numd1 > 0
    # Val_divisor is either 1 (fixed, withoutd1), 2 (fixed, withd1), var (not fixed, withd1)
    # Set numd1s
    divisors = np.ones(3, dtype=int)
    if not is_fixed:
        divisors *= (max_depth - 1)
    if is_withd1:
        divisors[1:] += 1

    maxnums = sizes // divisors
    # Numd1s
    numd1s = maxnums.copy() if is_withd1 else np.zeros(3, dtype=int)
    numd1s[0] = numd1
    return maxnums.tolist(), numd1s.tolist()

def get_datasets(data_str):
    sizes = np.array([80000, 10000, 10000]) # train, val, test
    is_fixed, max_depth, maxlen, is_nosm, numd1 = parse_data_str(data_str)
    tsvs = ['train.tsv', 'valid.tsv', 'test.tsv']
    # Gather the right paths
    depths = [[] for _ in range(len(tsvs))]
    datapaths = [[] for _ in range(len(tsvs))]
    if is_fixed:
        datadir = get_datadir(max_depth, maxlen, is_nosm)
        for i, tsv in enumerate(tsvs):
            datapaths[i].append(os.path.join(datadir, tsv))
            depths[i].append(max_depth)
    else:
        for d in range(1, max_depth):
            depth = d + 1 # start at depth 2
            datadir = get_datadir(depth, maxlen, is_nosm)
            for i, tsv in enumerate(tsvs):
                datapaths[i].append(os.path.join(datadir, tsv))
                depths[i].append(depth)
    # Gather d1 paths
    d1paths = []
    d1dir = get_datadir(1, maxlen, is_nosm)
    for i, tsv in enumerate(tsvs):
        d1paths.append(os.path.join(d1dir, tsv))
    maxnums, numd1s = get_nums(is_fixed, max_depth, numd1, sizes)
    print('maxnums')
    print(maxnums)
    # Create datasets
    datasets = []
    for data, depth, d1data, maxnum, numd1 in zip(datapaths, depths, d1paths, maxnums, numd1s):
        ds = _datasets.MultiListOpsDataset(data, depth, maxnum, d1data, numd1)
        datasets.append(ds)

    return datasets


# def get_datasets(data_str):
#     base_path = DATASETS[data_str]['base_path']
#     datasets = []
#     tsvs = ['train.tsv', 'valid.tsv', 'test.tsv']
#     maxlens = DATASETS[data_str]['maxlens']
#
#     for tsv, maxlen in zip(tsvs, maxlens):
#         d_path = os.path.join(base_path, tsv)
#         ds = _datasets.ListOpsDataset(d_path, maxlen)
#         datasets.append(ds)
#     return datasets

def get_dataloaders(datasets, batchsize, pin_mem=False, bucket=False,
                    eval_batchsize=1000):

    loaders = []
    batchsizes = [batchsize, eval_batchsize, eval_batchsize]
    shuffles = [True, False, False]
    drop_lasts = [True, False, False]
    if not bucket:
        for ds, bs, shuf, dropl in zip(datasets, batchsizes, shuffles, drop_lasts):
            print('ds')
            print(ds)
            loader = torch.utils.data.DataLoader(ds, bs, shuf, drop_last=dropl,
                        collate_fn=_datasets.MultiListOpsDataset.collate_fn,
                        pin_memory=pin_mem)
            loaders.append(loader)
    else:
        raise NotImplementedError
    return loaders


if __name__ == '__main__':
    data_str = 'var_5_50_nosm_20000'
    datasets = get_datasets(data_str)
    train_loader, val_loader, test_loader = get_dataloaders(datasets, 100)

    train_depths = {i : 0 for i in range(1, 6)}
    val_depths = {i : 0 for i in range(1, 6)}
    test_depths = {i : 0 for i in range(1, 6)}

    for batch_idx, (x, y, arcs, lengths, depths) in enumerate(train_loader):
        for depth in depths:
            train_depths[depth] += 1

    for batch_idx, (x, y, arcs, lengths, depths) in enumerate(val_loader):
        for depth in depths:
            val_depths[depth] += 1

    for batch_idx, (x, y, arcs, lengths, depths) in enumerate(test_loader):
        for depth in depths:
            test_depths[depth] += 1

    print(train_depths)
    print(val_depths)
    print(test_depths)

'''
if __name__ == '__main__':
    datasets_list = ["d2_ml30_nosm", "d5_ml30_nosm", "d2_ml30",
                     "d10_ml50_nosm", "d2to5_ml30_nosm"]
    for dataset in datasets_list:
        datasets = get_datasets(dataset)
        train_loader, val_loader, test_loader  = get_dataloaders(datasets, 1)
        training_set = set()
        # Convert data into string to uniquely identify the data.
        for batch in train_loader:
            string_data = "".join([str(i) for i in batch[0][0].numpy()])
            training_set.add(string_data)
        valid_set = set()
        for batch in val_loader:
            string_data = "".join([str(i) for i in batch[0][0].numpy()])
            valid_set.add(string_data)
        print(f"({dataset}) Overlapping: ", training_set.intersection(valid_set))
'''