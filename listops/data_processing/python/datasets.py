# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import listops.data_processing.python.loading as _loading
import numpy as np
import os
import torch
PADDING_IDX = _loading.word_to_ix['<PAD>']

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def pad2d(seq, padding_val=-1):
    'Pad a sequence of 2D tensors, always batch_first'
    assert seq[0].dim() == 2
    bs = len(seq)
    maxlen = max([t.shape[-1] for t in seq])
    out = seq[0].new_ones(bs, maxlen, maxlen) * padding_val
    for ix, t in enumerate(seq):
        l = len(t)
        out[ix,:l, :l] = t
    return out

class MultiListOpsDataset(torch.utils.data.Dataset):

    def __init__(self, data_paths, depths, maxnum, d1_datapath=None, numd1=0):
        print(depths)
        data = []
        self.depths = []
        for data_path, depth in zip(data_paths, depths):
            d = _loading.load_raw_data(data_path, np.inf)
            data += d[:maxnum]
            self.depths.extend([depth] * maxnum)
        if (d1_datapath is not None) and (numd1 > 0):
            d = _loading.load_raw_data(d1_datapath, np.inf)
            data += d[:numd1]
            self.depths.extend([1] * numd1)

        self.num_tokens = [torch.LongTensor(d['num_tokens']) for d in data]
        self.labels = [torch.LongTensor([d['label']]) for d in data]
        self.arcmats = [torch.LongTensor(d['hp_arcmat']) for d in data]

    def __getitem__(self, index):
        num_token = self.num_tokens[index]
        label = self.labels[index]
        arcmat = self.arcmats[index]
        depth = self.depths[index]
        assert len(num_token) == len(arcmat)
        length = len(num_token)
        return num_token, label, arcmat, length, depth

    def __len__(self):
        return len(self.num_tokens)

    @staticmethod
    def collate_fn(data):
        tokens, labels, arcmats, lengths, depths = zip(*data)
        sort_ixs = argsort(lengths)[::-1]
        sorted_seqs = [[seq[ix] for ix in sort_ixs]
                            for seq in  [tokens, labels, arcmats, lengths, depths]]
        tokens = torch.nn.utils.rnn.pad_sequence(sorted_seqs[0],
                    batch_first=True, padding_value=PADDING_IDX)
        labels = torch.cat(sorted_seqs[1])
        arcmats = pad2d(sorted_seqs[2], padding_val=0) # Pad zeros, it is like these nodes do not exist
        lengths = torch.LongTensor(sorted_seqs[3])
        depths = sorted_seqs[4]
        return tokens, labels, arcmats, lengths, depths

class ListOpsDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, maxlen=float("inf")):
        data = _loading.load_raw_data(data_path, maxlen)
        self.num_tokens = [torch.LongTensor(d['num_tokens']) for d in data]
        self.labels = [torch.LongTensor([d['label']]) for d in data]
        self.arcmats = [torch.LongTensor(d['hp_arcmat']) for d in data]

    def __getitem__(self, index):
        num_token = self.num_tokens[index]
        label = self.labels[index]
        arcmat = self.arcmats[index]
        assert len(num_token) == len(arcmat)
        length = len(num_token)
        return num_token, label, arcmat, length

    def __len__(self):
        return len(self.num_tokens)

    @staticmethod
    def collate_fn(data):
        tokens, labels, arcmats, lengths = zip(*data)
        sort_ixs = argsort(lengths)[::-1]
        sorted_seqs = [[seq[ix] for ix in sort_ixs]
                            for seq in  [tokens, labels, arcmats, lengths]]
        tokens = torch.nn.utils.rnn.pad_sequence(sorted_seqs[0],
                    batch_first=True, padding_value=PADDING_IDX)
        labels = torch.cat(sorted_seqs[1])
        arcmats = pad2d(sorted_seqs[2], padding_val=0) # Pad zeros, it is like these nodes do not exist
        lengths = torch.LongTensor(sorted_seqs[3])
        return tokens, labels, arcmats, lengths

if __name__ == '__main__':
    path = './listops/data_processing/processed_0/train.tsv'
    ds = ListOpsDataset(path, 100)
    dl = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True,
            collate_fn=ListOpsDataset.collate_fn)
    for batch in dl:
        break
    print(batch)
    import pdb; pdb.set_trace()
