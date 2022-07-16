from listops.model_modules.func_customparse import arcmask_from_lengths
import torch

def arc_sigmoid(A, lengths, mode):
    # Create samle
    soft = A.sigmoid()
    if mode == 'soft':
        sample = soft
    elif mode == 'st':
        hard = (A.detach() > 0.0)
        sample = hard - soft.detach() + soft
    elif mode == 'hard':
        sample = (A.detach() > 0.0).int()
    # Pad diagonal
    maxlen = A.shape[1]
    diag_mask = torch.eye(maxlen, device=A.device, dtype=bool).unsqueeze(0)
    sample = sample.masked_fill(diag_mask, 0.0)
    # Pad padded tokens
    arcmask = arcmask_from_lengths(A, lengths)
    sample = sample.masked_fill(arcmask, 0.0)
    # Return
    return sample

def edge_sigmoid(A, lengths, mode):
    # Upper triangle is pertubed, lower triangle is not
    # Can simpy add together, lower triangular will be zero
    # diagonal is undefined
    A = A.tril().transpose(-2, -1) + A.triu()
    # Create sample
    soft = A.sigmoid()
    if mode == 'soft':
        sample = soft
    elif mode == 'st':
        hard = (A.detach() > 0.0)
        sample = hard - soft.detach() + soft
    elif mode == 'hard':
        sample = (A.detach() > 0.0).int()
    # Pad diagonal
    maxlen = A.shape[1]
    diag_mask = torch.eye(maxlen, device=A.device, dtype=bool).unsqueeze(0)
    sample = sample.masked_fill(diag_mask, 0.0)
    # Pad padded tokens
    arcmask = arcmask_from_lengths(A, lengths)
    sample = sample.masked_fill(arcmask, 0.0)
    # Make symmetric (copy the upper triangular into the lower triangular)
    sample += sample.triu().transpose(-2, -1)
    return sample
