####
# Purpose of this module is to define our replacements for torch_struct.
###
import numpy as np
import torch

def clip_range(x, mingap=-np.inf):
    maxes = x.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    return torch.max(x, mingap * torch.ones_like(x) + maxes)

def nonprojective_soft_single(A, lengths, clip_or_rescale="rescale", 
                              mingap=-15, eps=1e-5):
    bs, maxlen, _ = A.shape
    if clip_or_rescale == "rescale":
        A = rescale_tensor(A)
    elif clip_or_rescale == "clip":
        A = clip_range(A, mingap=mingap)
    elif clip_or_rescale == "None":
        pass
    else:
        NotImplementedError
    Aexp = to_exp(A, lengths, eps)
    Aexp = fix_pads_single(Aexp, lengths)
    lap = get_laplacian_single(Aexp)
    # Directly from torch_struct (should work since we took care of pads)
    inv_laplacian = lap.inverse()
    factor = (
        torch.diagonal(inv_laplacian, 0, -2, -1)
        .unsqueeze(2)
        .expand_as(Aexp)
        .transpose(1, 2)
    )
    term1 = Aexp.mul(factor).clone()
    term2 = Aexp.mul(inv_laplacian.transpose(1, 2)).clone()
    term1[:, :, 0] = 0
    term2[:, 0] = 0
    output = term1 - term2
    roots_output = (
        torch.diagonal(Aexp, 0, -2, -1).mul(inv_laplacian.transpose(1, 2)[:, 0])
    )
    output = output + torch.diag_embed(roots_output, 0, -2, -1)
    # Remove padded marginals
    mask = mask_from_lengths(maxlen, lengths)
    output[:, 0] -= mask.float()
    return output

def nonprojective_soft_multi(A, lengths, clip_or_rescale="rescale", 
                             mingap=-15, eps=1e-5):
    r'''
    A is arc_logits (i.e. unnormalised log-potentials for each arc).
    This function assumes the same form as torch_struct.
    A should NOT include the root symbol, and its diagonal != 0, instead the
    diagonal holds the "root_potentials" (arc_potentials for connecting the
    respective node to the root.)

    In contrast to torch_struct, this function correctly handles padded As and
    DOES USE the lengths argument. Specifically, the lengths argument implicitly
    defines a mask + entries in A that correspond to "padded" nodes will be
    handled appropriately (they are not assumed to be zero or -inf).
    '''
    bs, maxlen, _ = A.shape
    if clip_or_rescale == "rescale":
        A = rescale_tensor(A)
    elif clip_or_rescale == "clip":
        A = clip_range(A, mingap=mingap)
    elif clip_or_rescale == "None":
        pass
    else:
        NotImplementedError
    Aexp = to_exp(A, lengths, eps)
    Aexp = fix_pads_multi(Aexp, lengths)
    lap = get_laplacian_multi(Aexp)

    # Now need to compute the marginals from this
    inv_laplacian = lap.inverse()
    # off-diagonal marginals
    diag_factor = (inv_laplacian.diagonal(0, -2, -1)
                    .unsqueeze(2)
                    .expand_as(A)
                    .transpose(1, 2))
    term1 = Aexp * diag_factor
    term2 = Aexp * inv_laplacian.transpose(1, 2)
    marginals = term1 - term2
    # on-diagonal marginals
    root_marginals = inv_laplacian.diagonal(0, -2, -1) * Aexp.diagonal(0, -2, -1)
    marginals = marginals + torch.diag_embed(root_marginals, 0, -2, -1)
    # Remove padded diagonal marginals
    mask = mask_from_lengths(maxlen, lengths)
    marginals -= torch.diag_embed(mask.float())
    return marginals

# Helper utilities
def mask_from_lengths(maxlen, lengths):
    r'''
    This function creates a boolean mask from a lengths argument.
    Entries are 1 at indices which correspond to padded tokens.
    '''
    assert lengths.dim() == 1
    device = lengths.device
    mask = torch.arange(maxlen)[None, :].to(device) >= lengths[:, None]
    return mask

def arcmask_from_lengths(A, lengths):
    r'''
    This function creates a boolean mask for an arc matrix using lengths.
    The mask is going to have the same shape as A.
    Entries are 1 at indices which correspond to padded tokens.
    '''
    assert A.dim() == 3
    assert A.size(1) == A.size(2)
    bs, maxlen, _ = A.shape
    mask = mask_from_lengths(maxlen, lengths)
    arcmask = mask.unsqueeze(1).expand_as(A) | mask.unsqueeze(2).expand_as(A)
    return arcmask

# Re-scale logits
def rescale_tensor(tensor, eps=1e-8):
    '''If tensor.min() < min and/or max < tensor.max() rescales all tensors
    to lie between MIN and MAX. If not, has no effect. Operates batchwise.
    '''
    MIN = -10
    MAX = 10
    # First normalise everything to [0, 1] (batchwise)
    min_logit = tensor.min(-1).values.min(-1).values.reshape(-1,1,1)
    max_logit = tensor.max(-1).values.max(-1).values.reshape(-1,1,1)
    tensor_u  = (tensor - min_logit) / (max_logit - min_logit + eps)
    # Scale max
    scale_min = torch.clamp(min_logit, min=MIN)
    scale_max = torch.clamp(max_logit, max=MAX)
    # Re-scaled tensor
    tensor_scaled = scale_min + tensor_u * (scale_max - scale_min)
    return tensor_scaled

def to_exp(A, lengths, eps=1e-5):
    r'''
    Take to exp space.
    Make computation safe here.
    '''
    c = A.max(-1).values.max(-1).values.reshape(-1, 1, 1)
    return torch.exp(A - c) + eps

# Fix pads
def fix_pads_single(Aexp, lengths):
    r'''
    This function applies a single-root pad to the Aexp matrix.
    This allows handling of variable lengths. The basic idea is that all
    padded nodes will always be directly connected to the node at position zero.
    Choosing the node at position zero is arbitrary. We make sure padded nodes
    have an arc potential of one in exp space to that node + potential of zero
    to all other nodes including the root symbol.
    '''
    bs, maxlen, _ = Aexp.shape
    mask = mask_from_lengths(maxlen, lengths)
    arcmask = arcmask_from_lengths(Aexp, lengths)
    fixed = Aexp.masked_fill(arcmask, 0.0) # masks out all arcs for padded
    fixed[:, 0] += mask.float()
    return fixed

def fix_pads_multi(Aexp, lengths):
    r'''
    This function applies a multi-root pad to the Aexp matrix.
    This allows handling of variable lengths. The basic idea is that all
    padded nodes will always be directly connected to the root symbol.
    This is ensured by defining the root potential of all padded nodes to be 1
    (in exp space, so that it doesn't change the overall value of any
    aborescence) and zero for arcs to any other nodes.
    '''
    bs, maxlen, _ = Aexp.shape
    mask = mask_from_lengths(maxlen, lengths)
    arcmask = arcmask_from_lengths(Aexp, lengths)
    fixed = Aexp.masked_fill(arcmask, 0.0)
    fixed = fixed + torch.diag_embed(mask.float())
    return fixed

# Laplacian matrices
def get_laplacian_single(Aexp):
    r'''
    This function produces \hat{L} as in
    http://www.cs.columbia.edu/~mcollins/papers/matrix-tree.pdf
    for the single-root case.

    It assumes A is torch_struct format, i.e. the root node is ghost and the
    diagonal holds root potentials. It also assumes that A has already been
    made padding-ready, so this function does not need to care of dealing with
    paddings in any way. A is given in exp space.
    '''
    # Compute L
    diag_mask = torch.eye(Aexp.shape[1], device=Aexp.device, dtype=bool)
    lap = Aexp.masked_fill(diag_mask, 0) # ignore root potentials for sum below
    lap = -lap + torch.diag_embed(lap.sum(1), offset=0, dim1=-2, dim2=-1)
    # Compute \hat{L}
    root_potentials = Aexp.diagonal(0, -2, -1)
    laphat = lap
    laphat[:, 0] = root_potentials
    return laphat

def get_laplacian_multi(Aexp):
    r'''
    This function produces \hat{L} as in
    http://www.cs.columbia.edu/~mcollins/papers/matrix-tree.pdf
    for the multi-root case.

    It assumes A is torch_struct format, i.e. the root node is ghost and the
    diagonal holds root potentials. It also assumes that A has already been
    made padding-ready, so this function does not need to care of dealing with
    paddings in any way. A is given in exp space.
    '''
    # Compute L
    diag_mask = torch.eye(Aexp.shape[1], device=Aexp.device, dtype=bool)
    lap = Aexp.masked_fill(diag_mask, 0) # ignore root potentials for sum below
    lap = -lap + torch.diag_embed(lap.sum(1), offset=0, dim1=-2, dim2=-1)
    # Compute \hat{L}
    root_potentials = Aexp.diagonal(0, -2, -1)
    laphat = lap + root_potentials.diag_embed(0, -2, -1)
    return laphat
