import torch
from torch.autograd import Function, Variable, grad
from torch.nn import Module
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr
import scipy.special as spec

from listops.model_modules.func_customparse import arcmask_from_lengths

INF = np.finfo(np.float32).max
MAX = torch.finfo(torch.float32).max


def softtopk_fixedk_forward_np(logits, k):
    batchsize, n = logits.shape
    messages = -INF * np.ones((batchsize, n, k+1))
    messages[:, 0,0] = 0
    messages[:, 0,1] = logits[:, 0]
    for i in range(1, n):
        for j in range(k + 1):
            logp_dont_use = messages[:, i - 1, j]
            logp_use = messages[:, i - 1, j - 1] + logits[:, i] if j > 0 else - INF
            message = np.logaddexp(logp_dont_use, logp_use)
            messages[:, i,j] = message
    return messages


def softtopk_fixedk_backward_np(logits, k):
    batchsize, n = logits.shape
    messages = -INF * np.ones((batchsize, n, k+1))
    messages[:, n - 1, k] = 0
    for i in range(n - 2, -1, -1):
        for j in range(k + 1):
            logp_dont_use = messages[:, i + 1, j]
            logp_use = messages[:, i + 1, j + 1] + logits[:, i + 1] if j < k else - INF
            message = np.logaddexp(logp_dont_use, logp_use)
            messages[:,i,j] = message
    return messages


def softtopk_fixedk_np(logits, k):
    batchsize = logits.shape[0]
    f = softtopk_fixedk_forward_np(logits, k)
    b = softtopk_fixedk_backward_np(logits, k)
    initial_f = -INF * np.ones((batchsize, 1, k + 1))
    initial_f[:, :, 0] = 0
    ff = np.concatenate([initial_f, f[:,:-1,:]], axis=1)
    lse0 = spec.logsumexp(ff + b, axis = 2)
    lse1 = spec.logsumexp(ff[:, :, :-1] + b[:, :, 1:], axis = 2) + logits
    return np.exp(lse1 - np.logaddexp(lse0, lse1))


class SoftTopKFixedK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, k, eps):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(logits)
        ctx.k = k
        ctx.eps = eps
        dtype = logits.dtype
        device = logits.device
        mu_np = softtopk_fixedk_np(logits.cpu().detach().numpy(), k)
        mu = torch.from_numpy(mu_np).type(dtype).to(device)
        return mu

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        r'''http://www.cs.toronto.edu/~kswersky/wp-content/uploads/carbm.pdf'''
        logits, = ctx.saved_tensors
        k = ctx.k
        eps= ctx.eps
        dtype = grad_output.dtype
        device = grad_output.device
        logits_np = logits.cpu().detach().numpy()
        grad_output_np = grad_output.cpu().detach().numpy()
        n1 = softtopk_fixedk_np(logits_np + eps * grad_output_np, k)
        n2 = softtopk_fixedk_np(logits_np - eps * grad_output_np, k)
        grad_np = (n1 - n2) / (2 * eps)
        grad = torch.from_numpy(grad_np).type(dtype).to(device)
        return grad, None, None


def get_mask_np(n, lengths):
    bs = lengths.shape[0]
    max_len = np.max(lengths).astype(np.int64)
    mask = np.tile(np.arange(max_len + 1), (bs, 1)) <= np.expand_dims(lengths, 1)
    return np.tile(np.expand_dims(mask, 1), (1, n, 1))

def softtopk_forward_np(logits, ks):
    max_k = np.max(ks).astype(np.int64)
    batchsize, n = logits.shape
    messages = -INF * np.ones((batchsize, n, max_k + 1))
    messages[:, 0, 0] = 0
    messages[:, 0, 1] = logits[:, 0]
    for i in range(1, n):
        for j in range(max_k + 1):
            logp_dont_use = messages[:, i - 1, j]
            logp_use = np.maximum(messages[:, i - 1, j - 1] + logits[:, i], -INF) if j > 0 else - INF
            message = np.logaddexp(logp_dont_use, logp_use)
            messages[:, i, j] = message
    
    k_mask = get_mask_np(n, ks)
    masked_messages = messages * k_mask - INF * ~k_mask
    return masked_messages


def softtopk_backward_np(logits, ks):
    max_k = np.max(ks).astype(np.int64)
    batchsize, n = logits.shape
    messages = -INF * np.ones((batchsize, n, max_k + 1))
    messages[np.arange(batchsize), n - 1, ks] = 0
    for i in range(n - 2, -1, -1):
        for j in range(max_k + 1):
            logp_dont_use = messages[:, i + 1, j]
            logp_use = np.maximum(messages[:, i + 1, j + 1] + logits[:, i + 1], -INF) if j < max_k else - INF
            message = np.logaddexp(logp_dont_use, logp_use)
            messages[:,i,j] = message
    k_mask = get_mask_np(n, ks)
    masked_messages = messages * k_mask - INF * ~k_mask
    return masked_messages


def softtopk_np(logits, ks):
    max_k = np.max(ks).astype(np.int64)
    batchsize = logits.shape[0]
    f = softtopk_forward_np(logits, ks)
    b = softtopk_backward_np(logits, ks)
    initial_f = -INF * np.ones((batchsize, 1, max_k + 1))
    initial_f[:, :, 0] = 0
    ff = np.concatenate([initial_f, f[:,:-1,:]], axis=1)
    lse0 = spec.logsumexp(ff + b, axis = 2)
    lse1 = spec.logsumexp(ff[:, :, :-1] + b[:, :, 1:], axis = 2) + logits
    return np.exp(lse1 - np.logaddexp(lse0, lse1))


class SoftTopK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, ks, eps):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(logits)
        ctx.ks = ks
        ctx.eps = eps
        dtype = logits.dtype
        device = logits.device
        mu_np = softtopk_np(logits.cpu().detach().numpy(),
                            ks.cpu().detach().numpy())
        mu = torch.from_numpy(mu_np).type(dtype).to(device)
        return mu

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        r'''http://www.cs.toronto.edu/~kswersky/wp-content/uploads/carbm.pdf'''
        logits, = ctx.saved_tensors
        ks = ctx.ks
        eps= ctx.eps
        dtype = grad_output.dtype
        device = grad_output.device
        logits_np = logits.cpu().detach().numpy()
        grad_output_np = grad_output.cpu().detach().numpy()
        n1 = softtopk_np(logits_np + eps * grad_output_np, ks.cpu().detach().numpy())
        n2 = softtopk_np(logits_np - eps * grad_output_np, ks.cpu().detach().numpy())
        grad_np = (n1 - n2) / (2 * eps)
        grad = torch.from_numpy(grad_np).type(dtype).to(device)
        return grad, None, None


def top_n_minus_one(A, lengths=None, mode='soft', eps=1e-2, use_loop=False):
    r'''
    Top |V| - 1
    -   Compute either soft, straight-through (st) or hard sample
    -   soft + st are differentiable, hard is not
    -   soft + st use marginals as relaxation

    Arguments:
        A: Arc matrix with (possibly perturbed) log-potentials
        lengths: Length integers (bs,) for masking
        mode: in ['soft', 'st', 'hard']
        eps: Epsilon value used for finite differences when computing
            gradients for the backward pass for mode == 'soft'.
        use_loop: Instead of using variable k code, use fixed k code in a loop.
            Should be True only for testing purposes.

    Returns:
        Projection (bs x N x N)
    '''
    assert mode in set(['soft', 'st', 'hard'])
    bs = A.shape[0]
    n =  A.shape[1]
    lengths = torch.tensor(n).repeat(bs) if lengths is None else lengths
    ks = lengths - 1

    if use_loop:
        samples = torch.zeros_like(A)
        for idx in range(bs):
            sliced_A = A[idx][:lengths[idx], :lengths[idx]]
            i, j = torch.triu_indices(lengths[idx], lengths[idx], 1)
            w = torch.cat((sliced_A[i, j], sliced_A[j, i]), axis=-1)
            if mode == 'hard':
                topk, indices = torch.topk(w, lengths[idx] - 1, axis=-1)
                X = torch.zeros_like(w).scatter(0, indices, topk.bool().float())
            elif mode == 'soft':
                X = SoftTopKFixedK.apply(w.unsqueeze(0), lengths[idx] - 1, eps)[0]
            else:
                # No point checking straight-through for loop.
                NotImplementedError
            # import pdb;pdb.set_trace()
            samples[idx, i, j] = X[:X.shape[0] // 2]
            samples[idx, j, i] = X[X.shape[0] // 2:]
        return samples

    arcmask = arcmask_from_lengths(A, lengths)
    masked_A = A.masked_fill(arcmask, -MAX)

    # Flatten masked_A excluding the diagonals.
    i, j = torch.triu_indices(n, n, 1)
    masked_weights = torch.cat((masked_A[:, i, j], masked_A[:, j, i]), axis=-1)

    if mode in ['st', 'hard']:
        # Get top (|V| - 1) where |V| varies by sample and is given by lengths.
        num_edges = n * (n - 1)
        mask = torch.arange(num_edges).expand(bs, num_edges).to(A.device) < ks.unsqueeze(1)
        sorted_values, indices = torch.sort(masked_weights, descending=True)
        sorted_values = sorted_values * mask
        X = torch.zeros_like(masked_weights).scatter_(1, indices, mask.float())

        hard_samples = torch.zeros_like(A)
        hard_samples[:, i, j] = X[:, :X.shape[1] // 2]
        hard_samples[:, j, i] = X[:, X.shape[1] // 2:]
    if mode in ['soft', 'st']:
        X = SoftTopK.apply(masked_weights, ks, eps)

        samples = torch.zeros_like(A)
        samples[:, i, j] = X[:, :X.shape[1] // 2]
        samples[:, j, i] = X[:, X.shape[1] // 2:]

    if mode == 'hard':
        samples = hard_samples
    elif mode == 'st':
        samples = (hard_samples - samples).detach() + samples

    return samples


if __name__ == '__main__':
    import numpy as np

    bs = 100
    n = 50
    lengths = torch.tensor(np.random.choice(np.arange(10, n), bs))
    w = torch.randn((bs, n, n), requires_grad = True)

    # Compute with variable length.
    res_varlen = top_n_minus_one(w, lengths, 'soft')
    res_varlen_hard = top_n_minus_one(w, lengths, 'hard')

    np.testing.assert_almost_equal(res_varlen.detach().numpy().sum(-1).sum(-1),
                                   lengths - 1,
                                   decimal=5)
    np.testing.assert_almost_equal(res_varlen_hard.detach().numpy().sum(-1).sum(-1),
                                   lengths - 1,
                                   decimal=10)

    # Compare with old code with fixed length in a loop
    res_varlen_loop = top_n_minus_one(w, lengths, 'soft', use_loop=True)
    res_varlen_hard_loop = top_n_minus_one(w, lengths, 'hard', use_loop=True)
    np.testing.assert_almost_equal(res_varlen.detach().numpy(),
                                   res_varlen_loop.detach().numpy())
    np.testing.assert_almost_equal(res_varlen_hard.detach().numpy(),
                                   res_varlen_hard_loop.detach().numpy())