import torch
#from listops.kruskals.kruskals import kruskals_cpp_pytorch

### Spanning Tree

def submatrix_index(n, i):
    bs = i.size(0)
    I = torch.ones((bs, n,n), dtype=bool)
    I[torch.arange(bs), i, :] = False
    I[torch.arange(bs), :, i] = False
    return I

def get_spanning_tree_marginals(logits, n):
    bs = logits.size(0)
    (i,j) = torch.triu_indices(n, n, offset=1)
    c = torch.max(logits, axis=-1, keepdims=True)[0]
    k = torch.argmax(logits, axis=-1)
    removei = i[k]

    weights = torch.exp(logits - c)

    W = torch.zeros(weights.size(0), n, n)
    W = W.cuda() if logits.is_cuda else W
    W[:, i, j] = weights
    W[:, j, i] = weights

    L = torch.diag_embed(W.sum(axis=-1)) - W
    subL = L[submatrix_index(n, removei)].view(bs, n-1, n-1)
    logzs = torch.slogdet(subL)[1]
    logzs = torch.sum(logzs + (n - 1) * c.flatten())
    sample = torch.autograd.grad(logzs, logits, create_graph=True)[0]
    return sample

def spanning_tree(A, lengths=None, mode='soft', mingap=-15):
    r'''
    Spanning tree
    -   Compute either soft, straight-through (st) or hard sample
    -   soft + st are differentiable, hard is not
    -   soft + st use marginals as relaxation

    Arguments:
        A: Arc matrix with (possibly perturbed) log-potentials
        lengths: Length integers (bs,) for masking
        mode: in ['soft', 'st', 'hard']
        mingap: Gap between max logit value and the all other logits.
            We clip any logit values that are lower such that the gap is
            greater than the mingap.

    Returns:
        Projection (bs x N x N)
    '''
    assert mode in set(['soft', 'st', 'hard'])
    bs = A.shape[0]
    n =  A.shape[1]
    lengths = torch.tensor(n).repeat(bs) if lengths is None else lengths

    i, j = torch.triu_indices(n, n, offset=1)
    # Symmeterize the arc matrix by summing the upper and lower
    # triangle of the arc matrix (average of score for i->j and j->i).
    A_sym = A + A.transpose(-2, -1)

    if mode in ['st', 'hard']:
        hard_samples = kruskals_cpp_pytorch(A_sym.detach().cpu(), lengths)
        hard_samples = hard_samples.to(A.device)
    if mode in ['soft', 'st']:
        # Mask logits based on lengths.
        maxes = A_sym.max(-1)[0].max(-1)[0].unsqueeze(0).transpose(0, 1)
        # Get mask to mask out varying lengths.
        mask_horiz = (
            torch.arange(n).expand(bs, n).to(A.device) < lengths.unsqueeze(1)
        ).repeat((1, 1, n)).reshape(A.shape)
        mask = mask_horiz * mask_horiz.transpose(1, 2)
        # Mask all rows and columns corresponding to the extraneous nodes to be
        # the minimum possible (max - mingap), except for the ones in the first row.
        tiled_maxes = maxes.repeat((1, n ** 2)).reshape(A.shape)
        masked_A = A_sym * mask + ~mask * (tiled_maxes + mingap)
        mask_horiz[:, 1:, :] = 1.0
        masked_A = masked_A * mask_horiz + ~mask_horiz * tiled_maxes

        # Clip logits such that the gap between the maximum logits and the rest
        # is less than mingap.]
        A = torch.max(masked_A, tiled_maxes + mingap)

        edge_logits = A[:, i, j]

        X = get_spanning_tree_marginals(edge_logits, n)

        samples = torch.zeros_like(A)
        samples[:, i, j] = X
        samples[:, j, i] = X

        # Mask out the extraneous nodes.
        samples = samples * mask

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
    res_varlen = spanning_tree(w, lengths, 'soft', mingap=-15)
    # Compute each sample individually and padding.
    res_ind = []
    for i in range(bs):
        w_i = w[i][:lengths[i], :lengths[i]]
        res = spanning_tree(w_i.unsqueeze(0), None, 'soft', mingap=-15)
        padded_res = torch.zeros_like(w[i])
        padded_res[:lengths[i], :lengths[i]] = res
        res_ind.append(padded_res)
    res_ind = torch.stack(res_ind)

    np.testing.assert_almost_equal(res_varlen.detach().numpy(),
                                   res_ind.detach().numpy(),
                                   decimal=3)
