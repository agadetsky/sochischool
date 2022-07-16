import numpy as np
import torch
import torch_struct as struct
import listops.model_modules.func_customparse as _customparse

from listops.chuliu_edmonds.edmonds import edmonds_cpp_pytorch
from listops.kruskals.kruskals import kruskals_cpp_pytorch


### Projective Parsing
def projective_parse(A, lengths=None, mode='soft', single_root=False):
    r'''
    Projective Parsing
    -   Compute either soft, straight-through (st) or hard sample
    -   soft + st are differentiable, hard is not
    -   soft + st use marginals as relaxation

    Arguments:
        A: Arc matrix with (possibly perturbed) log-potentials
        lengths: Length integers (bs,) for masking
        mode: in ['soft', 'st', 'hard']

    Returns:
        Projection (bs x N x N)
    '''
    assert mode in set(['soft', 'st', 'hard'])

    if single_root:
        # We think DependencyCRF is multi-root
        # https://github.com/harvardnlp/pytorch-struct/issues/65
        # We haven't implemented single-root
        raise NotImplementedError

    dist = struct.DependencyCRF(A, lengths)

    if mode == 'soft':
        sample = dist.marginals
    elif mode == 'st':
        soft = dist.marginals
        hard = dist.argmax.detach()
        sample = hard - soft.detach() + soft
    elif mode == 'hard':
        sample = dist.argmax.detach()
    return sample

### Non-Projective Parsing

def nonprojective_parse(A, lengths=None, mode='soft', single_root=False, clip_or_rescale="rescale"):
    r'''
    Non-Projective Parsing
    -   Compute either soft, straight-through (st) or hard sample
    -   soft + st are differentiable, hard is not
    -   soft + st use marginals as relaxation

    Arguments:
        A: Arc matrix with (possibly perturbed) log-potentials
        lengths: Length integers (bs,) for masking
        mode: in ['soft', 'st', 'hard']

    Returns:
        Projection (bs x N x N)
    '''
    assert mode in set(['soft', 'st', 'hard'])
    # dist = struct.NonProjectiveDependencyCRF(A, lengths)

    if mode == 'soft':
        #sample = dist.marginals
        if single_root:
            sample = _customparse.nonprojective_soft_single(A, lengths, clip_or_rescale)
        else:
            sample = _customparse.nonprojective_soft_multi(A, lengths, clip_or_rescale)
    else:
        # Hard sampling is faster on cpu than on gpu.
        hard = edmonds_cpp_pytorch(A.to('cpu'), lengths)
        hard = hard.to('cuda') if A.is_cuda else hard
        if mode == 'st':
            raise NotImplementedError
            # The problem here is that we haven't changed the API of edmonds_cpp_pytorch
            # And edmonds_cpp_pytorch expects a different arc representation than
            # nonprojective_soft_single or nonprojective_soft_multi
            if single_root:
                soft = _customparse.nonprojective_soft_single(A, lengths, clip_or_rescale)
            else:
                soft = _customparse.nonprojective_soft_multi(A, lengths, clip_or_rescale)
            sample = hard - soft.detach() + soft
        else:  # mode == "hard"
            sample = hard
    return sample


if __name__ == '__main__':
    import time
    batchsize = 128
    num_nodes = 50
    A = torch.randn(batchsize, num_nodes, num_nodes, requires_grad=True)
    end = time.time()
    soft = projective_parse(A, mode='soft')
    print('soft_proj.requires_grad {}'.format(soft.requires_grad))
    print('Time {}'.format(time.time() - end))
    end = time.time()
    st = projective_parse(A, mode='st')
    print('st_proj.requires_grad {}'.format(st.requires_grad))
    print('Time {}'.format(time.time() - end))
    end = time.time()
    hard = projective_parse(A, mode='hard')
    print('hard_proj.requires_grad {}'.format(hard.requires_grad))
    print('Time {}'.format(time.time() - end))
    end = time.time()
    soft = nonprojective_parse(A, mode='soft')
    print('soft_nonp.requires_grad {}'.format(soft.requires_grad))
    print('Time {}'.format(time.time() - end))
