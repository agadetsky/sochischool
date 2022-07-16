import listops.model_modules.func_parse as _parse
import listops.model_modules.func_spanningtree as _sp
import listops.model_modules.func_topk as _topk
import listops.model_modules.func_independent as _ind
import torch

EPS = torch.finfo(torch.float32).tiny

# This module contains the samplers, idea is to input logits, perturb and then
# call project (obviously call different projects for soft and for hard, so have
# separate methods here)

### Projective
def sample_projective(A, lengths=None, mode='soft', noise='gumbel', tau=1.0,
                        single_root=False, clip_or_rescale=None):
    del clip_or_rescale
    assert mode in set(['soft', 'st', 'hard'])
    assert noise in set(['gumbel', 'exp', 'gaussian'])

    # Perturb A
    A = get_perturbed(A, noise, tau)
    # Compute (possibly relaxed) sample
    sample = _parse.projective_parse(A, lengths, mode, single_root)
    return sample

### Non-projective
def sample_nonprojective(A, lengths=None, mode='soft', noise='gumbel', tau=1.0,
                         single_root=False, clip_or_rescale="rescale"):
    assert mode in set(['soft', 'st', 'hard'])
    assert noise in set(['gumbel', 'exp', 'gaussian'])
    # Perturb A
    A = get_perturbed(A, noise, tau)
    # Compute (possibly relaxed) sample
    sample = _parse.nonprojective_parse(A, lengths, mode, single_root, clip_or_rescale)
    return sample

def sample_spanning_tree(A, lengths=None, mode='soft', tau=1.0):
    assert mode in set(['soft', 'st', 'hard'])
    # only perturb the upper triangle, b/c will get folded
    A = torch.triu(get_perturbed(A, 'gumbel', tau), diagonal=1) + torch.tril(A)
    # Compute (possibly relaxed) sample
    sample = _sp.spanning_tree(A, lengths, mode)
    return sample

def sample_top_n_minus_one(A, lengths=None, mode='soft', tau=1.0, eps=1e-2):
    assert mode in set(['soft', 'st', 'hard'])
    # Perturb A
    A = get_perturbed(A, 'gumbel', tau)
    # Compute (possibly relaxed) sample
    sample = _topk.top_n_minus_one(A, lengths, mode, eps=eps)
    return sample

def sample_indepdentent_edges(A, lengths=None, mode='soft', tau=1.0):
    assert mode in set(['soft', 'st', 'hard'])
    # only perturb one potential b/c will get folded
    A = torch.triu(get_perturbed(A, 'logistic', tau)) + torch.tril(A)
    # Compute (possibly relaxed) sample
    sample = _ind.edge_sigmoid(A, lengths, mode)
    return sample

def sample_independent_arcs(A, lengths=None, mode='soft', tau=1.0):
    assert mode in set(['soft', 'st', 'hard'])
    # Perturb A
    A = get_perturbed(A, 'logistic', tau)
    # Compute (possibly relaxed) sample
    sample = _ind.arc_sigmoid(A, lengths, mode)
    return sample

def get_perturbed(logits, noise, tau):
    uniforms = torch.empty_like(logits).float().uniform_().clamp_(EPS, 1 - EPS)
    if noise == 'gumbel':
        noise = uniforms.log().neg().log().neg()
        perturbed = (logits + noise) / tau
    elif noise == 'exp': #exponential
        noise = uniforms.log().neg()
        # perturbed = noise / (torch.nn.functional.softplus(logits) * tau + EPS)
        # perturbed = noise / (torch.exp(logits - logits.max()) * tau + EPS)
        # perturbed = noise / (torch.exp(
        #     logits - logits.max(-1, keepdims=True)[0].max(-2, keepdims=True)[0]) * tau + EPS)
        perturbed = -noise * torch.nn.functional.softplus(logits) / tau
    elif noise == 'logistic':
        noise = uniforms.log() - uniforms.neg().log1p()
        perturbed = (logits + noise) / tau
    elif noise == 'gaussian':
        noise = torch.randn_like(logits)
        perturbed = (logits + noise) / tau
    else:
        raise NotImplementedError

    return perturbed
