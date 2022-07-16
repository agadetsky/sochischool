import listops.model_modules.func_sampling as _sampling
import torch
import torch.nn as nn

import sys
sys.path.append('../../../')

#import edmonds.distributions
#import edmonds.estimators

# TODO: Need to parse temperature in.
def get_sampler(sampler_str, tau):
    if 'proj' in sampler_str: # filters out projective + nonprojective sampler, keyword is reserved
        parse_type, mode, noise, root_known, single_root, clip_or_rescale = parse_sampler_strings(sampler_str)
        sampler = get_proj_nonproj_sampler(parse_type, mode, noise, tau, root_known, single_root, clip_or_rescale)
    elif 'identity' == sampler_str:
        sampler = IdentitySampler()
    elif 'spanning' in sampler_str:
        _, mode = sampler_str.split('_')
        sampler = SpanningTreeSampler(mode, tau)
    elif 'topk' in sampler_str:
        _, mode, eps = sampler_str.split('_')
        sampler = TopKSampler(mode, tau, float(eps))
    elif 'indarc' in sampler_str:
        _, mode = sampler_str.split('_')
        sampler = IndependentSampler('arc', mode, tau)
    elif 'indedge' in sampler_str:
        _, mode = sampler_str.split('_')
        sampler = IndependentSampler('edge', mode, tau)
    elif 'edmonds' in sampler_str:
        sampler = EdmondsSampler()
    else:
        raise NotImplementedError
    return sampler

##############################################################################

### Projective + Non-Projective Samplers


class EdmondsSampler(nn.Module):
    def __init__(self):
        super(EdmondsSampler, self).__init__()


    def forward(self, logits, lengths):
        u = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
        z = edmonds.estimators.to_z(logits, u)
        arb, stats = edmonds.estimators.to_b(z, lengths)
        return arb, stats, z


def parse_sampler_strings(str):
    parse_str, mode, noise, clip_or_rescale  = str.split('_')
    if 'proj' == parse_str[:4]:
        parse_type = 'proj'
    elif 'nonproj' == parse_str[:7]:
        parse_type = 'nonproj'
    else:
        raise ValueError('Unknown Parse Type')
    root_known = True if ('k' in parse_str) else False # k is flag for known
    single_root = True if ('s' in parse_str) else False # s is flag for single
    assert clip_or_rescale in ["rescale", "clip", "None", ""]

    return parse_type, mode, noise, root_known, single_root, clip_or_rescale

def get_proj_nonproj_sampler(parse_type, mode, noise, tau, root_known, single_root, clip_or_rescale=None):
    if parse_type == 'proj':
        sampler = ProjectiveSampler(mode, noise, tau, root_known, single_root)
    elif parse_type == 'nonproj':
        sampler = NonProjectiveSampler(mode, noise, tau, root_known, single_root, clip_or_rescale)
    else:
        raise NotImplementedError
    return sampler

class Sampler(nn.Module):
# Generic Sampler (abstract class) # TODO: make proper absract class with inhertiance

    def __init__(self, mode, noise, tau):
        super(Sampler, self).__init__()
        self.mode = mode
        self.noise = noise
        self.tau = tau

    def forward(self, A, lengths=None, training_mode=None):
        training_mode = self.training if training_mode is None else training_mode
        if training_mode:
            return self.forward_train(A, lengths)
        else:
            return self.forward_eval(A, lengths)

    def forward_train(self, A, lengths=None):
        raise NotImplementedError

    def forward_eval(self, A, lengths=None):
        raise NotImplementedError

    def batch_eval(self, A, lengths=None, num_samples=1):
        raise NotImplementedError
        # TODO: Need to correct mannipulation of lengths
        # batch is here not batch, but several samples
        # batch_shape = A.shape
        # event_shape = batch_shape[1:]
        # A_expand = A.unsqueeze(0).expand(num_samples, batch_shape).view(-1, *event_shape)
        # assert lengths.dim = 1
        # lengths_expand = lengths.unsqueeze(0).expand(num_samples, lengths.shape).view(-1)
        # sample_expand = self.forward_eval(A_expand, lengths_expand)
        # return sample_expand.view(num_samples, batch_shape)


class DependencySampler(Sampler):
    r'''This is the parent of the projective + nonprojective samplers'''

    def __init__(self, mode, noise, tau, root_known, single_root, clip_or_rescale=None):
        super(DependencySampler, self).__init__(mode, noise, tau)
        self.root_known = root_known
        self.single_root = single_root
        self.clip_or_rescale = clip_or_rescale

    def forward_train(self, A, lengths=None):
        A, lengths = self.prepare_input(A, lengths)
        sample = self.sample(A, lengths, self.mode)
        sample = self.prepare_output(sample)
        return sample

    def forward_eval(self, A, lengths=None):
        A, lengths = self.prepare_input(A, lengths)
        sample = self.sample(A, lengths, 'hard')
        sample = self.prepare_output(sample)
        return sample

    def sample(self, A, lengths, mode):
        raise NotImplementedError

    def prepare_input(self, A, lengths):
        if self.root_known:
            # The zeroth token is the root node + we enforce this
            old_diag = A[:, 1:, 1:].diagonal(0, -2, -1)
            new_diag = A[:, 0, 1:]
            prep_A = (A[:, 1:, 1:]
                        - old_diag.diag_embed(0, -2, -1)
                        + new_diag.diag_embed(0, -2, -1))
            if lengths is not None:
                prep_lengths = lengths - 1
            else:
                prep_lengths = None
            return prep_A, prep_lengths
        else:
            # Ghost root mode
            return A, lengths

    def prepare_output(self, sample):
        if self.root_known:
            bs, maxlenmin1, _ = sample.shape
            maxlen = maxlenmin1 + 1
            # Need to append zeroth node back
            head_row = sample.diagonal(0, -2, -1)
            prep_sample = sample - torch.diag_embed(head_row)
            prep_sample = torch.cat([head_row.unsqueeze(1), prep_sample], dim=1)
            head_col = sample.new_zeros((bs, maxlen, 1))
            prep_sample = torch.cat([head_col, prep_sample], dim=2)
            prep_sample = self.clean_diagonal(prep_sample)
            return prep_sample
        else:
            # Ghost root mode
            sample = self.clean_diagonal(sample)
            return sample

    def clean_diagonal(self, tensor):
        assert tensor.dim() == 3
        diag = tensor.diagonal(0, -2, -1)
        return tensor - torch.diag_embed(diag, 0, -2, -1)


### Projective Sampler
class ProjectiveSampler(DependencySampler):

    def __init__(self, mode, noise, tau, root_known, single_root):
        super(ProjectiveSampler, self).__init__(mode, noise, tau,
                                                root_known, single_root)

    def sample(self, A, lengths, mode):
        return _sampling.sample_projective(
                    A,
                    lengths,
                    mode=mode,
                    noise=self.noise,
                    tau=self.tau,
                    single_root=self.single_root)

### Non-Projective Sampler
class NonProjectiveSampler(DependencySampler):

    def __init__(self, mode, noise, tau, root_known, single_root, clip_or_rescale):
        super(NonProjectiveSampler, self).__init__(mode, noise, tau,
                                                    root_known, single_root,
                                                    clip_or_rescale)

    def sample(self, A, lengths, mode):
        return _sampling.sample_nonprojective(
                A,
                lengths,
                mode=mode,
                noise=self.noise,
                tau=self.tau,
                single_root=self.single_root,
                clip_or_rescale=self.clip_or_rescale)

    def forward_eval(self, A, lengths=None):
        r'''We override forward_eval (!)
        Usually, we could do same preparation as in train pass, but because
        sample_nonprojective hard for legacy reasons expects different signature
        than soft, we need to do a different preparation
        Chuli expects zeroth token is root + zero diagonal (no potentials are
        passed through diagonal)
        '''
        if self.root_known:
            # Zeroth token is assumed to be root, simply clean diagonal + then
            # done
            clean_A = self.clean_diagonal(A)
            sample = _sampling.sample_nonprojective(
                    clean_A,
                    lengths,
                    mode='hard',
                    noise=self.noise,
                    tau=1.0,
                    single_root=self.single_root)
            sample = self.clean_diagonal(sample)
        else:
            # For Chuli Ghost Form, we need to explicitly create the ghost node
            bs, maxlen, _ = A.shape
            head_row = A.diagonal(0, -2, -1)
            ghostA = torch.cat([head_row.unsqueeze(1), A], dim=1)
            head_col = A.new_zeros((bs, maxlen+1, 1))
            ghostA = torch.cat([head_col, ghostA], dim=2)
            cleanA = self.clean_diagonal(ghostA)
            if lengths is not None:
                ghost_lengths = lengths + 1
            else:
                ghost_lengths = None
            sample = _sampling.sample_nonprojective(
                        cleanA,
                        ghost_lengths,
                        mode='hard',
                        noise=self.noise,
                        tau=1.0,
                        single_root=self.single_root)
            # Now we need to cut out the ghost node again
            sample = sample[:, 1:, 1:]
            sample = self.clean_diagonal(sample)
        return sample

#### Ground-Truth Sampler

class IdentitySampler(nn.Module):
    r''' Simply returns Arc matrix passed initially.
    No sampling.'''

    def __init__(self):
        super(IdentitySampler, self).__init__()

    def forward(self, A, lengths=None, training_mode=None):
        return A


### Spanning Tree Sampler
class SpanningTreeSampler(Sampler):

    def __init__(self, mode, tau):
        super(SpanningTreeSampler, self).__init__(mode, None, tau)

    def forward_train(self, A, lengths=None):
        return _sampling.sample_spanning_tree(
                    A, lengths, mode=self.mode, tau=self.tau)

    def forward_eval(self, A, lengths=None):
        return _sampling.sample_spanning_tree(
                    A, lengths, mode='hard', tau=self.tau)

### Spanning Tree Sampler
class TopKSampler(Sampler):

    def __init__(self, mode, tau, eps):
        super(TopKSampler, self).__init__(mode, None, tau)
        self.eps = eps

    def forward_train(self, A, lengths=None):
        return _sampling.sample_top_n_minus_one(
                    A, lengths, mode=self.mode, tau=self.tau, eps=self.eps)

    def forward_eval(self, A, lengths=None):
        return _sampling.sample_spanning_tree(
                    A, lengths, mode='hard', tau=self.tau)


### Independent Sampler
class IndependentSampler(Sampler):

    def __init__(self, type, mode, tau):
        super(IndependentSampler, self).__init__(mode, None, tau)
        assert type in set(['arc', 'edge'])
        self.type = type

    def forward_train(self, A, lengths=None):
        sample = self.sample(A, lengths, self.mode)
        return sample

    def forward_eval(self, A, lengths=None):
        sample = self.sample(A, lengths, 'hard')
        return sample

    def sample(self, A, lengths, mode):
        if self.type == 'arc':
            sample = _sampling.sample_independent_arcs(
                A,
                lengths,
                mode=mode,
                tau=self.tau)
        elif self.type == 'edge':
            sample = _sampling.sample_indepdentent_edges(
                A,
                lengths,
                mode=mode,
                tau=self.tau)
        else:
            raise ValueError
        return sample

    
    
# Automasking for binary samplers
def mask(sample, lengths):
    maxlen = sample.shape[1]
    diag_mask = torch.eye(maxlen, device=sample.device, dtype=bool).unsqueeze(0)
    sample = sample.masked_fill(diag_mask, 0.0)
    arcmask = arcmask_from_lengths(sample, lengths)
    sample = sample.masked_fill(arcmask, 0.0)
    return sample


class BinaryIndependentSampler(Sampler):

    def __init__(self, mode, tau):
        super(BinaryIndependentSampler, self).__init__(mode, None, tau)

    def forward_train(self, A, lengths=None):
        sample = mask(self.sample(A, lengths, self.mode), lengths)
        return sample

    def forward_eval(self, A, lengths=None):
        sample = mask(self.sample(A, lengths, 'hard'), lengths)
        return sample