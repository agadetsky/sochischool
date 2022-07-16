import listops.model_modules as _modules
import listops.model_modules.archer as _archer
import listops.model_modules.sampler as _sampler
import listops.model_modules.computer as _computer
import listops.model_modules.decoder as _decoder
import listops.model_modules.critic as _critic
import torch
import torch.nn as nn
# This is the base model skeleton
# To define a model, settle on an archer, sampler, computer, decoder,


def get_school_model(sampler):
    m = Model(
        archer_str="lstm_1_60_60_0.1",
        sampler_str=sampler,
        comp_str="kipf_mlp_5_60_60_0.1",
        dec_str="mlp_1_60_60",
        tau=1.0 # dummy, not really used, tau is defined in Sampler
    )
    return m

def get_model(archer_str, sampler_str, comp_str, dec_str, critic_str, tau, reinforce):
    if not reinforce:
        return Model(archer_str, sampler_str, comp_str, dec_str, tau)

    return RModel(archer_str, sampler_str, comp_str, dec_str, critic_str, tau)


class ModelBase(nn.Module):

    def __init__(self, sampler_str, comp_str, dec_str, tau):
        super().__init__()
        if isinstance(sampler_str, str):
            self.sampler  = _sampler.get_sampler(sampler_str, tau)
        else:
            self.sampler = sampler_str
        self.computer = _computer.get_computer(comp_str)
        self.decoder  = _decoder.get_decoder(dec_str)
        self.tau = tau


class Model(nn.Module):

    def __init__(self, archer_str, sampler_str, comp_str, dec_str, tau):
        super().__init__()
        self.base = ModelBase(sampler_str, comp_str, dec_str, tau)
        self.archer  = _archer.get_archer(archer_str)
        self.sample = None
        self.critic = None

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, weights):
        self.load_state_dict(weights)

    def forward(self, x, arcs, lengths, training_mode=None):
        arc_logits = self.archer(x, arcs, lengths) # need to parse arcs for gt archer
        sample = self.base.sampler(arc_logits, lengths, training_mode=training_mode) # sampler never depends on tokens
        comp = self.base.computer(x, sample, lengths) # computer only depends on sample, not on arcs
        pred = self.base.decoder(comp)
        # Save for easy access in .eval() to inspect sample
        self.sample = sample.detach()
        return pred

class Model_(nn.Module):

    def __init__(self, archer_str, sampler_str, comp_str, dec_str, tau):

        super().__init__()
        self.archer  = _archer.get_archer(archer_str)
        self.sampler  = _sampler.get_sampler(sampler_str, tau)
        self.computer = _computer.get_computer(comp_str)
        self.decoder  = _decoder.get_decoder(dec_str)
        self.tau = tau
        self.sample = None # convenience attribute

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    #@profile
    def forward(self, x, arcs, lengths, training_mode=None):
        arc_logits = self.archer(x, arcs, lengths) # need to parse arcs for gt archer
        sample = self.sampler(arc_logits, lengths, training_mode=training_mode) # sampler never depends on tokens
        comp = self.computer(x, sample, lengths) # computer only depends on sample, not on arcs
        pred = self.decoder(comp)
        # Save for easy access in .eval() to inspect sample
        self.sample = sample.detach()
        return pred


class RModelBase(nn.Module):

    def __init__(self, sampler_str, comp_str, dec_str, tau):
        super().__init__()
        self.sampler  = _sampler.get_sampler(sampler_str, tau)
        self.computer = _computer.get_computer(comp_str)
        self.decoder  = _decoder.get_decoder(dec_str)
        self.tau = tau
        self.sample = None # convenience attribute


class RModel(nn.Module):
    def __init__(self, archer_str, sampler_str, comp_str, dec_str, critic_str, tau):
        super().__init__()
        self.base = RModelBase(sampler_str, comp_str, dec_str, tau)
        self.archer  = _archer.get_archer(archer_str)
        self.critic = _critic.get_critic(critic_str)
        self.sample = None

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, weights):
        self.load_state_dict(weights)

    #@profile
    def forward(self, x, arcs, lengths, training_mode=None):
        arc_logits = self.archer(x, arcs, lengths) # need to parse arcs for gt archer
        sample, stats, z = self.base.sampler(arc_logits, lengths) # sampler never depends on tokens
        sample = sample.detach()
        comp = self.base.computer(x, sample, lengths) # computer only depends on sample, not on arcs
        pred = self.base.decoder(comp)
        # Save for easy access in .eval() to inspect sample
        self.sample = sample.detach()
        return pred, arc_logits, sample, stats, z


    def forward_plus(self, x, arcs, lengths, K=1, training_mode=None):
        arc_logits = self.archer(x, arcs, lengths) # need to parse arcs for gt archer
        sample, stats, z = self.base.sampler(arc_logits, lengths) # sampler never depends on tokens

        sample = sample.detach()
        comp = self.base.computer(x, sample, lengths) # computer only depends on sample, not on arcs
        pred = self.base.decoder(comp)

        x_K = x.repeat(K, 1)
        arc_logits_K = arc_logits.repeat(K, 1, 1)
        lengths_K = lengths.repeat(K)
        sample_K, stats_K, z_K = self.base.sampler(arc_logits_K, lengths_K)
        sample_K = sample_K.detach()
        comp_K = self.base.computer(x_K, sample_K, lengths_K)
        pred_K = self.base.decoder(comp_K)

        # Save for easy access in .eval() to inspect sample
        self.sample = sample.detach()
        return pred, arc_logits, sample, stats, z, pred_K

    #@profile
    def forward_plus_mean(self, x, arcs, lengths, K=1, training_mode=None):
        arc_logits = self.archer(x, arcs, lengths)
        x_all = x.repeat(K + 1, 1)
        arc_logits_all = arc_logits.repeat(K + 1, 1, 1)
        lengths_all = lengths.repeat(K + 1)

        sample_all, stats_all, z_all = self.base.sampler(arc_logits_all, lengths_all)
        sample_all = sample_all.detach()
        comp_all = self.base.computer(x_all, sample_all, lengths_all)
        pred_all = self.base.decoder(comp_all)

        return pred_all, arc_logits, arc_logits_all, sample_all, stats_all, z_all


class RModel_(nn.Module):

    def __init__(self, archer_str, sampler_str, comp_str, dec_str, tau):

        super().__init__()
        self.archer  = _archer.get_archer(archer_str)
        self.sampler  = _sampler.get_sampler(sampler_str, tau)
        self.computer = _computer.get_computer(comp_str)
        self.decoder  = _decoder.get_decoder(dec_str)
        self.critic = None
        self.tau = tau
        self.sample = None # convenience attribute

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    #@profile
    def forward(self, x, arcs, lengths, training_mode=None):
        arc_logits = self.archer(x, arcs, lengths) # need to parse arcs for gt archer
        sample, stats, z = self.sampler(arc_logits, lengths) # sampler never depends on tokens
        sample = sample.detach()
        comp = self.computer(x, sample, lengths) # computer only depends on sample, not on arcs
        pred = self.decoder(comp)
        # Save for easy access in .eval() to inspect sample
        self.sample = sample.detach()
        return pred, arc_logits, sample, stats, z


    def forward_plus(self, x, arcs, lengths, K=1, training_mode=None):
        arc_logits = self.archer(x, arcs, lengths) # need to parse arcs for gt archer
        sample, stats, z = self.sampler(arc_logits, lengths) # sampler never depends on tokens

        sample = sample.detach()
        comp = self.computer(x, sample, lengths) # computer only depends on sample, not on arcs
        pred = self.decoder(comp)

        x_K = x.repeat(K, 1)
        arc_logits_K = arc_logits.repeat(K, 1, 1)
        lengths_K = lengths.repeat(K)
        sample_K, stats_K, z_K = self.sampler(arc_logits_K, lengths_K)
        sample_K = sample_K.detach()
        comp_K = self.computer(x_K, sample_K, lengths_K)
        pred_K = self.decoder(comp_K)

        # Save for easy access in .eval() to inspect sample
        self.sample = sample.detach()
        return pred, arc_logits, sample, stats, z, pred_K


    def forward_plus_mean(self, x, arcs, lengths, K=1, training_mode=None):
        arc_logits = self.archer(x, arcs, lengths)
        x_all = x.repeat(K + 1, 1)
        arc_logits_all = arc_logits.repeat(K + 1, 1, 1)
        lengths_all = lengths.repeat(K + 1)

        sample_all, stats_all, z_all = self.sampler(arc_logits_all, lengths_all)
        sample_all = sample_all.detach()
        comp_all = self.computer(x_all, sample_all, lengths_all)
        pred_all = self.decoder(comp_all)

        return pred_all, arc_logits, arc_logits_all, sample_all, stats_all, z_all


