import torch
import torch.nn as nn
import listops.model_modules.computer as _computer

OUT_DIM = 10

def get_decoder(dec_str):

    if 'mlp' in dec_str:
        num_hidden, in_dim, hidden_dim = parse_mlp_str(dec_str)
        decoder = get_mlp_decoder(num_hidden, in_dim, hidden_dim)
    else:
        raise NotImplementedError

    return decoder

def parse_mlp_str(dec_str):
    _, num_hidden, in_dim, hidden_dim = dec_str.split('_')
    return int(num_hidden), int(in_dim), int(hidden_dim)

def get_mlp_decoder(num_hidden, in_dim, hidden_dim):

    class Decoder(nn.Module):
        def __init__(self, num_hidden, in_dim, hidden_dim):
            super(Decoder, self).__init__()
            self.num_hidden = num_hidden
            self.in_dim = in_dim
            self.hidden_dim = hidden_dim
            modules = []
            modules.append(nn.Linear(in_dim, hidden_dim))
            modules.append(nn.ReLU())
            for _ in range(num_hidden):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_dim, OUT_DIM))
            self.net = nn.Sequential(*modules)

        def forward(self, compute):
            bs, feat_dim = compute.shape
            assert feat_dim == self.in_dim
            return self.net(compute)

    return Decoder(num_hidden, in_dim, hidden_dim)
