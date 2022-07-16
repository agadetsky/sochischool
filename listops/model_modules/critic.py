import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import listops.data_processing.python.loading as _loading
import listops.model_modules.gnn_modules as gnn_modules

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

VOCAB_SIZE = len(_loading.word_to_ix)
PADDING_IDX = _loading.word_to_ix['<PAD>']


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_critic(critic_str):
    r'''
    Any Encoder should have the following signature:
    Input: num_tokens, gt_arcmatrix (except for identity do not use!)
    Output: token_representations, arc_matrix
    '''

    critic = None
    if 'none' in critic_str:
        return None
    elif 'kipf_mlp' in critic_str:
        num_layers, embd_dim, msg_hid, dropout_prob = parse_kipf_mlp_merge_str(critic_str)
        gnn = get_kipf_mlp_gnn(num_layers, embd_dim, msg_hid, dropout_prob)
        critic = RelaxCriticGNN(gnn, embd_dim, msg_hid)
    elif 'lstm' in critic_str:
        bidirectional, num_layers, embd_dim, hidden_dim, do_prob = parse_lstm_str(critic_str)
        critic = get_lstm_critic(bidirectional, num_layers, embd_dim, hidden_dim, do_prob)
    else:
        raise NotImplementedError

    return critic

def parse_lstm_str(lstm_str):
     lstm, num_layers, embd_dim, hidden_dim, do_prob = lstm_str.split('_')
     bidirectional = ('bi' in lstm)
     return bidirectional, int(num_layers), int(embd_dim), int(hidden_dim), float(do_prob)

def parse_kipf_mlp_merge_str(critic_str):
    _, _, num_layers, embd_dim, msg_hid, dropout_prob = critic_str.split('_')
    return int(num_layers), int(embd_dim), int(msg_hid), float(dropout_prob)

def get_lstm_critic(bidirectional, num_layers, embd_dim, hidden_dim, do_prob, max_dim=50):
    return RelaxCriticLSTM(VOCAB_SIZE, PADDING_IDX, embd_dim, hidden_dim, num_layers, bidirectional, do_prob, max_dim)

def get_kipf_mlp_gnn(num_layers, embd_dim, msg_hid, do_prob, edge_types=1):
    return gnn_modules.KipfMLPGNN(
        VOCAB_SIZE, PADDING_IDX, embd_dim, num_layers, msg_hid, edge_types, do_prob)


class RelaxCriticGNN(nn.Module):

    def __init__(self, gnn, in_dim, hidden_dim):
        super(RelaxCriticGNN, self).__init__()
        self.gnn = gnn
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, x, lengths_x):
        mask = torch.arange(z.shape[1])[None, :].to(device) < lengths_x[:, None]
        mask = mask[:, None, :] * mask[:, :, None]
        z = (z - z.mean()) / (z.std() + 1e-12)
        return self.mlp(self.gnn(x, z, lengths_x))


class RelaxCriticLSTM(nn.Module):

    def __init__(self, vocab_size, padding_idx,
                    embd_dim, hidden_dim, num_layers, bidirectional,
                    do_prob, max_dim):
        super(RelaxCriticLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.embd_dim = embd_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.do_prob = do_prob

        self.embd = nn.Embedding(vocab_size, embd_dim, padding_idx)
        self.lstm = nn.LSTM(
            input_size=embd_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=bidirectional)
        self.dropout = nn.Dropout(do_prob)
        self.max_dim = max_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + max_dim ** 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, x, lengths_x):
        bs, maxlen = x.shape
        z = (z - z.mean()) / (z.std() + 1e-12)
        embd = self.embd(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embd, lengths_x.cpu(),
                            batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=embd.size(1))

        z_in = torch.zeros(bs, self.max_dim, self.max_dim).to(device)
        z_in[:, :z.shape[1], :z.shape[1]] += z

        z_in = z_in.view(bs, -1)
        out = out[torch.arange(bs), lengths_x - 1, :]
        mlp_in = torch.cat((out, z_in), dim=-1)
        mlp_out = self.mlp(mlp_in)
        return mlp_out

