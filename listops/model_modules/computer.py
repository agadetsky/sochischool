import copy
import torch
import torch.nn as nn
import listops.data_processing.python.loading as _loading
import listops.model_modules.archer as _archer
import listops.model_modules.gnn_modules as gnn_modules
import listops.model_modules.gnn_attention as _gnn_attention

VOCAB_SIZE = len(_loading.word_to_ix)
PADDING_IDX = _loading.word_to_ix['<PAD>']


def get_computer(comp_str, num_layers=None):
    r'''
    Any Encoder should have the following signature:
    Input: num_tokens, gt_arcmatrix (except for identity do not use!)
    Output: token_representations, arc_matrix
    '''
    if 'merge' in comp_str:
        embd_dim, hidden_dim, out_dim = parse_merge_str(comp_str)
        computer = get_merge_computer(embd_dim, hidden_dim, out_dim)
    elif 'lstm' in comp_str:
        bidirectional, num_layers, embd_dim, hidden_dim, out_dim =  parse_lstm_str(comp_str)
        computer = get_lstm_computer(embd_dim, hidden_dim, out_dim, num_layers, bidirectional)
    elif 'transformer' in comp_str:
        num_layers, embd_dim, nhead, dim_feedforward, out_dim = parse_transformer_str(comp_str)
        computer = get_transformer_computer(embd_dim, out_dim, nhead, dim_feedforward, num_layers)
    elif 'kipf_mlp' in comp_str:
        num_layers, embd_dim, msg_hid, dropout_prob = parse_kipf_mlp_merge_str(comp_str)
        computer = get_kipf_mlp_gnn(num_layers, embd_dim, msg_hid, dropout_prob)
    elif "kipf_rnn" in comp_str:
        num_layers, embd_dim, n_hid = parse_kipf_rnn_merge_str(comp_str)
        computer = get_kipf_rnn_gnn(num_layers, embd_dim, n_hid)
    elif 'gcn' in comp_str:
        num_layers, embd_dim, hidden_dim, dropout_prob, force_single, laynorm = parse_gcn_str(comp_str)
        computer = get_gcn_computer(num_layers, embd_dim, hidden_dim, force_single, laynorm, dropout_prob)
    elif 'gatt' in comp_str: # gatt or gattf
        num_layers, embd_dim, nhead, force_single = parse_gatt_str(comp_str)
        computer = get_gatt_computer(num_layers, embd_dim, nhead, force_single)
    else:
        raise NotImplementedError

    return computer

####################################
#### Merge
####################################

def parse_merge_str(comp_str):
    _, embd_dim, hidden_dim, out_dim = comp_str.split('_')
    return int(embd_dim), int(hidden_dim), int(out_dim)

def get_merge_computer(embd_dim, hidden_dim, out_dim):
    r'''Simply ignores arc structure instead creates bag of tokens and puts
    through a one hidden layer neural network'''

    class Computer(nn.Module):

        def __init__(self, vocab_size, padding_idx,
                        embd_dim, hidden_dim, out_dim):
            super(Computer, self).__init__()
            self.embd_dim = embd_dim
            self.hidden_dim = hidden_dim
            self.out_dim = out_dim
            self.embd = nn.Embedding(vocab_size, embd_dim, padding_idx)
            self.net = nn.Sequential(
                        nn.Linear(embd_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, out_dim))

        def forward(self, x, arcs, lengths):
            # Padded entries will be zero by virtue of embedding
            embd = self.embd(x)
            mean = embd.sum(1) / lengths.unsqueeze(1)
            comp = self.net(mean)
            return comp

    return Computer(VOCAB_SIZE, PADDING_IDX,
                        embd_dim, hidden_dim, out_dim)

##################
### LSTM
##################

def parse_lstm_str(comp_str):
     lstm, num_layers, embd_dim, hidden_dim, out_dim = comp_str.split('_')
     bidirectional = ('bi' in lstm)
     return bidirectional, int(num_layers), int(embd_dim), int(hidden_dim), int(out_dim)

def get_lstm_computer(embd_dim, hidden_dim, out_dim, num_layers, bidirectional):

    class Computer(nn.Module):

        def __init__(self, vocab_size, padding_idx,
                        embd_dim, hidden_dim, out_dim, num_layers, bidirectional):
            super(Computer, self).__init__()
            self.vocab_size = vocab_size
            self.padding_idx = padding_idx
            self.embd_dim = embd_dim
            self.hidden_dim = hidden_dim
            self.out_dim = out_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            self.embd = nn.Embedding(vocab_size, embd_dim, padding_idx)
            self.lstm = nn.LSTM(
                input_size=embd_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1,
                bidirectional=bidirectional)
            self.linear = nn.Linear(self.hidden_dim * self.num_directions, out_dim) # Extra linear to make sure we pass the same
            # dimension to decoder no matter if bidirectional or not


        def forward(self, x, arcs, lengths):
            bs, maxlen = x.shape
            # arc_logits computation
            embd = self.embd(x)
            packed = torch.nn.utils.rnn.pack_padded_sequence(embd, lengths,
                            batch_first=True, enforce_sorted=True)
            _, (h_n, _) = self.lstm(packed)
            h_n = h_n.reshape(self.num_layers, self.num_directions, bs, self.hidden_dim)
            h_n = h_n.transpose(1, 2)[-1] # Get last layer, batch_size first
            h_n = h_n.reshape(bs, -1) # flatten
            out = self.linear(h_n) # for consistency
            return out

    return Computer(VOCAB_SIZE, PADDING_IDX,
                    embd_dim, hidden_dim, out_dim, num_layers, bidirectional)

##################
### Transformer
##################

def parse_transformer_str(comp_str):
     _, num_layers, embd_dim, nhead, dim_feedforward, out_dim = comp_str.split('_')
     return int(num_layers), int(embd_dim), int(nhead), int(dim_feedforward), int(out_dim)

def get_transformer_computer(embd_dim, out_dim, nhead, dim_feedforward, num_layers):

    class Computer(nn.Module):

        def __init__(self, vocab_size, embd_dim, padding_idx,
                        out_dim, nhead, dim_feedforward, num_layers):
            super(Computer, self).__init__()
            self.vocab_size = vocab_size
            self.embd_dim = embd_dim
            self.padding_idx = padding_idx
            self.nhead = nhead
            self.dim_feedforward = dim_feedforward
            self.num_layers = num_layers
            self.out_dim = out_dim

            self.embd = nn.Embedding(vocab_size, embd_dim, padding_idx)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embd_dim,
                                nhead=nhead, dim_feedforward=dim_feedforward)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            self.linear = nn.Linear(embd_dim, out_dim)

        def forward(self, x, arcs, lengths):
            maxlen = x.shape[-1]
            key_padding_mask = torch.arange(maxlen)[None, :] >= lengths[:, None]
            embd = self.embd(x.transpose(0, 1)) # transfomer expects inputs S x N x *
            transformed = self.encoder(embd, src_key_padding_mask=key_padding_mask)
            # TODO: Find out how transformer has treated the masking?
            transformed = transformed.permute(1, 0, 2)[:, 0, :] # Extract root representation
            out = self.linear(transformed)
            return out

    return Computer(VOCAB_SIZE, embd_dim, PADDING_IDX,
                    out_dim, nhead, dim_feedforward, num_layers)

####################################
#### MLP GNN (NRI)
####################################
def parse_kipf_mlp_merge_str(comp_str):
    _, _, num_layers, embd_dim, msg_hid, dropout_prob = comp_str.split('_')
    return int(num_layers), int(embd_dim), int(msg_hid), float(dropout_prob)

def get_kipf_mlp_gnn(num_layers, embd_dim, msg_hid, do_prob, edge_types=1):
    return gnn_modules.KipfMLPGNN(
        VOCAB_SIZE, PADDING_IDX, embd_dim, num_layers, msg_hid, edge_types, do_prob)

def parse_kipf_rnn_merge_str(comp_str):
    _, _, num_layers, embd_dim, n_hid = comp_str.split('_')
    return int(num_layers), int(embd_dim), int(n_hid)

def get_kipf_rnn_gnn(num_layers, embd_dim, n_hid, edge_types=1, do_prob=0.):
    return gnn_modules.KipfRNNGNN(
        VOCAB_SIZE, PADDING_IDX, embd_dim, num_layers, edge_types,
        n_hid, do_prob)

####################################
#### GCN (from Corro and Titov, 2019)
####################################

def parse_gcn_str(comp_str):
    gcn_str, num_layers, embd_dim, hidden_dim, dropout_prob = comp_str.split('_')
    force_single = 'f' in gcn_str
    laynorm = 'l' in gcn_str
    return int(num_layers), int(embd_dim), int(hidden_dim), float(dropout_prob), force_single, laynorm

def get_gcn_computer(num_layers, embd_dim, hidden_dim, force_single, laynorm,
                     do_prob):

    class Computer(nn.Module):

        def __init__(self, vocab_size, padding_idx,
                     num_layers, embd_dim, hidden_dim,
                     force_single, laynorm, do_prob):
            super(Computer, self).__init__()
            self.num_layers = num_layers
            self.embd_dim = embd_dim
            self.hidden_dim = hidden_dim
            self.force_single = force_single
            self.laynorm = laynorm
            self.embd = nn.Embedding(vocab_size, embd_dim, padding_idx)
            # Self-connection.
            self_mlp = nn.Sequential(
                nn.Linear(embd_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=do_prob),
                nn.Linear(hidden_dim, embd_dim))
            # Children -> Parent connection.
            child_mlp = nn.Sequential(
                nn.Linear(embd_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=do_prob),
                nn.Linear(hidden_dim, embd_dim))

            self_layers = []
            child_layers = []

            if force_single:
                # Re-use the same layer multiple times
                self_layers = [self_mlp] * num_layers
                child_layers = [child_mlp] * num_layers
            else:
                # Use separate instances of this layer
                self_layers = _gnn_attention.get_clones(self_mlp, num_layers)
                child_layers = _gnn_attention.get_clones(child_mlp, num_layers)

            self.self_layers = nn.modules.container.ModuleList(self_layers)
            self.child_layers = nn.modules.container.ModuleList(child_layers)

            if laynorm:
                self.laynorms = nn.modules.container.ModuleList(
                    _gnn_attention.get_clones(nn.LayerNorm(embd_dim), num_layers))
            else:
                self.laynorms = [None] * num_layers

        def forward(self, x, arcs, lengths):
            # Padded entries will be zero by virtue of embedding
            E = self.embd(x)  # (bs, n, embd_dim).
            arcs = arcs.float()
            for f, h, lnorm in zip(self.self_layers, self.child_layers, self.laynorms):
                E = (#nn.functional.softmax(
                    f(E) +
                    # torch.matmul(arcs.transpose(1, 2), self.g(E)) +
                    torch.matmul(arcs, h(E))
                    )
                if self.laynorm:
                    E = lnorm(E)

            return E[:, 0, :]

    return Computer(VOCAB_SIZE, PADDING_IDX,
                    num_layers, embd_dim, hidden_dim,
                    force_single, laynorm, do_prob)


####################################
#### Graph Attention
####################################

def parse_gatt_str(comp_str):
    gatt, num_layers, embd_dim, nhead = comp_str.split('_')
    return int(num_layers), int(embd_dim), int(nhead), ('f' in gatt)

def get_gatt_computer(num_layers, embd_dim, nhead, force_single):

    class Computer(nn.Module):

        def __init__(self, vocab_size, padding_idx,
                num_layers, embd_dim, nhead, force_single=True):

            super(Computer, self).__init__()
            self.vocab_size = vocab_size
            self.embd_dim = embd_dim
            self.padding_idx = padding_idx
            self.num_layers = num_layers
            self.embd_dim = embd_dim
            self.nhead = nhead
            self.force_single = force_single

            # Initial Node embedding
            self.embd = nn.Embedding(vocab_size, embd_dim)

            # Attention Layer(s)
            attn_layer = _gnn_attention.SimpleGraphAttentionLayer(
                            vocab_size, embd_dim, nhead)
            if force_single:
                # Re-use the same layer multiple times
                layers = [attn_layer] * num_layers
            else:
                # Use separate instances of this layer
                layers = _gnn_attention.get_clones(attn_layer, num_layers)

            self.layers = nn.modules.container.ModuleList(layers)

        def forward(self, x, arcs, lengths):
            nodes = self.embd(x)

            for mod in self.layers:
                nodes = mod(x, nodes, arcs, lengths)

            out = nodes[:, 0, :] # root node embedding.
            return out

    return Computer(VOCAB_SIZE, PADDING_IDX,
                        num_layers, embd_dim, nhead, force_single)
