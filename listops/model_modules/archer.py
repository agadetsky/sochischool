import listops.data_processing.python.loading as _loading
import torch
import torch.nn as nn

VOCAB_SIZE = len(_loading.word_to_ix)
PADDING_IDX = _loading.word_to_ix['<PAD>']

def get_archer(archer_str):
    r'''
    Any Archer should have the following signature:
    Input: x, arcs, lengths
    Output: arc_logits
    '''
    if archer_str == 'gt':
        archer = get_gt_archer()
    elif archer_str == 'uniform':
        archer = get_uniform_archer()
    elif 'transformer' in archer_str:
        num_layers, embd_dim, nhead, dim_feedforward = parse_transformer_str(archer_str)
        archer = get_transformer_archer(embd_dim, nhead, dim_feedforward, num_layers)
    elif 'lstm' in archer_str:
        bidirectional, num_layers, embd_dim, hidden_dim, do_prob =  parse_lstm_str(archer_str)
        archer = get_lstm_archer(embd_dim, hidden_dim, num_layers, bidirectional, do_prob)
    elif 'corro' in archer_str:
        embd_dim, hidden_dim, do_prob = parse_corro_str(archer_str)
        archer = get_corro_archer(embd_dim, hidden_dim, do_prob)
    elif 'uniform' == archer_str:
        archer = get_uniform_archer()
    else:
        raise NotImplementedError
    return archer

##################
### Ground-Truth
##################

def get_gt_archer():
    r'''
    Return Ground-Truth Arc Matrix
    '''
    class Archer(nn.Module):

        def __init__(self):
            super(Archer, self).__init__()

        def forward(self, x, arcs, lengths):
            return arcs

    return Archer()

##################
### Uniform
##################

def get_uniform_archer():
    r'''
    Return Uniform Arc Matrix (can be any constant, we choose zero)

    '''
    class Archer(nn.Module):

        def __init__(self):
            super(Archer, self).__init__()

        def forward(self, x, arcs, lengths):
            return torch.zeros_like(arcs) # just any constant

    return Archer()

##################
### LSTM
##################

def parse_lstm_str(archer_str):
     lstm, num_layers, embd_dim, hidden_dim, do_prob = archer_str.split('_')
     bidirectional = ('bi' in lstm)
     return bidirectional, int(num_layers), int(embd_dim), int(hidden_dim), float(do_prob)

def get_lstm_archer(embd_dim, hidden_dim, num_layers, bidirectional, do_prob):

    class Archer(nn.Module):

        def __init__(self, vocab_size, padding_idx,
                        embd_dim, hidden_dim, num_layers, bidirectional,
                        do_prob):
            super(Archer, self).__init__()
            self.vocab_size = vocab_size
            self.padding_idx = padding_idx
            self.embd_dim = embd_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.do_prob = do_prob

            self.embd = nn.Embedding(vocab_size, embd_dim, padding_idx)
            self.head_lstm = nn.LSTM(
                input_size=embd_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1,
                bidirectional=bidirectional)
            self.head_dropout = nn.Dropout(do_prob)
            self.modif_lstm = nn.LSTM(
                input_size=embd_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1,
                bidirectional=bidirectional)
            self.modif_dropout = nn.Dropout(do_prob)


        def forward(self, x, arcs, lengths):
            bs, maxlen = x.shape
            # arc_logits computation
            embd = self.embd(x)
            packed = torch.nn.utils.rnn.pack_padded_sequence(embd, lengths.cpu(),
                            batch_first=True, enforce_sorted=True)
            # Head LSTM
            head_out, _ = self.head_lstm(packed)
            head_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(head_out,
                            batch_first=True)
            head_padded = self.head_dropout(head_padded)
            # Modifier LSTM
            modif_out, _ = self.modif_lstm(packed)
            modif_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(modif_out,
                            batch_first=True)
            modif_padded = self.modif_dropout(modif_padded)

            arc_logits = torch.matmul(head_padded, modif_padded.permute(0, 2, 1))
            return arc_logits

    return Archer(VOCAB_SIZE, PADDING_IDX,
                    embd_dim, hidden_dim, num_layers, bidirectional, do_prob)

##################
### Transformer
##################

def parse_transformer_str(archer_str):
     _, num_layers, embd_dim, nhead, dim_feedforward = archer_str.split('_')
     return int(num_layers), int(embd_dim), int(nhead), int(dim_feedforward)

def get_transformer_archer(embd_dim, nhead, dim_feedforward, num_layers):

    class Archer(nn.Module):

        def __init__(self, vocab_size, embd_dim, padding_idx,
                        nhead, dim_feedforward, num_layers):
            super(Archer, self).__init__()
            self.vocab_size = vocab_size
            self.embd_dim = embd_dim
            self.padding_idx = padding_idx
            self.nhead = nhead
            self.dim_feedforward = dim_feedforward
            self.num_layers = num_layers

            self.embd = nn.Embedding(vocab_size, embd_dim, padding_idx)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embd_dim,
                                nhead=nhead, dim_feedforward=dim_feedforward)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        def forward(self, x, arcs, lengths):
            maxlen = x.shape[-1]
            key_padding_mask = torch.arange(maxlen)[None, :] >= lengths[:, None]
            embd = self.embd(x.transpose(0, 1)) # transfomer expects inputs S x N x *
            transformed = self.encoder(embd, src_key_padding_mask=key_padding_mask)
            arc_logits = torch.matmul(transformed.permute(1, 0, 2), transformed.permute(1, 2, 0))
            return arc_logits

    return Archer(VOCAB_SIZE, embd_dim, PADDING_IDX,
                    nhead, dim_feedforward, num_layers)

##################
### Corro
##################

def parse_corro_str(archer_str):
     _, embd_dim, hidden_dim, do_prob = archer_str.split('_')
     return int(embd_dim), int(hidden_dim), float(do_prob)

def get_corro_archer(embd_dim, hidden_dim, do_prob):

    class Archer(nn.Module):

        def __init__(self, vocab_size, padding_idx,
                        embd_dim, hidden_dim,
                        do_prob):
            super(Archer, self).__init__()
            self.vocab_size = vocab_size
            self.padding_idx = padding_idx
            self.embd_dim = embd_dim
            self.hidden_dim = hidden_dim
            self.do_prob = do_prob

            self.embd = nn.Embedding(vocab_size, embd_dim, padding_idx)

            self.lstm = nn.LSTM(
                input_size=embd_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True)

            self.head_mlp = nn.Sequential(
                                nn.Dropout(do_prob),
                                nn.Linear(2*hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim))

            self.mod_mlp = nn.Sequential(
                                nn.Dropout(do_prob),
                                nn.Linear(2*hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim))


        def forward(self, x, arcs, lengths):
            bs, maxlen = x.shape
            # arc_logits computation
            embd = self.embd(x)
            packed = torch.nn.utils.rnn.pack_padded_sequence(embd, lengths,
                            batch_first=True, enforce_sorted=True)
            # Head LSTM
            lstm_out, _ = self.lstm(packed)
            lstm_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,
                                    batch_first=True)
            head_vec = self.head_mlp(lstm_padded)
            mod_vec = self.mod_mlp(lstm_padded)
            arc_logits = torch.matmul(head_vec, mod_vec.permute(0, 2, 1))
            return arc_logits

    return Archer(VOCAB_SIZE, PADDING_IDX,
                    embd_dim, hidden_dim,
                    do_prob)
