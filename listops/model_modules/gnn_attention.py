import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_kpm(maxlen, lengths):
    device = lengths.device
    kpm = torch.arange(maxlen)[None, :].to(device) >= lengths[:, None]
    return kpm

class SimpleGraphAttentionLayer(nn.Module):
    r'''
    This is simple, because we use the same representation of a node (node
    embedding) for both keys and values. This is one reason why embd_dim should
    be larger than the vocabulary size. This also fails to capture interactions
    between the children of a node directly through the keys. Maybe this is not
    too restrictive, otherwise we may want to explore a more sophisticated
    mechanism to compute keys.

    Some key ideas behind this module are:

    1. for MIN, MED, MAX operators, the representation (node embedding) of an
    operator is the value of one of its children. Hence, a sophisticated averaging
    mechanism over children makes sense for these operators. We parameterise
    different queries for each of these operators.

    2. Operators should always evaluate to a representation of their children
    while digits should always evaluate to their original representation. By
    using a gate_embd with a single parameter which is directly determined by
    the identity of a node (through its index in vocabulary), we should be able
    to learn to a) ignore the attention output for digits and simply return the
    original embedding of the digit, b) for an operator to only return the output
    of the attention head (expression it evaluates to). This behaviour is not
    harmful, because when stacking SimpleGraphAttentionLayer, in the next layer
    the attn_output of an operator again only depends on the node embeddings of
    its children (query head is directly determined by node identity not by node
    embedding). It does not depend on its own node_embedding which was returned
    by the previous layer.

    3. No parameters in this module (and in the original computaiton graph to
    process a data string) have a depth-dependent meaning. While this may be true
    for other architectures, it is an appealing property. It suggests that one
    should be able to re-use this exact layer (parameter sharing) when growing a
    deeper Graph Parser.

    4. Open question:
        - Does it work on a dataset with only MIN-MED-MAX operators?
        - Do we need more sophisticated keys?
        - Can we augment the architecture to resolve SUMMOD?
        - Is their any case here for using more than one attention head?

    '''

    def __init__(self, vocab_size, embd_dim, nhead=1, neginf=-100):
        super(SimpleGraphAttentionLayer, self).__init__()
        self.query_embd = nn.Embedding(vocab_size, embd_dim) # head means query node
        self.attention = nn.MultiheadAttention(embd_dim, nhead)
        self.gate_embd = nn.Embedding(vocab_size, 1)
        self.neginf = neginf

    def forward(self, x, node_embd, arcs, lengths):
        # Convenience
        bsz, maxlen = x.shape
        bsz, maxlen, embd_dim = node_embd.shape
        # Attention
        query = self.query_embd(x).transpose(0, 1) # maxlen x bsz x embd_dim
        key = node_embd.transpose(0, 1)
        value = node_embd.transpose(0, 1)
        # attention mask is added to dot-product weights before softmax
        mask = self.neginf + (arcs * -self.neginf) # bsz x maxlen x maxlen
        kpm = get_kpm(maxlen, lengths)
        # import pdb;pdb.set_trace()
        attn_out, _ = self.attention(query, key, value,
                        key_padding_mask=kpm, attn_mask=mask)
        attn_out = attn_out.transpose(0, 1) # bsz x maxlen x embd_dim
        # Gating
        gates = self.gate_embd(x).sigmoid() # batchsize x max_len x 1
        node_out = node_embd * gates + attn_out + (1 - gates)
        # Return
        return node_out

def get_clones(module, N):
    # wratp into module list afterwards
    return [copy.deepcopy(module) for i in range(N)]
