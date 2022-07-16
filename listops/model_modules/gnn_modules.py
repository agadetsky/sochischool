import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable


class KipfMLPGNN(nn.Module):
    """MLP decoder module."""

    def __init__(self, vocab_size, padding_idx, embd_dim,
                 num_layers, msg_hid, edge_types, do_prob=0.):
        super(KipfMLPGNN, self).__init__()
        # Make sure there are no more than 2 edge types.
        assert edge_types <= 2
        self.edge_types = edge_types
        # Setup embeddings (unlexicalized for now).
        self.embd = nn.Embedding(vocab_size, embd_dim, padding_idx)
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * embd_dim, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, embd_dim) for _ in range(edge_types)])
        self.msg_out_shape = embd_dim

        self.num_layers = num_layers

        self.dropout_prob = do_prob

    def single_step_forward(self, E, senders_idxs, receivers_idxs, arcs):
        """
        Args:
            E: E^t; shape (batch_size, n, embd_dim)
            senders_idxs: One-hot indices of all sender indices;
                shape (n ** 2, n)
            receivers_idxs: One-hot indices of all receiving indices;
                shape (n ** 2, n)
            arcs: Shape (batch_size, n ** 2, edge_types)
            lengths: Shape (batch_size, n)
        Returns:
            E^{t+1} of shape (batch_size, embd_dim)
        """
        # Node2edge
        senders = torch.matmul(senders_idxs, E)      # (bs, n ** 2 , embd_dim).
        receivers = torch.matmul(receivers_idxs, E)  # (bs, n ** 2, embd_dim).
        pre_msg = torch.cat([senders, receivers], dim=-1)

        # (bs, n ** 2, embd_dim)
        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        self.msg_out_shape))
        if E.is_cuda:
            all_msgs = all_msgs.cuda()

        # Run separate MLP for every edge type
        for i in range(self.edge_types):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * arcs[:, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(receivers_idxs).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        return agg_msgs

    def forward(self, x, arcs, lengths):
        """
        Args:
            x: Shape (batch_size, n)
            arcs: Shape (batch_size, n, n)
            lengths: Shape (batch_size, n)
        Returns:
            E^T where T = self.num_layers, shape (batch_size, n, embd_dim)
        """
        del lengths
        embd = self.embd(x)  # (bs, n, embd_dim)

        # Define matrices representing receiving and sending nodes.
        # Need to do this every forward pass since the size of x might
        # change per batch.
        send_indices, rec_indices = torch.where(torch.ones_like(arcs[0]))
        senders = F.one_hot(send_indices).float()
        receivers = F.one_hot(rec_indices).float()

        # Flatten arcs matrix to be shape (bs, n ** 2)
        arcs = arcs.flatten(1, 2).float()
        # If edge_types == 2, we add the null edge by taking 1.0 - arcs.
        if self.edge_types == 2:
            arcs = torch.stack([1.0 - arcs, arcs], axis=-1)
        else:
            arcs = arcs.unsqueeze(-1)
        # Arcs is now shape (bs, n ** 2, edge_types).

        pred = embd
        for _ in range(self.num_layers):
            # Use receivers as senders, and vice versa, such that
            # messages are passed from children to parents.
            pred += self.single_step_forward(pred, senders_idxs=receivers,
                                            receivers_idxs=senders, arcs=arcs)

        return pred[:, 0, :]
