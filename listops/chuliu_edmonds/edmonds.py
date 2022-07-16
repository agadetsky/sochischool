from functools import partial
import networkx as nx
import numpy as np

import torch
# Compile the edmonds c++ code by running
#   python setup_edmonds.py install
# once before importing edmonds_cpp.
#import edmonds_cpp

from listops.chuliu_edmonds.arborescence_chr import minium_rooted_arborescence


def edmonds_python(weights, lengths, method="chr"):
    """
    Gets the maximum spanning arborescence given weights of edges.
    We assume the root is node (idx) 0.
    Args:
        weights: shape (batch_size, n, n), where
            weights[.][i][j] is the weight for edge i -> j.
        lengths: shape (batch_size,) where lengths[i] is the true dim of
            the i-th sample in adjs. lengths[i] <= n for all i.
    Returns:
        Adjacency matrix of size (batch_size, n, n);
            adjs[.][i][j] = 1 if edge i -> j exists.
    """
    # Convert roots and weights_and_edges to numpy arrays on the cpu.
    if torch.is_tensor(weights):
        weights = weights.detach().to("cpu").numpy()

    n = weights.shape[-1]
    # Loop over batch dimension to get the maximum spanning arborescence for
    # each graph.
    batch_size = weights.shape[0]
    adjs = np.zeros((batch_size, n, n))
    for sample_idx in range(batch_size):
        # We first extract the true adjacency matrix given length.
        w = weights[sample_idx][:lengths[sample_idx], :lengths[sample_idx]]
        np.fill_diagonal(w, 0.0)
        # We multiply by -1.0 since both methods below obtains the
        # minimum spanning arborescence. We want the maximum.
        G = nx.from_numpy_matrix(-1.0 * w, create_using=nx.DiGraph())

        if method == "chr":
            msa = minium_rooted_arborescence(G, 0)
        elif method == "nx":
            Gcopy = G.copy()
            # Remove all incoming edges for the root such that
            # the given "root" is forced to be selected as the root.
            Gcopy.remove_edges_from(G.in_edges(nbunch=[0]))
            msa = nx.minimum_spanning_arborescence(Gcopy)
        else:
            raise ValueError("Method must be one of {'chr', 'nx'}, "
                            "but was given %s"%method)

        # Convert msa nx graph to heads list.
        for i, j in msa.edges:
            i, j = int(i), int(j)
            adjs[sample_idx][i][j] = 1.0

    return adjs


def edmonds_cpp_pytorch(weights, lengths):
    """
    Gets the maximum spanning arborescence given weights of edges.
    We assume the root is node (idx) 0.
    Args:
        weights: shape (batch_size, n, n), where
            weights[.][i][j] is the weight for edge i -> j.
        lengths: shape (batch_size,) where lengths[i] is the true dim of
            the i-th sample in adjs. lengths[i] <= n for all i.
    Returns:
        Adjacency matrix of size (batch_size, n, n);
            adjs[.][i][j] = 1 if edge i -> j exists.
    """
    # Transpose weights, since the function expects
    # weights[i][j] to be the weights for edge j -> i.
    heads = edmonds_cpp.get_maximum_spanning_arborescence(
        weights.transpose(-2, -1), lengths)
    return heads


if __name__ == "__main__":
    n = 10
    bs = 100
    lengths = np.random.choice(np.arange(3, n), bs)
    np.random.seed(42)
    weights = np.zeros((bs, n, n))
    for i in range(bs):
        w = np.random.rand(lengths[i], lengths[i])
        weights[i:, :lengths[i], :lengths[i]] = w

    res_chr = edmonds_python(weights, lengths, "chr")
    res_nx = edmonds_python(weights, lengths, "nx")
    res_cpp = edmonds_cpp_pytorch(torch.tensor(weights),
                                  torch.tensor(lengths)).numpy()

    np.testing.assert_almost_equal(res_chr, res_nx)
    np.testing.assert_almost_equal(res_nx, res_cpp)
