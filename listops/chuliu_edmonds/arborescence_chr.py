import numpy as np
import networkx as nx
from networkx.algorithms.tree.branchings import maximum_spanning_arborescence, minimum_spanning_arborescence
import itertools
import random
import collections

def unpackbits(x, num_bits):
  xshape = list(x.shape)
  x = x.reshape([-1, 1])
  to_and = 2**np.arange(num_bits).reshape([1, num_bits])
  return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

def cube_corners(dim):
  return np.array(unpackbits(np.arange(2**dim, dtype=np.int), dim).T).T

"""
UTILITY FUNCTION
"""

def has_arborescence(dg, r):
    descendants = nx.descendants(dg, r)
    descendants.add(r)
    return descendants == set(dg.nodes())

def subgraph_weight(dg, subdg):
    return sum(dg[u][v]["weight"] for (u, v, d) in subdg.edges(data=True))


"""
BASE ALGORITHM
"""

def get_cycle(candidate):
    """Extract a cycle from a candidate arborescence.

    Returns the cycle and the remaining subgraph separately."""

    cycles = list(nx.simple_cycles(candidate))
    near_arborescence = nx.MultiDiGraph(candidate)
    if len(cycles) > 0:
        cycle_nodes = cycles[0]
        cycle = nx.MultiDiGraph()
        cycle.add_edges_from(candidate.in_edges(nbunch=cycle_nodes, data=True, keys=True))
        near_arborescence.remove_edges_from(candidate.in_edges(nbunch=cycle_nodes, data=True, keys=True))
        return cycle, near_arborescence
    else:
        arborescence = near_arborescence
        return None, arborescence

def contract_cycle(mdg, cycle, v_cycle):
    """Contract a cycle <= mdg into a single node, v_cycle.

    Returns the contracted graph and a reference dictionary to allow for
    expanding it back."""


    mdg_prime = nx.MultiDiGraph()
    reference = dict()

    for (u, v, k, d) in mdg.edges(data=True, keys=True):

        if u not in cycle and v in cycle:
            src = u
            dst = v_cycle
            key = mdg_prime.number_of_edges(src, dst)
            mdg_prime.add_edge(src, dst, key=key, **d)
            reference[(src, dst, key)] = (u, v, k, d)

        elif u in cycle and v not in cycle:
            src = v_cycle
            dst = v
            key = mdg_prime.number_of_edges(src, dst)
            mdg_prime.add_edge(src, dst, key=key, **d)
            reference[(src, dst, key)] = (u, v, k, d)

        elif u not in cycle and v not in cycle:
            src = u
            dst = v
            mdg_prime.add_edge(src, dst, key=k, **d)

    return mdg_prime, reference

def expand_graph(mdg, reference, v_expand):
    """Take node v_expand in mdg and expand it using the reference dictionary.
    """

    mdg_prime = nx.MultiDiGraph()

    for (u, v, k, d) in mdg.edges(data=True, keys=True):
        if (v == v_expand) or (u == v_expand):
            (trueu, truev, truek, trued) = reference[(u, v, k)]
            mdg_prime.add_edge(trueu, truev, key=truek, **trued)
        else:
            mdg_prime.add_edge(u, v, key=k, **d)

    return mdg_prime

def _mra_recursion(mdg, root, arborescence_proposer):
    # Normalize all costs to be >= 0
    near_arborescence = arborescence_proposer(mdg, root)
    # Check for cycles and return if none
    cycle, near_arborescence = get_cycle(near_arborescence)
    if cycle is None:
        return near_arborescence

    # Contract the cycle
    new_node = max(mdg.nodes) + 1
    mdg_prime, reference = contract_cycle(mdg, cycle, new_node)

    # Recurse
    arborescence_prime = _mra_recursion(mdg_prime, root, arborescence_proposer)

    # Expand the arborescence to be a subgraph of the current graph
    arborescence = expand_graph(arborescence_prime, reference, new_node)

    # Remove the edge from the cycle that conflicts with the entering edge
    u, _, k, _ = list(arborescence_prime.in_edges(nbunch=[new_node], data=True, keys=True))[0]
    _, cycle_v, _, _ = reference[(u, new_node, k)]
    cycle_u, cycle_v, _, _ = list(cycle.in_edges(nbunch=[cycle_v], data=True, keys=True))[0]
    cycle.remove_edge(cycle_u, cycle_v)

    # and add cycle edges
    arborescence.add_edges_from(cycle.edges(data=True, keys=True))

    return arborescence

"""
MIN VARIANTS
"""

def modify_costs(mdg, root):
    for v in mdg.nodes():
        if v != root:
            entering_edges = list(mdg.in_edges(nbunch=[v], data=True, keys=True))
            minweight = min(entering_edges, key=lambda e: e[3]["weight"], default=(None, None, None, {"weight":0}))[3]["weight"]
            for u, _, k, d in entering_edges:
                d["weight"] = d["weight"] - minweight

def get_zero_entering_edges(mdg, root):
    submdg = nx.MultiDiGraph()

    for v in mdg.nodes():
        if v != root:
            entering_edges = list(mdg.in_edges(nbunch=[v], data=True, keys=True))
            zero_weight_edges = list(filter(lambda e: e[3]["weight"] == 0, entering_edges))
            if len(zero_weight_edges) > 0:
                u, v, k, d = zero_weight_edges[0]
                submdg.add_edge(u, v, key=k, **d)

    return submdg

def min_proposer(mdg, root):
    modify_costs(mdg, root)
    return get_zero_entering_edges(mdg, root)

def minium_rooted_arborescence(digraph, root):
    dg = nx.MultiDiGraph(digraph)
    dg.remove_edges_from(nx.selfloop_edges(dg))
    for (u, v, k, d) in dg.edges(data=True, keys=True):
        assert "weight" in d

    if has_arborescence(dg, root):
        return nx.DiGraph(_mra_recursion(dg, root, min_proposer))

"""
SAMPLE VARIANTS
"""

def sample_entering_edges(mdg, root):
    submdg = nx.MultiDiGraph()

    for v in mdg.nodes():
        if v != root:
            entering_edges = list(mdg.in_edges(nbunch=[v], data=True, keys=True))
            marked_entering_edges = list(filter(lambda e: e[3]["marked"], entering_edges))

            if len(marked_entering_edges) > 0:
                (u, v, k, d) = random.choice(marked_entering_edges)
                submdg.add_edge(u, v, key=k, **d)
            else:
                potentials = [d["potential"] for (u, v, k, d) in entering_edges]
                (u, v, k, d) = random.choices(entering_edges, weights=potentials)[0]
                d["marked"] = True
                submdg.add_edge(u, v, key=k, **d)

    return submdg

def sample_minium_rooted_arborescence(digraph, root):
    dg = nx.MultiDiGraph(digraph)
    dg.remove_edges_from(nx.selfloop_edges(dg))
    for (u, v, k, d) in dg.edges(data=True, keys=True):
        d["marked"] = False
        assert "potential" in d

    if has_arborescence(dg, root):
        return nx.DiGraph(_mra_recursion(dg, root, sample_entering_edges))


"""
PRINTING AND MAIN
"""

def print_digraph(digraph, name, predecessor_view=True):
    print(name)
    nodes = list(digraph.nodes())
    nodes.sort()
    for u in nodes:
        if predecessor_view:
            print("{}  ".format(u))
            for v in digraph.predecessors(u):
                print("   ({} <- {}) : {:.2f}".format(u, v, digraph[v][u]["weight"]))
        else:
            print("{}  ".format(u))
            for v in digraph.successors(u):
                print("   ({} -> {}) : {:.2f}".format(u, v, digraph[u][v]["weight"]))
    print()

def main_sample():
    n = 4
    root = 0
    dg = nx.DiGraph(nx.complete_graph(n))
    for (u, v, d) in dg.edges(data=True):
        d["potential"] = 5*np.random.rand() + 1
    samples = 10000

    print("GO!\n\n")
    sample_counts = collections.Counter()
    exponential_counts = collections.Counter()
    for s in range(samples):
        sample_tree = sample_minium_rooted_arborescence(dg, root)
        sample_tree = list(sample_tree.edges())
        sample_tree.sort()
        sample_counts[tuple(sample_tree)] += 1

        for (u, v, d) in dg.edges(data=True):
            d["weight"] = np.random.exponential(scale = 1./d["potential"])
        exponential_tree = minium_rooted_arborescence(dg, root)
        exponential_tree = list(exponential_tree.edges())
        exponential_tree.sort()
        exponential_counts[tuple(exponential_tree)] += 1

    arbor = set(exponential_counts.keys()).union(set(sample_counts.keys()))
    for arborescence in arbor:
        print("arborescence:{}".format(arborescence))
        print("\t\tsample process:{}".format(sample_counts[arborescence]/samples))
        print("\t\texponential weights:{}".format(exponential_counts[arborescence]/samples))


def main_print():
    n = 50
    root = 0
    DG = nx.DiGraph(nx.complete_graph(n))
    for (u, v, d) in DG.edges(data=True):
        d["weight"] = np.random.randn()

    print_digraph(DG, "Predecessors", predecessor_view=True)
    print_digraph(DG, "Successors", predecessor_view=False)
    print("GO!\n\n")
    MRA = minium_rooted_arborescence(DG, root)
    print("Weight:{}".format(subgraph_weight(DG, MRA)))
    print_digraph(MRA, "MRA")

    try:
        DGcopy = DG.copy()
        DGcopy.remove_edges_from(DG.in_edges(nbunch=[root]))
        MSA = nx.minimum_spanning_arborescence(DGcopy)
        print("Weight:{}".format(subgraph_weight(DG, MSA)))
        print_digraph(MSA, "MSA")
    except:
        print("NO SPANNING ARBORESCENCE")

def main_test():
    n = 4
    root = 0
    corners = cube_corners(n ** 2 - n)
    i = 1
    for corner in corners:
        adj = np.zeros((n, n))
        adj[np.triu_indices(n, 1)] = corner[: n * (n-1) // 2]
        adj[np.tril_indices(n, -1)] = corner[n * (n-1) // 2:]
        print(i)
        print(adj)
        dg = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
        print(list(dg.nodes()))
        for (u, v, d) in dg.edges(data=True):
            d["weight"] = np.random.randn()

        try:
            mra = minium_rooted_arborescence(dg, root)
        except:
            mra = minium_rooted_arborescence(dg, root, verbose=True)

        try:
            dgcopy = dg.copy()
            dgcopy.remove_edges_from(dg.in_edges(nbunch=[root]))
            msa = nx.minimum_spanning_arborescence(dgcopy)
        except:
            print("NO SPANNING ARBORESCENCE")
            continue

        print(subgraph_weight(dg, msa), subgraph_weight(dg, mra))
        assert np.isclose(subgraph_weight(dg, msa), subgraph_weight(dg, mra))

        i += 1

if __name__ == '__main__':
    main_sample()
