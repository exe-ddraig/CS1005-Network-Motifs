"""
count.py
========
Task T4: triad census of a directed graph, WITHOUT enumerating all
C(|V|, 3) subsets.

Why the neighbour-pair trick is correct and complete
------------------------------------------------------
A 3-node induced subgraph is "connected" (its underlying undirected graph
is connected) iff it has at least 2 of the 3 possible undirected edges.
With exactly 2 edges (a path x-y-z) the middle node y is adjacent (in the
underlying sense) to BOTH other nodes. With all 3 edges (a triangle) every
node is adjacent to both others. So every connected triad has at least one
"centre" node that is underlying-adjacent to the other two.

That means: if we go through every node x, look at its underlying
neighbours N(x) (nodes reachable by an edge in EITHER direction), and take
every unordered pair {a, b} from N(x), the triad {x, a, b} is guaranteed to
be connected (x-a and x-b edges already exist). And every connected triad
gets generated this way at least once (by one of its centre nodes).

A triad can have more than one centre (e.g. a triangle has three), so the
same triad is generated multiple times from different x -- we de-duplicate
with a set of frozensets before classifying.

Cost: O(sum_x deg(x)^2), far below O(|V|^3) for the sparse department
subgraphs we use here.
"""

from itertools import combinations
from collections import defaultdict

from triad import triad_class, NUM_CLASSES


def build_underlying_adjacency(edge_set):
    """node -> set of neighbours, ignoring edge direction."""
    adj = defaultdict(set)
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def enumerate_connected_triads(edge_set):
    """Yield each connected triad {a, b, c} exactly once, as a sorted tuple."""
    adj = build_underlying_adjacency(edge_set)
    seen = set()
    for x in adj:
        neigh = sorted(adj[x])
        for a, b in combinations(neigh, 2):
            tri = frozenset((x, a, b))
            if tri in seen:
                continue
            seen.add(tri)
            yield tuple(sorted(tri))


def triad_census(edge_set):
    """
    Compute N(G) = (N_1, ..., N_13): counts of each of the 13 connected
    triad classes. Returns a dict {class_number: count}, classes 1..13 all
    present (possibly 0).
    """
    counts = {k: 0 for k in range(1, NUM_CLASSES + 1)}
    for a, b, c in enumerate_connected_triads(edge_set):
        cls = triad_class(a, b, c, edge_set)
        counts[cls] += 1
    return counts


if __name__ == "__main__":
    # tiny self-test: a 4-node directed graph, hand-countable
    #   0->1, 1->0, 1->2, 2->3, 3->2
    test_edges = {(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)}
    # connected triads: {0,1,2} (0<->1, 1->2)  and  {1,2,3} (1->2, 2<->3)
    # {0,1,3} and {0,2,3} are disconnected (no path through them)
    c = triad_census(test_edges)
    print("Test census:", {k: v for k, v in c.items() if v})
    total = sum(c.values())
    print("Total connected triads found:", total, "(expected 2)")
    assert total == 2
    print("count.py self-check OK.")
