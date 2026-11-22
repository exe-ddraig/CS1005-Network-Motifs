"""
randomize.py
============
Task T5: null-model sampling via double-edge swaps.

One swap attempt: pick two DISTINCT edges (a, b) and (c, d) uniformly at
random from the current edge list, and try to replace them with (a, d) and
(c, b). Reject (leave the graph unchanged) if:
    - a == d   (would create a self-loop a->a)
    - c == b   (would create a self-loop c->c)
    - (a, d) already in the edge set   (would create a parallel edge)
    - (c, b) already in the edge set   (would create a parallel edge)
Rejected attempts still count towards the total number of attempts.

Double-edge swaps preserve every node's in-degree AND out-degree exactly:
edge (a,b) is removed and (a,d) added -> a's out-degree unchanged, b's
in-degree lost one; edge (c,d) removed and (c,b) added -> c's out-degree
unchanged, d's in-degree lost one, but b gained an in-edge from c and d
gained... let's just verify: before: a->b, c->d. after: a->d, c->b.
    out-degree: a still has exactly one of these two edges leaving it (a->b
    became a->d): unchanged count. c symmetric: unchanged.
    in-degree: b went from receiving-from-a to receiving-from-c: unchanged
    count. d symmetric: unchanged.
So every node's in/out-degree sequence is identical before and after any
number of swaps -- this is exactly why the resulting random graphs are a
valid null model for "same degree sequence, otherwise random".
"""

import random
from collections import Counter


def randomize(edges, attempts, seed):
    """
    edges: iterable of (u, v) directed edges (a simple digraph: no
           self-loops, no parallel edges going in).
    attempts: number of swap attempts to make.
    seed: RNG seed, for reproducibility.

    Returns a NEW list of (u, v) edges after `attempts` swap attempts.
    """
    rng = random.Random(seed)
    edge_list = list(edges)
    edge_set = set(edge_list)
    n = len(edge_list)
    if n < 2:
        return edge_list

    for _ in range(attempts):
        i = rng.randrange(n)
        j = rng.randrange(n)
        while j == i:
            j = rng.randrange(n)

        a, b = edge_list[i]
        c, d = edge_list[j]

        if a == d or c == b:
            continue
        if (a, d) in edge_set or (c, b) in edge_set:
            continue

        # perform the swap
        edge_set.discard((a, b))
        edge_set.discard((c, d))
        edge_set.add((a, d))
        edge_set.add((c, b))
        edge_list[i] = (a, d)
        edge_list[j] = (c, b)

    return edge_list


def degree_sequences(edges):
    """{node: in-degree} and {node: out-degree} as Counters."""
    indeg, outdeg = Counter(), Counter()
    for u, v in edges:
        outdeg[u] += 1
        indeg[v] += 1
    return indeg, outdeg


def assert_degrees_preserved(original_edges, randomized_edges):
    """Verify in/out-degree sequences are identical (Task T5 requirement)."""
    in0, out0 = degree_sequences(original_edges)
    in1, out1 = degree_sequences(randomized_edges)
    assert in0 == in1, "in-degree sequence changed after randomization!"
    assert out0 == out1, "out-degree sequence changed after randomization!"


if __name__ == "__main__":
    test_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
    randomized = randomize(test_edges, attempts=200, seed=42)
    assert_degrees_preserved(test_edges, randomized)
    assert len(randomized) == len(test_edges)
    assert len(set(randomized)) == len(randomized), "duplicate edge created!"
    assert all(u != v for u, v in randomized), "self-loop created!"
    print("Original :", sorted(test_edges))
    print("Randomized:", sorted(randomized))
    print("Degree sequences preserved, no self-loops/duplicates. randomize.py self-check OK.")
