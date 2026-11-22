"""
triad.py
========
Triad classification (Task T3).

A triad T = {a, b, c} is encoded, for a given ORDER (x0, x1, x2) of its three
nodes, as a 6-bit integer. Bit k is 1 iff the k-th directed edge below exists:

    Bit:   0     1     2     3     4     5
    Edge: 0->1  0->2  1->0  1->2  2->0  2->1

The CANONICAL CODE of T is the minimum of this integer over all 3! = 6
orderings of {a, b, c}. Two triads are isomorphic iff they share a canonical
code. There are 13 canonical codes that correspond to CONNECTED triads:

    Class:  1  2  3  4   5   6   7   8   9   10  11  12  13
    Code:   3  6  7  10  11  15  21  23  25  27  30  31  63

(Codes 0 and the other two 16-13=3 disconnected orbits never appear here
because every triad we ever classify is already known to be connected --
see count.py for why.)
"""

from itertools import permutations

# canonical connected codes, sorted ascending -> class number 1..13
_CONNECTED_CODES = [3, 6, 7, 10, 11, 15, 21, 23, 25, 27, 30, 31, 63]
CODE_TO_CLASS = {code: i + 1 for i, code in enumerate(_CONNECTED_CODES)}
CLASS_TO_CODE = {v: k for k, v in CODE_TO_CLASS.items()}
NUM_CLASSES = 13


def _encode(order, edge_set):
    """6-bit code of the triad for one fixed ordering (x0, x1, x2)."""
    x0, x1, x2 = order
    bits = 0
    if (x0, x1) in edge_set:
        bits |= 1 << 0
    if (x0, x2) in edge_set:
        bits |= 1 << 1
    if (x1, x0) in edge_set:
        bits |= 1 << 2
    if (x1, x2) in edge_set:
        bits |= 1 << 3
    if (x2, x0) in edge_set:
        bits |= 1 << 4
    if (x2, x1) in edge_set:
        bits |= 1 << 5
    return bits


def canonical_code(a, b, c, edge_set):
    """Minimum 6-bit code over all 6 orderings of (a, b, c)."""
    return min(_encode(order, edge_set) for order in permutations((a, b, c)))


def triad_class(a, b, c, edge_set):
    """
    Return the class number (1-13) of the triad {a, b, c} under the
    directed-edge set `edge_set` (a Python set of (u, v) tuples).

    Only ever called on triads already known to be CONNECTED (see
    count.py's enumeration), so the canonical code is always one of the
    13 entries in CODE_TO_CLASS.
    """
    code = canonical_code(a, b, c, edge_set)
    return CODE_TO_CLASS[code]


# -----------------------------------------------------------------------
# Hand-worked example (T3 deliverable: "include one example worked by hand")
# -----------------------------------------------------------------------
def hand_worked_example():
    """
    Worked example using three REAL node IDs from the cleaned email-Eu-core
    graph, all inside Department 1 (our Department A): nodes 0, 1, 18.

    Real directed edges among them (verified directly against the data):
        0  -> 1
        0  -> 18
        18 -> 0

    So edge_set (restricted to these 3 nodes) = {(0,1), (0,18), (18,0)}
    (node 1 and node 18 have no edge between them at all.)

    Step 1 - try every ordering (x0, x1, x2) of (0, 1, 18) and compute the
    6-bit code using the bit table:
        bit0: x0->x1   bit1: x0->x2   bit2: x1->x0
        bit3: x1->x2   bit4: x2->x0   bit5: x2->x1

    Ordering (0, 1, 18):
        0->1(1), 0->18(1->2), 1->0(0), 1->18(0), 18->0(1->16), 18->1(0)
        code = 1 + 2 + 16 = 19

    Ordering (0, 18, 1):
        0->18(1), 0->1(1->2), 18->0(1->4), 18->1(0), 1->0(0), 1->18(0)
        code = 1 + 2 + 4 = 7

    Ordering (1, 0, 18):
        1->0(0), 1->18(0), 0->1(1->4), 0->18(1->8), 18->1(0), 18->0(1->32)
        code = 4 + 8 + 32 = 44

    Ordering (1, 18, 0):
        1->18(0), 1->0(0), 18->1(0), 18->0(1->8), 0->1(1->16), 0->18(1->32)
        code = 8 + 16 + 32 = 56

    Ordering (18, 0, 1):
        18->0(1), 18->1(0), 0->18(1->4), 0->1(1->8), 1->18(0), 1->0(0)
        code = 1 + 4 + 8 = 13

    Ordering (18, 1, 0):
        18->1(0), 18->0(1->2), 1->18(0), 1->0(0), 0->18(1->16), 0->1(1->32)
        code = 2 + 16 + 32 = 50

    Step 2 - canonical code = min(19, 7, 44, 56, 13, 50) = 7

    Step 3 - 7 is in the connected-code table [3,6,7,10,11,15,21,23,25,27,
    30,31,63] at position 3 -> Class 3.
    (Class 3 is the "out-star with one reciprocated arm" shape: one node
    sends to both others, and one of those replies back.)

    (This matches what the code below computes automatically.)
    """
    edge_set = {(0, 1), (0, 18), (18, 0)}
    code = canonical_code(0, 1, 18, edge_set)
    cls = triad_class(0, 1, 18, edge_set)
    return code, cls


if __name__ == "__main__":
    code, cls = hand_worked_example()
    print(f"Hand-worked example: canonical code = {code}  ->  Class {cls}")
    assert code == 7 and cls == 3, "hand example does not match code logic!"
    print("Matches the by-hand derivation. triad.py self-check OK.")
