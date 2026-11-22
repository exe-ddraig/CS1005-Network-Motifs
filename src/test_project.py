"""
test_project.py
================
Lightweight unit tests for triad.py, count.py, and randomize.py.

Not an assignment deliverable on its own, but useful to run before your
demo to show every module behaves as specified. Run with:

    python3 -m unittest test_project.py -v
"""

import unittest
from triad import canonical_code, triad_class, CODE_TO_CLASS
from count import triad_census, enumerate_connected_triads
from randomize import randomize, assert_degrees_preserved


class TestTriadClassification(unittest.TestCase):

    def test_all_13_canonical_codes_present(self):
        expected = [3, 6, 7, 10, 11, 15, 21, 23, 25, 27, 30, 31, 63]
        self.assertEqual(sorted(CODE_TO_CLASS.keys()), expected)
        self.assertEqual(set(CODE_TO_CLASS.values()), set(range(1, 14)))

    def test_directed_3cycle_is_class_9(self):
        # a->b->c->a, code 25 per Section 2's worked logic
        edges = {(0, 1), (1, 2), (2, 0)}
        self.assertEqual(canonical_code(0, 1, 2, edges), 25)
        self.assertEqual(triad_class(0, 1, 2, edges), 9)

    def test_feedforward_loop_is_class_5(self):
        # a->b, a->c, b->c : code 11
        edges = {(0, 1), (0, 2), (1, 2)}
        self.assertEqual(canonical_code(0, 1, 2, edges), 11)
        self.assertEqual(triad_class(0, 1, 2, edges), 5)

    def test_full_reciprocal_triangle_is_class_13(self):
        edges = {(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)}
        self.assertEqual(canonical_code(0, 1, 2, edges), 63)
        self.assertEqual(triad_class(0, 1, 2, edges), 13)

    def test_canonical_code_is_relabel_invariant(self):
        # relabeling the same real-world triad must give the same code
        edges = {(10, 20), (20, 10), (20, 30)}
        c1 = canonical_code(10, 20, 30, edges)
        c2 = canonical_code(20, 30, 10, edges)  # same nodes, different arg order
        c3 = canonical_code(30, 10, 20, edges)
        self.assertEqual(c1, c2)
        self.assertEqual(c2, c3)


class TestTriadCensus(unittest.TestCase):

    def test_single_triangle(self):
        edges = {(0, 1), (1, 2), (2, 0)}
        counts = triad_census(edges)
        self.assertEqual(sum(counts.values()), 1)
        self.assertEqual(counts[9], 1)  # directed 3-cycle

    def test_disconnected_triad_not_counted(self):
        # (0,1) and (2,3) are two separate components -> no triad spans them
        edges = {(0, 1), (2, 3)}
        counts = triad_census(edges)
        self.assertEqual(sum(counts.values()), 0)

    def test_path_plus_isolated_edge(self):
        # 0->1->2 is one connected triad {0,1,2}; node 3 with edge 3->0
        # also forms a connected triad with {0, 1} via node 0 as centre,
        # and with {0, 2} is NOT connected (0-2 only path is through 1,
        # but 3 isn't adjacent to 1, 2 directly) -- verify exact count
        # by comparing against brute-force C(n,3) enumeration.
        edges = {(0, 1), (1, 2), (3, 0)}
        nodes = {0, 1, 2, 3}
        counts = triad_census(edges)
        total_fast = sum(counts.values())

        # brute-force cross-check
        from itertools import combinations
        total_brute = 0
        for a, b, c in combinations(sorted(nodes), 3):
            adj = {a: set(), b: set(), c: set()}
            for u, v in edges:
                if u in adj and v in adj:
                    adj[u].add(v)
                    adj[v].add(u)
            # connected iff the 3-node undirected induced subgraph is connected
            seen = {a}
            frontier = [a]
            local = {a: adj[a] & {a, b, c}, b: adj[b] & {a, b, c}, c: adj[c] & {a, b, c}}
            while frontier:
                x = frontier.pop()
                for y in local[x]:
                    if y not in seen:
                        seen.add(y)
                        frontier.append(y)
            if seen == {a, b, c}:
                total_brute += 1
        self.assertEqual(total_fast, total_brute)


class TestRandomize(unittest.TestCase):

    def test_degree_sequences_preserved(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        out = randomize(edges, attempts=500, seed=99)
        assert_degrees_preserved(edges, out)  # raises on failure

    def test_no_self_loops_or_duplicates_created(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        out = randomize(edges, attempts=500, seed=7)
        self.assertTrue(all(u != v for u, v in out))
        self.assertEqual(len(set(out)), len(out))

    def test_edge_count_unchanged(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        out = randomize(edges, attempts=500, seed=1)
        self.assertEqual(len(out), len(edges))


if __name__ == "__main__":
    unittest.main(verbosity=2)
