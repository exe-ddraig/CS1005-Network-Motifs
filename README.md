# CS1005 Discrete Structures — Fall 2026 Project
Finding Network Motifs in an Email Network

## Group

| # | Roll number | Section |
|---|---|---|
| 1 | `25P-0718` | A |
| 2 | `25P-0635` | A |
| 3 | `25I-0566` | B |
| 4 | `25I-1015` | B |

## R and derived parameters

```
R = 718 + 635 + 566 + 1015 = 2934

Pair index       = R mod 190 = 84          -> (i, j) = (4, 19)
Department A     = top20[4]                -> Department 1
Department B     = top20[19]               -> Department 20
Random networks  = 20 + (R mod 31) = 20+20  -> 40
Swap multiplier  = 5 + (R mod 6)  = 5+0     -> 5
Swap attempts    = 5 x (edges in that department's subgraph)
                     Dept 1  (65 nodes, 502 edges): 5 x 502 = 2510
                     Dept 20 (14 nodes,  70 edges): 5 x  70 =  350
Random seed      = R = 2934
```

## Dataset

SNAP `email-Eu-core`: 1,005 nodes, 25,571 directed edges, 42 departments.
Files are included under `data/` so the project runs offline:

- `data/email-Eu-core.txt` — edge list (`u v` = u emailed v)
- `data/email-Eu-core-department-labels.txt` — `node department`

Preprocessing removes 642 self-loops (there were no duplicate directed
edges in the raw file), leaving 24,929 clean edges — done in
`clean_edges()` in `main.py`.

## How to run

```bash
cd src
pip install networkx matplotlib numpy     # networkx/matplotlib used only for
                                          # I/O, drawing, and the optional
                                          # triadic_census cross-check
python3 main.py
```

This runs T1–T7 end to end, prints every required table to the console
(parameters, triad censuses, Z-scores), and writes:
- `output/motif_chart.png` — the T7 grouped bar chart

You can also run each module's self-check independently:
```bash
python3 src/triad.py        # T3 hand-worked example + assertion
python3 src/count.py        # T4 triad census tiny self-test
python3 src/randomize.py    # T5 degree-preservation self-test
```

For a fuller check before your demo, run the unit test suite (11 tests
covering canonical codes, census correctness against a brute-force
cross-check, and degree preservation):
```bash
cd src
python3 -m unittest test_project.py -v
```

## Code structure

```
src/
  triad.py         # T3: triad_class(), canonical_code(), hand-worked example
  count.py         # T4: triad_census() via the neighbour-pair enumeration
                   #     (no C(|V|,3) brute force — see module docstring)
  randomize.py     # T5: randomize() double-edge-swap null-model sampler
  main.py          # T1, T2, T6, T7: load/clean data, build subgraphs,
                   #     compute Z-scores, draw the chart
  test_project.py  # unit tests (not a deliverable, but useful to run
                   #     before the demo to show every module behaves)
data/              # the two SNAP files
output/            # generated chart (created by running main.py)
```

Only `networkx` is used, and only for reading the edge list into a
`DiGraph` for the `nx.triadic_census` **cross-check** (explicitly allowed
by the rules, and labelled as such in the output) and for nothing in the
actual triad-classification / counting / randomization / Z-score logic,
which is all hand-written in `triad.py`, `count.py`, and `randomize.py`.
`matplotlib`/`numpy` are used only for T7's chart.

## T3 — hand-worked example

See the full by-hand derivation in the docstring of
`hand_worked_example()` in `src/triad.py`. Summary: nodes **0, 1, 18**
(all in Department 1) have real edges `0→1`, `0→18`, `18→0`. Trying all 6
orderings of the three nodes and taking the minimum 6-bit code gives
**canonical code 7**, which is **Class 3**.

## T4/T6 — results

**Triad census** (13 classes, both departments):

| Class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Dept 1  | 443 | 174 | 1074 | 91 | 59 | 189 | 266 | 740 | 1 | 50 | 61 | 370 | 385 |
| Dept 20 | 2 | 2 | 23 | 0 | 0 | 1 | 16 | 24 | 0 | 1 | 0 | 14 | 48 |

Cross-checked: our total connected-triad count (3,903 for Dept 1, 131 for
Dept 20) matches summing the 13 connected classes out of `nx.triadic_census`
exactly (note `nx.triadic_census` uses a different 16-class naming scheme —
we only used its *total connected* count as a sanity check, not its labels
or the function itself for classification).

**Z-scores** (40 random graphs each, double-edge-swap null model):

| Class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Z (Dept 1)  | -8.03 | -9.31 | -2.04 | -7.89 | -5.50 | -4.37 | -7.10 | **12.15** | -4.14 | -9.33 | -4.20 | -3.34 | **14.27** |
| Z (Dept 20) | -0.20 | -1.45 | -1.84 | -1.23 | -0.66 | -0.88 | -1.12 | 1.92 | -0.50 | -1.44 | -1.41 | -1.07 | 1.67 |

**Motifs** (Z > 2): Dept 1 → **Class 8** and **Class 13**.
**Anti-motifs** (Z < -2): Dept 1 → classes 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12
(every other class). Dept 20 → none pass either threshold at all.

## T8 — how do the two departments differ?

Department 1 (65 members, 502 internal edges — the densest department
outside the top four) has a triad profile dominated by **two classes far
above the null model**: Class 13 (the fully-reciprocated triangle — every
pair of the three members emails both ways) is almost 3.5x more common than
chance (observed 385 vs. an expected ~164), and Class 8 (a triangle with one
one-way arm plus two reciprocated edges) is roughly 2.4x more common than
chance (740 vs. ~305). Every other triad shape is significantly
**under**-represented. In other words, once three people in Department 1
are connected at all, the department pushes hard towards *fully mutual*
three-way communication rather than any lopsided or one-directional
pattern — consistent with a tight-knit team where email is reciprocal by
norm (replies, group threads, back-and-forth), not broadcast.

Department 20 (14 members, 70 internal edges — the smallest department in
the top 20) shows **no statistically significant motif or anti-motif at
all** (every |Z| < 2, even for the same two classes 8 and 13 that spike in
Dept 1). With so few members and edges, the pool of possible random graphs
sharing that degree sequence isn't very different from the real one, so no
triad shape stands out — the department is simply too small for its
local structure to diverge meaningfully from a random graph with the same
degrees. This itself is a meaningful comparison: Department 1's size and
density let real organizational behaviour (strong reciprocity) show up
clearly against the null model, while Department 20 looks statistically
indistinguishable from a random network of its size.

## Contributions

- **Member 1 (`25P-0718`, Section A):** 
  - Designed the overall project structure and repository workflow.
  - Implemented the data loading and cleaning pipeline (`load_raw_edges`, `load_labels`, `clean_edges`), stripping self-loops and duplicate edges.
  - Developed the Top-20 department ranking algorithm and group parameter derivation formulas ($R = 2934$, pair index $84 \to (4, 19)$, random networks, swap multiplier).
  - Built the department induced subgraph extraction module (Task T1 & T2).

- **Member 2 (`25P-0635`, Section A):** 
  - Implemented the 6-bit binary triad configuration encoding and canonical minimization over all $3! = 6$ permutations in $S_3$ (`src/triad.py`).
  - Mapped the 16 orbits to the 13 connected canonical classes (Task T3).
  - Authored the step-by-step mathematical hand-worked example for nodes `(0, 1, 18)` deriving canonical code 7 $\to$ Class 3.
  - Contributed to presentation slides for the problem definition and canonical orbit theory.

- **Member 3 (`25I-0566`, Section B):** 
  - Implemented the efficient neighbor-pair enumeration algorithm (`src/count.py`) avoiding $O(V^3)$ brute force, enabling millisecond census computation.
  - Built the 13-class triad census counter (`triad_census`) and validated against `nx.triadic_census` cross-checks (Task T4).
  - Wrote the comprehensive automated unit test suite (`src/test_project.py`) verifying classification, relabel invariance, and degree preservation.
  - Co-authored the comparative communication structure analysis for Task T8.

- **Member 4 (`25I-1015`, Section B):** 
  - Implemented the double-edge swap null-model sampling algorithm with strict rejection conditions (`src/randomize.py`).
  - Implemented degree-sequence tracking and rigorous assertions (`assert_degrees_preserved`) ensuring exact in- and out-degree preservation (Task T5).
  - Developed the statistical $Z$-score engine with population standard deviation and motif/anti-motif classification (Task T6).
  - Built the publication-grade grouped bar chart visualizer with dashed significance threshold lines at $Z = \pm 2$ (Task T7). 
