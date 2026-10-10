# Order-Agnostic Generation of Finite Lattices via Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](pyproject.toml)
[![DOI](https://img.shields.io/badge/DOI-10.14293%2FPR2199.004125.v1-informational.svg)](https://doi.org/10.14293/pr2199.004125.v1)

A reinforcement-learning framework for generating finite partially ordered sets (posets) that satisfy a prescribed order-theoretic closure property: **lattice**, **join-semilattice**, **meet-semilattice**, or **Boolean algebra**. A recurrent policy builds a candidate poset by deciding the relation between *pairs* of elements, visited in a freshly randomized order at every rollout. An exact, policy-independent verification oracle certifies each candidate, and the policy is trained with Proximal Policy Optimization (PPO).

This work targets finite poset and lattice generation with a learned, verifier-in-the-loop sampler.

---

## Contents

1. [Motivation](#1-motivation)
2. [Key results](#2-key-results)
3. [Method](#3-method)
4. [Repository structure](#4-repository-structure)
5. [Installation](#5-installation)
6. [Usage](#6-usage)
7. [Output data format](#7-output-data-format)
8. [Reproducing the reported experiments](#8-reproducing-the-reported-experiments)
9. [Scope and limitations](#9-scope-and-limitations)
10. [Testing](#10-testing)
11. [Citation](#11-citation)
12. [License](#12-license)

---

## 1. Motivation

Let $P = (X, \le)$ be a finite poset.

- $P$ is a **join-semilattice** if every pair $x, y \in X$ has a unique least upper bound $x \vee y$.
- $P$ is a **meet-semilattice** if every pair $x, y \in X$ has a unique greatest lower bound $x \wedge y$.
- $P$ is a **lattice** if it is both.
- A lattice is **distributive** if $x \wedge (y \vee z) = (x \wedge y) \vee (x \wedge z)$ for all $x, y, z$, and **modular** if $x \le b \Rightarrow x \vee (a \wedge b) = (x \vee a) \wedge b$.
- A lattice is **Boolean** if it is distributive and complemented.

The number of non-isomorphic lattices on $n$ elements (OEIS A006966) grows super-exponentially (5,994 at $n = 10$), which confines exact, exhaustive enumeration to modest $n$. A uniformly random relation matrix satisfies the lattice axioms with vanishing probability as $n$ grows, so neither enumeration nor naive sampling reaches larger sizes. This repository studies whether a learned sampler can.

**A design finding.** Generating the relation matrix with a fixed, sequential visitation order (the default in node-by-node generators such as GraphRNN) structurally starves early-visited elements of admissible relations: under a strict order, node $j$ can only have predecessors among $\{0,\dots,j-1\}$, regardless of policy quality. Visiting all $\binom{n}{2}$ pairs in a freshly randomized order each rollout removes this bias and is the central design choice of this work.

## 2. Key results

All quantities below are measured on the *batch lattice rate*: the fraction of sampled posets in a rollout batch that the verification oracle certifies as lattices. Statistics are computed over logged checkpoints (iteration 1 and every 20th iteration). Results follow the revised manuscript that accompanies this repository; see [Scope and limitations](#9-scope-and-limitations) for how to read them.

**Order-agnostic vs. sequential construction ($n = 20$, CPU, no property-aware shaping).** Removing the sequential bias raises the mean lattice rate from 10.5% to 49.7% (a 4.7× improvement), and 48.4% under a corrected PPO importance-ratio computation (actions are replayed rather than resampled during the update).

**Scaling with problem size (property-aware shaping, entropy-augmented GAE, GPU).**

| $n$ | Target  | Batch $B$ | Iterations | Mean rate | Peak rate | Confirmed structures | Wall-clock |
|----:|---------|----------:|-----------:|----------:|----------:|---------------------:|-----------:|
| 20  | lattice | 16 | 300 | 38.3% | 75.0%  | 1,965 | 867 s |
| 20  | lattice | 32 | 150 | 49.6% | 68.8%  | 2,209 | 464 s |
| 20  | lattice | 64 | 75  | 48.1% | 53.1%  | 1,887 | 258 s |
| 30  | lattice | 16 | 300 | 61.3% | 87.5%  | 2,920 | 2,190 s |
| 40  | lattice | 16 | 300 | 80.5% | 100.0% | 3,895 | 3,712 s |
| 50  | lattice | 16 | 300 | 75.8% | 100.0% | 3,543 | 5,970 s |
| 20 / 30 / 40 | join-semilattice | 16 | 150 | 60.2% / 62.5% / 68.0% | 81.3% / 87.5% / 87.5% | 1,302 / 1,350 / 1,512 | — |
| 20 / 30 / 40 | meet-semilattice | 16 | 150 | 72.7% / 49.2% / 49.2% | 87.5% / 68.8% / 75.0% | 1,608 / 1,090 / 1,152 | — |

Wall-clock times are for a single run on a Google Colab GPU (GPU model not recorded). For $n \ge 30$ lattice runs, training sustained mean rates of 61–81% and peak rates of 87.5–100% with no collapse, at problem sizes roughly double the exact-enumeration ceiling. The $n = 20$, $B = 16$ run showed a late-training decline that did not reproduce at $B = 32$ or $B = 64$.

**Rare structure classes.** Under the property-aware reward, the policy repeatedly produced lattices that are modular but not distributive (necessarily containing an $M_3$ sublattice) and, once, a lattice that is upper semimodular but not modular (necessarily containing an $N_5$ sublattice). Bonus events in the $n = 20, 30, 40, 50$ lattice runs ($B=16$): 13, 15, 10, and 6 respectively. These count reward events, not distinct structures.

## 3. Method

### 3.1 Generative process

A rollout is a finite-horizon episode of $H = \binom{n}{2}$ decisions. A random permutation $\pi$ of all unordered pairs is drawn; for each pair $(u, v)$ the policy chooses one of three actions:

$$a \in \{\varnothing,\; u \to v,\; v \to u\}.$$

The policy is a GRU cell (hidden size 32) whose input is the sum of learned embeddings of the two endpoints concatenated with the one-hot previous action; actor and critic heads read the recurrent state. A single hidden state is carried across the $H$ steps, so a step costs $O(d)$ rather than a re-encoding of the partial graph.

### 3.2 Legality by construction

The sampler maintains a reflexive transitive closure (reachability) tensor $R$. The two edge actions are **masked out before the softmax** whenever $u$ and $v$ are already comparable. This one condition simultaneously guarantees acyclicity and prevents re-asserting an implied relation. After an accepted edge $u \to v$, $R$ is updated by an outer product,

$$R'[a,b] = R[a,b] \;\lor\; \big(R[a,u] \wedge R[v,b]\big),$$

executed only on the batch elements that took the edge.

### 3.3 Exact verification oracle

For every pair $(x, y)$, the oracle counts the elements that are an extremal (least upper / greatest lower) common bound. $P$ is a join-semilattice iff this count is exactly 1 for every off-diagonal pair, and likewise for meets; a lattice satisfies both. Verification is independent of the policy that produced the candidate. Its dominant cost is a $B \times n \times n \times n \times n$ boolean tensor, i.e. $O(B\,n^4)$ memory, which is why the batch size is reduced to 16 for $n \ge 40$.

### 3.4 Reward

The reward is sparse and terminal, supplemented by dense shaping:

| Term | Value | Purpose |
|---|---|---|
| Novelty | $+2.0$ for an unseen sampled DAG, $+0.2$ otherwise | discourages collapse onto repeated outputs |
| Per-pair difficulty shaping | $w \cdot \sum_{x \ne y} \pm\,\mathrm{fail}_{xy}\,/\,(2n)$ (lattice) | rewards resolving pairs that historically fail, via an EMA of per-pair failure rates |
| Target satisfied | $+20$ | terminal bonus for a verified structure |
| Property-aware shaping (`target=lattice`) | $-5$ distributive; $+30$ modular non-distributive; $+60$ semimodular non-modular | steers away from the easily reached Boolean/distributive class |
| Saturation penalty | $-5$ if the target is missed and $\lvert A\rvert \ge \binom{n}{2} - n$ | discourages near-total orders |
| Edge count | $+0.02\,\lvert A\rvert$ | mild density prior |

`target=join` / `meet` use only the corresponding pair-shaping term. `target=dual` rewards join and meet independently ($+10$ each). `target=boolean` adds shaping on distributivity and complementation fractions and a $+30$ bonus for a verified Boolean algebra.

### 3.5 Optimization

PPO with $K = 4$ epochs per iteration, clip range 0.2, Adam ($\eta = 2\times10^{-4}$), gradient-norm clipping at 0.5, and loss $\mathcal{L}_\text{actor} + 0.5\,\mathcal{L}_\text{critic} + 0.02\,\mathcal{L}_\text{ent}$ (all masked to genuine decisions).

Two implementation points matter for correctness and are covered by the manuscript:

- **Action replay.** The importance ratio is computed by *replaying the recorded actions* under the updated parameters. Resampling actions during the update makes the ratio meaningless and silently invalidates PPO's clipping guarantee.
- **Entropy-augmented GAE.** Generalized Advantage Estimation ($\gamma = 0.95$, $\lambda = 0.90$) receives the terminal reward at the last step plus a per-step intrinsic bonus $\iota \cdot \mathbb{1}[\text{decision}]$, with $\iota = 0.01 + 0.04\,(1 - \text{iter}/\text{iterations})$, a linear decay from 0.05 to 0.01. This acts on the advantage estimate and is separate from the entropy term in the loss.

## 4. Repository structure

```
.
├── main.py                      # CLI entry point (optional JSON config)
├── src/
│   ├── models.py                # GeneralDAGPolicy: GRU policy, masked batched generation
│   ├── diagnostics.py           # exact join/meet oracle; Boolean check; join/meet tables
│   ├── classifier.py            # modular / semimodular / distributive classification
│   ├── ppo.py                   # entropy-augmented GAE
│   ├── trainer.py               # train_and_log: rollout, reward, PPO, logging
│   ├── data_utils.py            # matrix <-> bitstring serialization
│   └── visualization.py         # Hasse-diagram rendering and Graphviz export
├── tests/                       # pytest suite
├── outputs/                     # logged runs, histories, classified result sets
├── plots/                       # generated figures
├── rare_lattice_plots/          # Hasse adjacency plots of rare structures
├── analyze_dataset.py           # classify a results CSV (modular / semimodular / distributive)
├── analyze_unique_counts.py     # duplicate and overlap audit across result files
├── summarize_results.py         # property-distribution report from the classified CSV
├── extract_rare_lattices.py     # isolate modular, non-distributive structures
├── plot_distribution.py         # discovery-yield bar chart across n
├── visualize_poset.py           # Hasse diagram (PNG) and Graphviz DOT for one row
├── visualize_rare_lattices.py   # plots for all rare modular non-distributive structures
├── visualize_n5_lattice.py      # Hasse diagram of a semimodular non-modular example
├── render_pdf.py                # vector-PDF rendering of one lattice
├── validate_citation.py         # lightweight CITATION.cff sanity check
├── CITATION.cff   LICENSE   Dockerfile   pyproject.toml   requirements.txt
```

## 5. Installation

Requirements: Python ≥ 3.10 and PyTorch ≥ 2.0 (a CUDA GPU is strongly recommended for $n \ge 30$; the code falls back to CPU automatically).

```bash
git clone https://github.com/arnauldmwafise/poset-lattice-generation-rl.git
cd poset-lattice-generation-rl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, numpy, matplotlib, pytest
pip install pandas networkx              # needed by the analysis and visualization scripts
```

The code imports `src.*`, so run all commands from the repository root.

## 6. Usage

### 6.1 Command line

```bash
python main.py                           # defaults: n=20, target=lattice
python main.py --config my_config.json   # override any parameter
```

`my_config.json` may set any of the parameters below, for example:

```json
{ "n": 30, "target": "lattice", "total_episodes": 4800, "batch_size": 16, "output_dir": "./outputs" }
```

| Parameter | Default | Meaning |
|---|---:|---|
| `n` | 20 | number of poset elements |
| `target` | `lattice` | `lattice`, `join`, `meet`, `dual`, or `boolean` |
| `total_episodes` | 4500 | total rollouts; iterations = `total_episodes // batch_size` |
| `batch_size` | 32 | rollouts per iteration (memory grows as $O(B\,n^4)$) |
| `ppo_epochs` | 4 | PPO epochs per iteration |
| `lr` | 2e-4 | Adam learning rate |
| `ema_decay` | 0.95 | decay of per-pair failure estimates |
| `pair_weight` | 2.5 | weight of per-pair difficulty shaping |
| `max_grad_norm` | 0.5 | gradient clipping threshold |
| `dist_shape_weight`, `comp_shape_weight` | 2.0, 15.0 | shaping weights for `target=boolean` |
| `output_dir` | `./outputs` | where CSV and JSON outputs are written |

Console output reports, every 20 iterations (and at iteration 1), the batch hit-count, mean reward, cumulative confirmed structures, and the adjacent-pair failure rate `gap1_fail`.

### 6.2 Python API

```python
from src.trainer import train_and_log

policy, history = train_and_log(
    n=30, target="lattice", total_episodes=4800, batch_size=16, output_dir="./outputs"
)
```

### 6.3 Inspecting and classifying a discovered structure

```python
import csv
from src.data_utils import bitstring_to_matrix
from src.classifier import analyze_lattice_properties

row = next(csv.DictReader(open("outputs/outputs_n30_lattice_v5.csv")))
n = int(row["n"])
M = bitstring_to_matrix(row["matrix_bits"], n)      # M[i, j] = 1  <=>  j <= i
print(analyze_lattice_properties(M, n))             # {'modular': ..., 'semimodular': ..., 'distributive': ...}
```

### 6.4 Analysis and visualization

```bash
python analyze_dataset.py --input outputs/outputs_n30_lattice_v5.csv   # writes lattice_analysis_results.csv
python summarize_results.py                                            # property-distribution report
python extract_rare_lattices.py                                        # rare_modular_non_distributive.csv
python analyze_unique_counts.py                                        # duplication audit of outputs/*_v5_colab.csv
python plot_distribution.py                                            # plots/dataset_distribution_yield.pdf
python visualize_poset.py --csv_path outputs/outputs_n30_lattice_v5.csv --row_index 0
python visualize_rare_lattices.py                                      # rare_lattice_plots/*.png
```

`render_pdf.py` and `visualize_n5_lattice.py` read fixed input paths defined at the top of each file; edit them to point at your own result files.

## 7. Output data format

A run with target `T` and size `n` writes to `output_dir`:

| File | Content |
|---|---|
| `outputs_n{n}_{T}_v5.csv` | one row per newly confirmed structure: `hash, n, target, matrix_bits` |
| `outputs_n{n}_join_semilattice_v5.csv`, `outputs_n{n}_meet_semilattice_v5.csv` | join-/meet-semilattices encountered during a non-`dual` run |
| `history_n{n}_{T}_v5.json` | per-iteration traces: batch hits, join/meet coverage fractions, mean reward, cumulative confirmed count, `gap1_fail` |

`target=dual` instead writes separate join and meet files.

**Encoding.** `matrix_bits` is the row-major flattening of an $n \times n$ binary matrix $M$ holding the **reflexive transitive closure** of the generated order, with $M[i,j] = 1 \iff j \le i$. Decode with `bitstring_to_matrix(bits, n)`. The matrix is the full order relation, not the Hasse (cover) relation; the visualization scripts compute the transitive reduction.

**What "confirmed" counts.** Structures are *labeled* posets on $\{0,\dots,n-1\}$, not isomorphism classes, and de-duplication uses a hash of the sampled *generating* adjacency matrix, not of the closure. Two different generating DAGs can yield the same closure, so the number of distinct closures can be smaller than the confirmed count. `hash` uses Python's built-in `hash`, which is salted per process; hash values are therefore not comparable across runs. To compare or merge result files, deduplicate on `matrix_bits`. `analyze_unique_counts.py` reports unique hashes and unique bitstrings side by side.

## 8. Reproducing the reported experiments

The reported GPU experiments were run on Google Colab by calling `train_and_log` over a grid of $(n, \text{target})$ configurations:

```python
sweeps = [
    {"n": 30, "target": "lattice", "total_episodes": 4800, "batch_size": 16},
    {"n": 40, "target": "lattice", "total_episodes": 4800, "batch_size": 16},
    {"n": 50, "target": "lattice", "total_episodes": 4800, "batch_size": 16},
    {"n": 30, "target": "join",    "total_episodes": 2400, "batch_size": 16},
    {"n": 30, "target": "meet",    "total_episodes": 2400, "batch_size": 16},
]
for c in sweeps:
    train_and_log(output_dir="./outputs", **c)
```

Lattice discoveries are then classified with `analyze_lattice_properties` into four categories: distributive; modular non-distributive; semimodular non-modular; and other lattices.

The training procedure does not fix random seeds, so individual runs are not bit-reproducible; expect run-to-run variation, particularly in the rare-class counts.

## 9. Scope and limitations

- **Single runs, no seed replication.** Each configuration in [Key results](#2-key-results) is one training run, with metrics computed at 4–16 logged checkpoints. They describe what the procedure achieved, not a confidence interval.
- **Not an ablation.** Property-aware shaping, entropy-augmented GAE, gradient clipping, and GPU execution were introduced together; the results do not isolate their individual contributions. Batch size also differs across some rows.
- **Counting convention.** Counts are of labeled structures de-duplicated by generating DAG (see [Section 7](#7-output-data-format)); they are not counts of non-isomorphic lattices and are not comparable to enumeration totals such as OEIS A006966.
- **A sampler, not an enumerator.** Unlike exhaustive enumeration, the method gives no completeness or uniformity guarantee over the set of lattices. The two are complementary: enumeration for small $n$, sampling where enumeration is infeasible.
- **Rare-class claims.** "Modular non-distributive" and "semimodular non-modular" counts are reward-event counts. Membership of the rare classes is established by `analyze_lattice_properties`; the classifier call in the trainer is wrapped in a broad exception handler, so a classifier failure leaves the sample at the base $+20$ reward rather than raising.
- **Memory.** The oracle's $O(B\,n^4)$ memory sets the practical batch size at large $n$.

## 10. Testing

```bash
pip install pytest
python -m pytest
```

## 11. Citation

If you use this software, its datasets, or the methodology, please cite the preprint (GitHub also exposes this through **Cite this repository**, driven by `CITATION.cff`).

```bibtex
@article{mwafise2026order,
  author  = {Mwafise, Arnauld Mesinga},
  title   = {Order-Agnostic Generation of Lattices via Reinforcement Learning},
  journal = {ScienceOpen Preprints},
  year    = {2026},
  doi     = {10.14293/PR2199.004125.v2},
  url     = {https://doi.org/10.14293/pr2199.004125.v2}
}
```

Mwafise, A. M. (2026). *Order-Agnostic Generation of Lattices via Reinforcement Learning*. ScienceOpen Preprints. https://doi.org/10.14293/pr2199.004125.v2

## 12. License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Arnauld Mesinga Mwafise.
