# Reinforcement Learning for Finite Lattice Generation via Proximal Policy Optimization

This software implements a non-sequential, random-ordered pair sampling policy framework to explore combinatorial Directed Acyclic Graph (DAG) architectures optimized toward specific order-theoretic targets (Join/Meet Semilattices, Lattices, and Boolean Algebras).

## Citation

If you use this software, datasets, or methodology in your research, please cite the following preprint:

### BibTeX
```bibtex
@article{mwafise2026order,
  author    = {Mesinga Mwafise, Arnauld},
  title     = {Order-Agnostic Generation of Lattices via Reinforcement Learning},
  journal   = {ScienceOpen Preprints},
  year      = {2026},
  doi       = {10.14293/PR2199.004125.v1},
  url       = {https://github.com}
}
```

### APA Style
Mwafise, A. M. (2026). Order-Agnostic Generation of Lattices via Reinforcement Learning. *ScienceOpen Preprints*. https://doi.org

## Mathematical Formulation

### 1. Finite Posets and Lattices
Let $P = (X, \le)$ be a finite partially ordered set (poset).
* $P$ is a **join-semilattice** if every pair $x,y \in X$ has a unique least upper bound (supremum), denoted $x \vee y$.
* $P$ is a **meet-semilattice** if every pair $x,y \in X$ has a unique greatest lower bound (infimum), denoted $x \wedge y$.
* $P$ is a **lattice** if it is both a join- and meet-semilattice.

### 2. Distributivity and Boolean Algebraic Targets
A lattice $L$ is **Boolean** if it is both distributive:
$$x \wedge (y \vee z) = (x \wedge y) \vee (x \wedge z) \quad \forall x,y,z \in L$$
and complemented:
$$\forall x \in L, \exists! y \in L \text{ s.t. } x \wedge y = \mathbf{0} \text{ and } x \vee y = \mathbf{1}$$

## Reinforcement Learning Approach
Instead of mapping actions linearly across an incremental node assembly path, this software samples a stochastic permutation of $\binom{n}{2}$ pairs. For each evaluated pair $(u, v)$, the policy generates an edge action $A \in \{ \emptyset, u \to v, v \to u \}$. 

The structural masking engine runs incremental reachability updates via automated transitive closure maps to maintain strict acyclicity guarantees throughout the rollout duration.
