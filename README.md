# Gradient-Based Geometry Repair for Robotic Assembly in Explicit and Learned Representation Spaces

Master's thesis by Fabio Schaub, IDEAL Lab, ETH Zürich.
Supervised by Dr. Millicent Schlafly and Prof. Dr. Mark Fuge.

A generated design is often not buildable by the robot cell that has to assemble it. This work
predicts buildability with a differentiable model and repairs the designs that fail by following
that model's gradients, editing the geometry of the parts rather than only their placement.

This branch holds no code. The work is organised as three branches, one per part, and this file
says what is where.

---

## The three parts

The same mechanism appears in all three: a graph network is trained as a differentiable
surrogate of assembly feasibility, and an infeasible design is repaired by descending its
gradient. What changes from part to part is only **the variable the optimiser holds**, which is
what makes the outcomes comparable.

| Part | Optimisation variable | Branch | Tag |
|---|---|---|---|
| I | The explicit edge lengths and placement of rectangular parts | `feature/parameter-optimization` | `thesis-part1` |
| II | A learned code of those same dimensions | `feature/latentspace-optimization` | `thesis-part2` |
| III | A learned code of the shape, with size beside it | `feature/z-continuum-meshes` | `thesis-part3` |

**Part I** repairs assemblies of blocks and reaches 86.7% of designs passing the simulated
assembly checks against 8.3% for repair that only moves parts. Ten of the repaired designs were
physically built by two UR5e arms.

**Part II** replaces the explicit dimensions by a learned code and compares the two variables on
identical inputs, everything else held fixed. Repairing the code succeeds on 45 of 54
connections against 37, and deceives its own surrogate less often.

**Part III** carries the substitution from the size of a part to its shape, over a vocabulary of
fourteen shapes encoded by a point-cloud encoder and a distance-field decoder. Here the two
variables separate: repairing size and placement transfers to physics unchanged, while repairing
the shape code raises the surrogate's confidence to 0.97 while true feasibility falls to 5%.

The central finding of the thesis is the condition that separates the two cases. Optimising in a
learned space is sound only where every reachable point corresponds to a valid object, where the
correspondence is effectively one-to-one, and where a drifted point can be returned to a geometry
that has actually been checked. Sizes satisfy these conditions and shapes do not, which neither
compression nor a denser training distribution changes.

---

## Where to start

Each of the three branches carries its own `README.md` describing, file by file, what the
repository contains, which scripts are entry points, and in what order they are meant to be run.
Each also states which figure or table of the thesis a given script produces.

```
git checkout feature/parameter-optimization    # Part I
git checkout feature/latentspace-optimization  # Part II
git checkout feature/z-continuum-meshes        # Part III
```

The tags `thesis-part1`, `thesis-part2` and `thesis-part3` mark the exact state of each branch
that produced the results reported in the thesis. Prefer them over the branch heads if you want
to reproduce a number.

To follow the argument rather than the code, Part II is the shortest path: it is the controlled
comparison against which everything in Part III is diagnosed.

---

## Physics simulation

The feasibility labels of Part III come from a separate simulation repository, on branches
`dataset_stls` and `dataset_stls-z-continuum`. It produces the labelled configurations and
replays the repaired ones for the final validation. The two repositories meet at exactly two
points: meshes are exported to the simulator, and a labelled table plus the replay verdict come
back.

---

## What is not in version control

Trained networks, labelled datasets, shape meshes and simulation logs are outputs rather than
sources and are archived separately. Each branch's README lists them together with the script
that produces each one, so that any of them can be regenerated.

---

## Branch layout

```
main                                  this overview only
feature/parameter-optimization        Part I
feature/latentspace-optimization      Part II
feature/z-continuum-meshes            Part III
```

The three parts are deliberately kept apart rather than merged. Each was developed against its
own dataset, checkpoints and configuration, and keeping them separate is what allows the state
behind each result to be checked out exactly.

## What each branch ships

Each part carries the data and the checkpoints its results rest on, so the repair and the
figures can be reproduced without the cluster and without Isaac Sim. Part III ships both
labelled datasets, the compressed autoencoder and the code tables; the surrogate checkpoint
there is a smoke-test artifact and does not reproduce the reported metrics — the README of that
branch says so and names where the real numbers come from.
