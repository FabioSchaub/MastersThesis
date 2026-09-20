# Part I — Parameter-Space Repair for Robotic Assembly

Code for Part I of the Master's thesis *Automated Repair of Generated Assemblies for a
Robotic Cell* (ETH Zurich, IDEAL lab). This branch contains the pipeline that learns a
feasibility surrogate over an assembly graph and repairs infeasible designs by descending the
gradient of that surrogate with respect to block sizes and positions.

Branch `feature/parameter-optimization`, tagged `thesis-part1` at the state cited in the
thesis. Parts II and III live on their own branches (`feature/latentspace-optimization`,
`feature/z-continuum-meshes`) and are not merged into this one; the separation mirrors the
three parts of the thesis.

## Scope

* Parts are axis-aligned boxes, described by three edge lengths.
* Feasibility is assessed per connected pair, against two screwdriving constraints: a
  horizontal contact of at least 10 mm and a joint thickness of at most 20 mm.
* The repair edits sizes and positions. It does not touch orientation, and it does not
  consider collisions between parts that share no joint.

What is **not** in this repository: the Isaac Sim scene that produces the labels, the
generative agent that proposes designs, and the motion-planning stack of the cell. Those live
in the separate simulation repository.

## Requirements

Conda environment from `environment.yaml` (Python 3.11, PyTorch 2.2.2, PyTorch Geometric,
pandas, numpy, scikit-learn, dash, plotly, wandb, tqdm). Training was run on the ETH Euler
cluster through `gnn_train.slurm`; repair and evaluation run on a CPU.

Note: `environment.yaml` is a Windows CPU-only export with pinned build strings and does not
resolve on the cluster. Re-export it before handing the code on.

## Inputs

The dataset the results of this part rest on is in the repository:

| File | What it is |
|---|---|
| `data/mixed_dataset_small_10mm.txt` | The labelled block configurations the surrogate is trained on, derived from the raw Isaac Sim output. The same file is used by Part II, so the two parts are compared on identical data. |

It is held against the ignore rule by an explicit exception in `.gitignore`, and `.gitattributes`
keeps git from rewriting its line endings — a CRLF checkout would change the file and with it
every number derived from it. The trained surrogate is in `gnn_models/`.

Still outside the repository: the raw Isaac Sim CSV this file was derived from, and the design
CSVs produced by the generative stage.

## Repository layout

```
.
├── config/                          # every hyperparameter and threshold, in one place
│   ├── config.yaml
│   └── config.py
├── src/                             # library: data, surrogate, training, repair
│   ├── simulation_dataset.py
│   ├── analytical_metrics.py
│   ├── dataset_generation.py
│   ├── gnn_dataset_preparation.py
│   ├── gnn.py
│   ├── gnn_training.py
│   ├── gnn_training_smoke.py
│   ├── repair_optimizer.py
│   ├── repair_process.py
│   └── label_analysis/              # offline scoring of a trained checkpoint
│       ├── analyze_paper_v0.py
│       └── analyze_label_generalization.py
├── pipeline/                        # the repair as a callable step, plus two viewers
│   ├── repair_strategies.py
│   ├── dashboard.py
│   └── visualize_design.py
├── tools/                           # dataset construction, batch evaluation, smoke tests
│   ├── csv_to_paper_v0_txt.py
│   ├── build_finetune_dataset.py
│   ├── build_mixed_dataset.py
│   ├── verify_dataset.py
│   ├── batch_test_repair.py
│   ├── sweep_thickness_thresholds.py
│   └── smoke_test_imports.py
├── gnn_models/                      # the trained surrogate checkpoint
│   └── gnn_small-mixed-10mm_node5_batchsize2048_20260519-092003.pth
├── pose_orientation_two_robots/     # interface to the robotic cell (see below)
│   ├── model_interface.py
│   └── ml_verifier/
│       ├── fabio_opti/optimization.py
│       └── claire_opti/geometric_optimization.py
├── docs/experiments/                # working notes from earlier experiments
│   └── 01-gap-bottleneck-plan.md
├── data/                            # datasets; empty here, ignored by git
├── gnn_train.slurm                  # cluster job that runs src/gnn_training.py
├── environment.yaml                 # conda environment
└── README.md
```

Scripts that are meant to be called are marked **Entry point.** below; everything else is
imported by them. The `__init__.py` files of `config/` and `src/` carry a docstring and
nothing else. Not shown: `__pycache__/` directories, and the CSV, text and image files that
`.gitignore` keeps out of the repository.

### `config/` — the single place hyperparameters and thresholds are read from

| File | Purpose |
|---|---|
| `config.yaml` | Holds every hyperparameter and both screwdriving limits, and names the dataset that training reads; source of the 80/10/10 split and of the repair thresholds quoted in the thesis. |
| `config.py` | Loads the YAML into one typed `config` object, so a missing or misspelled key fails at import rather than halfway through a run. |

### `src/` — the library of Part I: data handling, the surrogate, and the gradient repair

| File | Purpose |
|---|---|
| `simulation_dataset.py` | Reads the labelled dataset file and detects its block columns, and holds the box point and signed-distance samplers that the later parts build on. |
| `analytical_metrics.py` | Closed-form overlap, thickness and contact-snap formulas for a pair of boxes, used both to label configurations that never went through the simulator and to give the independent verdict a repair is checked against. |
| `dataset_generation.py` | Entry point. Builds labelled two-block configurations directly from chosen target quantities instead of from random geometry, and reports the realised distribution; source of the 7 to 200 mm block edge lengths. |
| `gnn_dataset_preparation.py` | Turns the labelled table into two-node PyTorch Geometric graphs and computes the target statistics and class weights, fixing the raw-metre and log-overlap conventions the checkpoint depends on. |
| `gnn.py` | The graph-attention surrogate: two shared layers, one further layer per group of heads, and heads for the two quantities, for feasibility and for the auxiliary failure mode; 131,813 parameters at the configured widths. |
| `gnn_training.py` | Entry point. Fits the surrogate with a regression loss weighted towards configurations near a limit, selects the checkpoint on regression error rather than on a classification score, and writes it to `gnn_models/` with its target standardisation inside. |
| `gnn_training_smoke.py` | Entry point. Runs the same training path on a small subset without a cluster, without a checkpoint and without logging. |
| `repair_optimizer.py` | The repair itself: Adam on the child's three edge lengths and three coordinates, with hinge losses on the physical quantities, a projection after every step and a periodic snap back onto the contact face; the caps and thresholds the repair actually uses are the constants at the top of this file. |
| `repair_process.py` | Entry point. Walks an assembly joint by joint, repairing each child against an already-fixed parent, and scores the outcome against the closed form so that bluffs and overcaution are counted; writes `results_paramter_optimization/repair_results_chain.csv`. |

### `src/label_analysis/` — what a trained checkpoint is worth, measured offline

| File | Purpose |
|---|---|
| `analyze_paper_v0.py` | Entry point. Scores a checkpoint on the held-out split as a regressor on the two quantities and as a classifier of feasibility, stratified by distance to the threshold; this is where the surrogate metrics quoted in the thesis come from. |
| `analyze_label_generalization.py` | Entry point. Fits a slope per band of each target range, which shows where the surrogate tracks a quantity rather than merely ordering it, and therefore where the repair has a gradient to follow. |

### `pipeline/` — the repair as a step other software can call, and two viewers

| File | Purpose |
|---|---|
| `repair_strategies.py` | Entry point. Repairs a whole design in four stages -- separate interpenetrations, snap to contact, run the surrogate-driven repair upwards from the table, place the screws -- with one CSV row in and the same row out; source of the fastener geometry (5 mm inset, 10 mm lateral, 22 mm vertical). |
| `dashboard.py` | Entry point. Serves a local page showing one design before, after contact and after the repair, with every joint listed as the surrogate sees it beside what the closed form measures. |
| `visualize_design.py` | Entry point. Serves a local page that draws one row of any design CSV in three dimensions, blocks and screws together, and writes nothing. |

### `tools/` — building the dataset, and evaluating in batch

| File | Purpose |
|---|---|
| `csv_to_paper_v0_txt.py` | Entry point. Narrows the raw simulator CSV to the two criteria of Part I, dropping rows that failed for an out-of-scope reason and relabelling the rest from the measured geometry. |
| `build_finetune_dataset.py` | Entry point. Mines the joints out of the designs the generative stage proposed and produces many analytically labelled variations around each one. |
| `build_mixed_dataset.py` | Entry point. Merges a class-balanced subsample of the simulated data with the mined data into `data/mixed_dataset_small_10mm.txt`, the 174,000 configurations used for training. |
| `verify_dataset.py` | Entry point. Recomputes every label from the geometry stored beside it and reports the distribution; it expects the earlier four-criterion schema and therefore does not apply to the merged file as it stands. |
| `batch_test_repair.py` | Entry point. Repairs every design CSV found in `pipeline/` and reports per design how many joints became feasible as measured, next to how many the surrogate claims. |
| `sweep_thickness_thresholds.py` | Entry point. Repeats the whole repair on the same designs with nothing changed but the thickness target, which is how `REPAIR_TARGET_THICKNESS_M` in `pipeline/repair_strategies.py` was chosen. |
| `smoke_test_imports.py` | Entry point. Checks the interpreter, the tensor library and the device, and that the configured dataset loads, before anything long is started. |

### `gnn_models/` — the trained surrogate

| File | Purpose |
|---|---|
| `gnn_small-mixed-10mm_node5_batchsize2048_20260519-092003.pth` | The checkpoint cited in the thesis, trained on the merged dataset; it carries its own target standardisation, so it can be loaded without the training configuration, and it is the file the parameter count of 131,813 is read from. |

The directory is otherwise ignored by git; this one checkpoint is committed so that repair and
evaluation run without a training run first.

### `pose_orientation_two_robots/` — the interface the robotic cell calls

This directory is the deployment path rather than part of the work of Part I: the CSV contract
and the surrounding structure come from the preceding group project, and the second repair
model is someone else's. It is included so that the interface is complete and both repairs can
be run on the same design. The model directories the two repairs load their weights from
(`ml_verifier/fabio_model/`, `ml_verifier/claire_model/`) are ignored by git and are not in the
repository.

| File | Purpose |
|---|---|
| `model_interface.py` | Defines the CSV contract between the design pipeline and the repair models -- geometry and screws replaced, connectivity passed through, no orientation -- and one entry point per repair model. |
| `ml_verifier/fabio_opti/optimization.py` | A deliberately self-contained copy of the repair of Part I, restating the surrogate, the closed-form geometry, the gradient repair and the screw placement so that the cell can run it without the repository around it. |
| `ml_verifier/claire_opti/geometric_optimization.py` | The second repair model behind the same interface, developed alongside this thesis with its own surrogate and scaling; not part of Part I. |

### `docs/experiments/` — working notes

| File | Purpose |
|---|---|
| `01-gap-bottleneck-plan.md` | A German planning note from the earlier latent-based branch on why the joint-gap regression was the bottleneck, listing what had been tried and what was proposed next; the gap criterion is outside the scope of Part I, so the note is kept for the record only. |

### Root

| File | Purpose |
|---|---|
| `environment.yaml` | The conda environment; a Windows CPU-only export, see the note under Requirements. |
| `gnn_train.slurm` | Entry point. The Euler batch job that runs `src/gnn_training.py` on one GPU, submitted with `sbatch`. |

## Order of execution

1. `tools/csv_to_paper_v0_txt.py` converts the raw simulator CSV into the training format.
2. `tools/build_finetune_dataset.py` jitters designs from the generative stage and labels them
   in closed form with the same two criteria.
3. `tools/build_mixed_dataset.py` merges both into `data/mixed_dataset_small_10mm.txt`, the
   174,000 configurations used for training.
4. `tools/verify_dataset.py` checks the result. It validates the column layout of the earlier
   datasets and does not apply to the merged file as it stands.
5. `src/gnn_training.py` trains the surrogate (`gnn_train.slurm` on the cluster,
   `src/gnn_training_smoke.py` for a local run).
6. `src/label_analysis/analyze_paper_v0.py` reports the surrogate metrics offline from a
   checkpoint.
7. `pipeline/repair_strategies.py` repairs one design: CSV in, repaired CSV out, fasteners
   placed. `src/repair_process.py` and `tools/batch_test_repair.py` run it in batch.
8. `pipeline/dashboard.py` shows a design before and after the repair, and
   `pipeline/visualize_design.py` renders a single design on its own.

## Where the numbers of the thesis come from

| Statement in the thesis | Source |
|---|---|
| 131,813 surrogate parameters | checkpoint in `gnn_models/`, excluding the two standardisation buffers |
| Surrogate classification and regression metrics | `src/label_analysis/analyze_paper_v0.py` on the test split |
| 174,000 configurations, split 80/10/10 | `tools/build_mixed_dataset.py`, `config/config.yaml` |
| Repair hyperparameters, thresholds, caps | `config/config.yaml` and `src/repair_optimizer.py` |
| Fastener geometry (5 mm inset, 10 mm lateral, 22 mm vertical) | `pipeline/repair_strategies.py` |
| Block edge lengths 7 to 200 mm | `src/dataset_generation.py`, `tools/build_finetune_dataset.py` |
| Benchmark over 60 prompts, ablation, physical builds | not reproducible here: they come from the pipeline as a whole, including the generative agent and the cell |

## Smoke tests

`tools/smoke_test_imports.py` and `src/gnn_training_smoke.py` run without a GPU and without
the full dataset.

## Known limitations

* The labels omit the gripper, so gripper collisions cannot be learned.
* Feasibility is assessed pairwise; a design is repaired one connection at a time.
* `config.yaml` still carries a few keys under `design_repair` that the optimiser does not
  read; the values that are actually used are the constants in `src/repair_optimizer.py`.
