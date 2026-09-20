# Part II — Latent-Space Repair: A Proof of Concept

Code for Part II of the Master's thesis *Automated Repair of Generated Assemblies for a
Robotic Cell* (ETH Zurich, IDEAL lab). This branch repeats the repair of Part I with one
component exchanged: the optimiser holds a learned latent code of a block instead of its three
edge lengths. Everything else, the dataset, the loss, the bounds, the stopping rule and the
analytical corrections, is inherited from Part I unchanged, so that a difference in behaviour
can be attributed to the optimisation variable and to nothing else.

Branch `feature/latentspace-optimization`, tagged `thesis-part2`. Parts I and III live on
their own branches and are not merged in.

## Scope

* Parts are axis-aligned boxes. The latent code describes a size and nothing else.
* Feasibility is assessed per connected pair against the same two screwdriving constraints as
  in Part I: at least 10 mm of horizontal contact, at most 20 mm of joint thickness.
* The head-to-head evaluation covers 9 designs with 54 pairs and measures the repair stage in
  isolation, not the pipeline as a whole.

## The three networks

Two of them are decoders, which is easy to confuse:

1. an auxiliary signed-distance decoder, trained in stage 1 together with one free latent code
   per training shape, purely to give the codes a geometric meaning;
2. the box encoder, distilled in stage 2 onto those codes and fine-tuned jointly in stage 3;
3. the box decoder, trained afterwards on codes of the frozen encoder, which maps a code back
   to three half-extents. This is the decoder used in the repair loop, in the drift term and
   in the projection after every step. The auxiliary decoder is set aside after stage 1.

## Repository layout

```
p2/
├── config/                            # every hyperparameter of Part II, in one typed object
├── src/                               # the networks, their training, and the repair loop
│   └── label_analysis/                # offline evaluation of a trained surrogate
├── pipeline/                          # design level: repair one CSV, or look at one
│   └── new_csv/                       # the nine evaluation designs (git-ignored, not shipped)
├── tools/                             # dataset builders, box-decoder training, diagnostics, exports
├── docs/
│   └── experiments/                   # experiment plans written before a run
├── results/                           # written by the tools; only dashboard_data/ is committed
│   └── dashboard_data/                # repair_comparison.json, the one committed result artefact
├── data/                              # datasets and design tables (git-ignored, not shipped)
├── encoder_decoder_model/             # box-encoder and box-decoder checkpoints
├── gnn_models/                        # surrogate checkpoints
├── pose_orientation_two_robots/       # interface to the robotic cell, from the group project
│   └── ml_verifier/                   # the two refinement models the cell calls
├── autoencoder_train_twostage.slurm   # cluster entry point, encoder training
├── gnn_train_latentspace.slurm        # cluster entry point, surrogate training
├── gnn_train.slurm                    # the same job without the latent checkpoint tag
├── environment.yaml
└── README.md
```

Files are listed below in the order they are reached in the pipeline, from the data through the
autoencoder and the surrogate to the repair and its evaluation, not alphabetically. Scripts that
are actually invoked are marked "Entry point."; everything else is imported. Empty `__init__.py`
package markers are omitted throughout.

### `config/` — the single source of every hyperparameter and path

| File | Purpose |
|---|---|
| `config.yaml` | Every value the code reads, with the reasoning behind the non-obvious ones recorded in end-of-line comments. |
| `config.py` | Typed schema that loads `config.yaml` at import and exposes it as the one `config` object; a missing key is an error rather than a silent default. |

### `src/` — the three networks above, the surrogate, and the repair loop

The numbering of the three networks is the one of the section above.

| File | Purpose |
|---|---|
| `simulation_dataset.py` | Reads a design table into a DataFrame and derives from its column names which blocks it holds and the factor between metres and the encoder's range. |
| `dataset_generation.py` | Entry point. Builds labelled two-block assemblies backwards from drawn target metrics, so the samples concentrate around the two thresholds instead of far from them. |
| `analytical_metrics.py` | Closed-form geometry that decides whether a pair is buildable; it is the reference the surrogate is judged against and it supplies the snap-to-contact correction. |
| `enc_dec_dataset_generation.py` | Generates the random boxes and the analytical signed-distance samples that stage 1 of the autoencoder training consumes in memory. |
| `dec_sdf.py` | Network 1, the auxiliary signed-distance decoder: takes a code and a query point and returns a distance. Used in stage 1 only and set aside afterwards. |
| `enc_box.py` | Network 2, the box encoder: an MLP from three half-extents to the latent code, and therefore the definition of the space the repair optimises in. |
| `enc_dec_training.py` | Entry point. Runs the three stages — free codes against the auxiliary decoder, distillation of the encoder onto them, joint fine-tune — and writes the encoder checkpoint. |
| `dec_box.py` | Network 3, the box decoder: maps a code back to three half-extents in log space. This is the decoder that stands inside the repair loop. |
| `gnn_dataset_preparation.py` | Turns each labelled row into a two-node graph whose nodes carry the frozen encoder's code instead of the edge lengths, and owns the encoder cache and the target standardisation. |
| `gnn.py` | The feasibility surrogate, which is not one of the three: two graph attention layers, a late fork per head group, and a read-out from the part under test alone. |
| `gnn_training.py` | Entry point. Trains the surrogate on the latent graph and writes the best epoch, with the target standardisation stored inside the checkpoint rather than in a config file. |
| `gnn_training_smoke.py` | Entry point. Trains a few epochs on a subset using the real loss and epoch loop, logging nothing and writing no checkpoint, as a local check that the path still works. |
| `repair_optimizer.py` | The repair itself: Adam over the latent code and the position of the part under test, following the surrogate's gradient. Its top block holds the repair constants that are not in `config.yaml`. |
| `repair_process.py` | Entry point. Decomposes an assembly into a chain of pair repairs, runs them in order, records the closed-form verdict beside each prediction, and pins the surrogate checkpoint the repair runs against. |

### `src/label_analysis/` — what a trained surrogate is worth, recomputed from a checkpoint

| File | Purpose |
|---|---|
| `analyze_paper_v0.py` | Entry point. Reports the latent surrogate's regression and decision metrics, overall and in bands around each threshold; this is the reproducible source of the surrogate metrics quoted for Part II. |
| `analyze_label_generalization.py` | Entry point. The same band analysis for a surrogate of the parameter branch, whose nodes are five entries wide; it builds those node features itself and does not read the latent surrogate. |

### `pipeline/` — one design in, one repaired design out

| File | Purpose |
|---|---|
| `repair_strategies.py` | Entry point. Repairs one whole design CSV in four stages — resolve penetrations, snap faces, repair in layers from the table upwards, place screws — and holds `POST_SHRINK_M`, the additional millimetre after the final snap. |
| `dashboard.py` | Entry point. Serves one design in three states with a per-pair table that puts the surrogate's prediction next to the closed-form measurement, flagging bluffs and over-cautious refusals. |
| `visualize_design.py` | Entry point. Plain browser viewer for a design CSV, blocks and screws only; it computes no metric and passes no judgement on feasibility. |

### `tools/` — datasets, the box-decoder training, diagnostics, and the result exports

| File | Purpose |
|---|---|
| `csv_to_paper_v0_txt.py` | Entry point. Reduces raw simulator output to the two-criterion training set, dropping the failure modes this work does not certify and recomputing the labels from the recorded geometry. |
| `build_finetune_dataset.py` | Entry point. Synthesises two-block variations around the pairs found in the design CSVs, to cover the aspect ratios the simulated set covers thinly. |
| `build_mixed_dataset.py` | Entry point. Concatenates a stratified subset of the simulated set with all of the design-derived one into the single file `config.data.data_file` points at. |
| `verify_dataset.py` | Entry point. Recomputes every label of a training file from the stored geometry and exits non-zero on a mismatch, so it can gate a training run; it expects the older, wider column schema. |
| `train_box_decoder.py` | Entry point. Trains the box decoder (network 3) on codes of the frozen encoder and writes the checkpoint whose `val_mae_mm` and `val_rel_err` fields are the reconstruction figures quoted in the thesis. |
| `codec_floor_test.py` | Entry point. Measures the thinnest block the box encoder and box decoder can still represent together, which decides whether the thickness criterion is reachable at all. |
| `codec_aspect_test.py` | Entry point. Sweeps a block from near-cube to extreme slab and reports the encoder-decoder roundtrip error on the contact axis alone. |
| `check_encoder_range.py` | Entry point. Tests reconstruction, smoothness and injectivity of the box encoder below its training range, where the smallest blocks of a design land; the only tool that uses the auxiliary signed-distance decoder. |
| `latent_thickness_probe.py` | Entry point. Trains a small probe from the code to the thickness alone, to separate a code that cannot carry thickness from a surrogate that fails to read it. |
| `gnn_overlap_bias.py` | Entry point. Sweeps overlap through the 10 mm threshold on synthetic pairs and reports the surrogate's local slope and signed bias right at the criterion. |
| `gnn_thickness_bias.py` | Entry point. The same for thickness at 20 mm, repeated for four footprints, since a prediction below the true value near the threshold is the mechanism behind a bluff. |
| `repair_csv_sweep.py` | Entry point. Runs the full latent repair over every design CSV and prints each joint before and after with both verdicts, the analytical one and the surrogate's, counting the bluffs per design and in total. |
| `batch_test_repair.py` | Entry point. The coarser counterpart: per design, how many pairs the analytical check and the surrogate call feasible before and after the repair. |
| `repair_thickness_trace.py` | Entry point. Traces one production repair step by step on pairs that fail on thickness alone, logging the decoded thickness against the predicted one to see whether the optimiser can really thin a beam. |
| `sweep_thickness_thresholds.py` | Entry point. Repeats the repair with the optimiser's thickness target tightened below the real criterion, to see how much of the surrogate's bias that buys back. |
| `export_repair_comparison.py` | Entry point. Produces `results/dashboard_data/repair_comparison.json`, the four geometry states per pair from which the central table of Part II is derived; run twice with `--stage dump`, once per branch, then once with `--stage merge`. |
| `repair_dashboard.py` | Entry point. Shows one design in three states on a shared camera, reading only that JSON and no checkpoint; source of the before-and-after figure of Part II. |
| `smoke_test_imports.py` | Entry point. Smallest check that the environment is intact and the training file can be read, without touching a checkpoint. |
| `smoke_test_graph.py` | Entry point. Builds one latent graph end to end and asserts that the node width equals `latent_dim + 2` and agrees with `config.gnn.node_dim`. |
| Remaining files: `make_gnn_dataset.py`, `make_slab_boundary_dataset.py`, `make_slab_finetune_dataset.py` | Entry points. Three generators of synthetic, slab-enriched training sets. `config.data.data_file` records that these were reverted in favour of the simulation-derived set, so none of them produced the reported surrogate. |

### `pose_orientation_two_robots/` — the interface to the robotic cell

This directory comes from the group project the thesis is embedded in and is largely not the
work of this thesis. It is kept here because the latent repair has to run inside it. The
checkpoints these files expect are git-ignored and are not in the repository.

| File | Purpose |
|---|---|
| `model_interface.py` | The CSV contract between the cell and the refinement models, and the two entry functions the cell calls. From the group project. |
| `ml_verifier/fabio_opti/optimization.py` | The latent repair of this branch packaged behind that contract: the same four stages as `pipeline/repair_strategies.py`, but self-contained and loading its three checkpoints from a sibling model folder. |
| `ml_verifier/claire_opti/geometric_optimization.py` | The second model behind the same contract, a different surrogate optimisation contributed by a project partner. Not part of this thesis. |

### Cluster entry points and directories that hold artefacts rather than code

| Path | Purpose |
|---|---|
| `autoencoder_train_twostage.slurm` | Entry point. Submits `src/enc_dec_training.py` on Euler with one GPU and a 24-hour limit. |
| `gnn_train_latentspace.slurm` | Entry point. Submits `src/gnn_training.py` on Euler and sets `GNN_EXP_NAME=latentspace`, which tags the checkpoint so it cannot be confused with a parameter-branch one. |
| `gnn_train.slurm` | Entry point. The same job without that tag, inherited from Part I and kept for comparability. |
| `environment.yaml` | Conda export of the environment; see the note under Requirements before reusing it on the cluster. |
| `encoder_decoder_model/` | `best_encoder_decoder_latentdim8.pth` holds the box encoder and the auxiliary decoder, `best_box_decoder_latentdim8.pth` the box decoder. Both are committed. |
| `gnn_models/` | The surrogate checkpoint the repair runs against, pinned by name in `src/repair_process.py`. |
| `results/dashboard_data/` | `repair_comparison.json`, deliberately committed; the tools also write to further subfolders of `results/`, which are not. |
| `docs/experiments/` | Plans written before a run, kept as a record of what was tried and why. |
| `data/`, `pipeline/new_csv/` | Datasets and the nine evaluation designs. Git-ignored, see the section above. |

## Requirements

Conda environment from `environment.yaml` (Python 3.11, PyTorch 2.2.2, PyTorch Geometric,
dash, plotly, wandb). Repair, evaluation and the dashboard run on a CPU; training uses the
`*.slurm` entry points on the ETH Euler cluster. Re-export `environment.yaml` before handing
the code on: the committed file is a Windows CPU-only export and does not resolve on the
cluster.

## Inputs

The training set is in the repository:

| File | What it is |
|---|---|
| `data/mixed_dataset_small_10mm.txt` | The labelled block configurations the surrogate is trained on. Byte-identical to the file used in Part I, which is what makes the two repair variants comparable: the substitution of the representation is the only difference between them. |

It is held against the ignore rule by an explicit exception in `.gitignore`, and `.gitattributes`
keeps git from rewriting its line endings — a CRLF checkout would change the file and with it
every number derived from it. The three checkpoints are committed as well, so the repair of this
part runs without the cluster.

Still outside the repository: the nine design CSVs of the generative stage.

## Order of execution

1. `src/enc_dec_training.py` runs the three stages and writes the encoder checkpoint
   (`autoencoder_train_twostage.slurm` on the cluster).
2. `tools/train_box_decoder.py` trains the decoder from a code back to half-extents, on codes
   of the frozen encoder.
3. `src/gnn_training.py` trains the surrogate on the latent graph
   (`gnn_train_latentspace.slurm`).
4. `pipeline/repair_strategies.py` repairs one design; `tools/repair_csv_sweep.py` runs the
   whole set.
5. `tools/export_repair_comparison.py` produces the head-to-head comparison: twice with
   `--stage dump`, once per branch, then once with `--stage merge`. Running the parameter
   branch needs a checkout of `feature/parameter-optimization`.
6. `tools/repair_dashboard.py` shows one design in three states on a shared camera. This is
   the source of the before-and-after figure of the thesis.

## Where the numbers of the thesis come from

| Statement in the thesis | Source |
|---|---|
| Reconstruction error 0.309 mm and 1.58 % | fields `val_mae_mm` and `val_rel_err` in the decoder checkpoint |
| 45 of 54 pairs repaired in the latent branch, 37 of 54 in the parameter branch | `results/dashboard_data/repair_comparison.json` |
| Bluffs 1 against 3, over-cautious pairs 0 against 4, breakdown by design, McNemar contingency 32/13/5/4 | the same file |
| Surrogate classification and regression metrics | training run; reproducible offline from a checkpoint with `src/label_analysis/analyze_paper_v0.py` |
| Repair constants that are not in the config | the block at the top of `src/repair_optimizer.py` |
| The additional millimetre after the final snap | `POST_SHRINK_M` in `pipeline/repair_strategies.py` |

`results/dashboard_data/repair_comparison.json` is deliberately kept in the repository: it is
the only artefact from which the central table of Part II can be re-derived without rerunning
both branches.

## Diagnostics

`tools/gnn_overlap_bias.py` and `tools/gnn_thickness_bias.py` report the regression bias at the
threshold, `tools/codec_floor_test.py` and `tools/codec_aspect_test.py` probe what the
autoencoder can represent, `tools/repair_thickness_trace.py` traces a single repair step by
step, and `tools/latent_thickness_probe.py` walks the latent space along one axis. Smoke tests:
`tools/smoke_test_imports.py`, `tools/smoke_test_graph.py`, `src/gnn_training_smoke.py`.

## Known limitations

* The pairwise decomposition cannot express a member that has to satisfy two parents at once.
* The evaluation is geometric: it certifies the two screwdriving criteria and is silent about
  everything else.
* 54 pairs are too few for the head-to-head difference to be significant; the paired test
  reports a trend.
* `config.yaml` names an older checkpoint in a comment; the one actually used is pinned in
  `src/repair_process.py`.
