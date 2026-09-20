# Part III — Generalisation to a Vocabulary of Fourteen Shapes

Code for Part III of the Master's thesis *Automated Repair of Generated Assemblies for a
Robotic Cell* (ETH Zurich, IDEAL lab). This branch carries the substitution of Part II to its
intended destination: the optimiser holds a learned code that describes the *form* of a part,
alongside a uniform scale and a placement that carry its size.

The central result is negative and this code is what produced it. Repairing size and placement
while holding the form fixed transfers to physics unchanged, at 35 % feasible configurations.
Freeing the shape code raises the confidence of the surrogate to about 0.97 while real
feasibility under physics falls to 5 %: the optimiser improves the model instead of the object.

Branch `feature/z-continuum-meshes`, tagged `thesis-part3`. Parts I and II live on their own
branches and are not merged in.

## What is here and what is not

This repository holds the learning side: the shape vocabulary, the autoencoder, the Least
Volume compression, the dataset conversion, the feasibility surrogate, the repair and the
figure tooling.

The physics lives in a **separate simulation repository** (branches `dataset_stls` and
`dataset_stls-z-continuum`). It produces the labels, and it replays the repaired
configurations for the final validation. The two repositories meet at exactly two points:
meshes are exported to the simulator (`tools/export_shape_objs.py`), and a labelled CSV plus
the replay verdict come back.

Run the tools as modules from the repository root, for example
`python -m tools.verify_shape_vocab`, so that `src` is importable.

## Repository layout

```
Code/
├── src/                              # the pipeline as importable modules
│   ├── shape_primitives.py
│   ├── enc_dec_dataset_generation.py
│   ├── enc_pointnet.py
│   ├── enc_box.py
│   ├── dec_sdf.py
│   ├── least_volume.py
│   ├── enc_dec_training.py
│   ├── validate_shape_encoder.py
│   ├── sim_gnn_dataset.py
│   ├── gnn.py
│   ├── sim_gnn_training.py
│   └── sim_repair_optimizer.py
├── tools/                            # the commands: gates, exports, evaluations, figures
│   ├── verify_shape_vocab.py
│   ├── export_shape_objs.py
│   ├── preflight_autoencoder.py
│   ├── validate_vocab_recon.py
│   ├── codec_quality_report.py
│   ├── precompute_shape_latents.py
│   ├── sim_csv_to_txt.py
│   ├── sim_gnn_eval_by_shape.py
│   ├── sim_repair_demo.py
│   ├── sim_repair_export.py
│   ├── z_drift_scales.py
│   ├── export_shape_repair_views.py
│   └── shape_repair_dashboard.py
├── config/                           # every hyperparameter, in one typed file
│   ├── config.py
│   └── config.yaml
├── docs/                             # the record of the experiments behind the runs
│   ├── SESSION_HANDOFF.md
│   └── experiments/                  # 01 to 06, one file per experiment
├── autoencoder_general.slurm         # cluster entry point: the autoencoder and the sweep
├── sim_gnn_train.slurm               # cluster entry point: the surrogate
├── env_setup.sh                      # the cluster environment, sourced by both
├── environment.yaml                  # the local environment
├── README.md
└── data/  encoder_decoder_model/  gnn_models/  logs/  wandb/
                                     # written while running, ignored by git
```

### `src/` — the vocabulary, the autoencoder, the dataset, the surrogate and the repair

| File | Purpose |
|---|---|
| `shape_primitives.py` | The analytic signed distance functions of the members of the vocabulary that are extrusions of a plane section. |
| `enc_dec_dataset_generation.py` | Defines the fourteen-shape vocabulary and generates one training sample per shape: surface points for the encoder, query points and distances for the decoder. |
| `enc_pointnet.py` | The point cloud encoder that maps a sampled surface, in the canonical frame, to a latent code. |
| `enc_box.py` | The encoder of the earlier parts, from three half extents to a code; not used here, kept so that the branches share one training file. |
| `dec_sdf.py` | The decoder that turns a code and a query point into a signed distance, in a weight-normalised, a sinusoidal or a spectrally normalised variant. |
| `least_volume.py` | The volume penalty and the Lipschitz bound of the Least Volume compression, and the read-out of how many entries of a code stay active. |
| `enc_dec_training.py` | Entry point. Trains the autoencoder in three stages and applies the compression in the third, with the settings of the sweep read from the environment. |
| `validate_shape_encoder.py` | Entry point. Measures the reconstruction of externally generated parts, and supplies the canonical normalisation the tools of the vocabulary reuse. |
| `sim_gnn_dataset.py` | Turns the labelled rows of the simulator into two-node graphs: code, bounding box and role per node, offset and plane distances per edge. |
| `gnn.py` | Both surrogates: `GNN`, the model of Parts I and II, and `SimAssemblyGNN`, the model of this part, which reads a shape code and predicts the three failure modes and the overall verdict. |
| `sim_gnn_training.py` | Entry point. Trains the surrogate on the labelled configurations and writes the input standardisation and the per-target thresholds into the checkpoint. |
| `sim_repair_optimizer.py` | The repair itself: the descent over size, placement and shape code through the gradient of the surrogate, and the discrete search over the vocabulary. |

### `tools/` — the commands of the pipeline, one per step of the table above

| File | Purpose |
|---|---|
| `verify_shape_vocab.py` | Entry point. Checks that every shape generates and samples consistently, and exits non-zero if one does not. |
| `export_shape_objs.py` | Entry point. Marches every distance field into one watertight mesh, which is the first of the two handovers to the simulator. |
| `preflight_autoencoder.py` | Entry point. Builds the model locally, pushes one batch through it and estimates the memory of a training run against the budget of the card. |
| `validate_vocab_recon.py` | Entry point. Reports the reconstruction error per shape, in canonical units and in millimetres, and gates the next step through its exit code. |
| `codec_quality_report.py` | Entry point. Describes the compressed set of codes by spectrum, correlation and spread, and tests it by emptying the unused entries. |
| `precompute_shape_latents.py` | Entry point. Encodes the exported meshes once into the table of codes that the dataset and the repair look up. |
| `sim_csv_to_txt.py` | Entry point. Reduces the simulator's output to the columns the surrogate reads and writes the file that is copied to the cluster. |
| `sim_gnn_eval_by_shape.py` | Entry point. Scores the trained surrogate on the split the training used, overall and broken down by the shape of the part under test. |
| `sim_repair_demo.py` | Entry point. Repairs the same infeasible configurations once per repair mode and once by discrete shape search, and prints the comparison of Part III. |
| `sim_repair_export.py` | Entry point. Writes `repair_configs.json`, one entry per branch and configuration, which is the second handover to the simulator and the basis of the verdict under physics. |
| `z_drift_scales.py` | Entry point. Puts the drift of the shape code on a scale — the length of a code, the spacing of the training codes, the reach of the shape continuum — and counts the repaired codes that end up outside it. |
| `export_shape_repair_views.py` | Entry point. Rebuilds the three states of every configuration from the export and writes the bundle behind the two repair figures. |
| `shape_repair_dashboard.py` | Entry point. Dash application showing the three states side by side on a shared camera; the two repair figures are screenshots of it. |

### `config/` — every hyperparameter of the pipeline

| File | Purpose |
|---|---|
| `config.py` | The typed schema of the configuration, and the object the whole code imports; a key missing from the file is an error at import. |
| `config.yaml` | The values, with the reason for each in a comment beside it. `lambda_vol` here is not the setting of the thesis; the sweep overrides it through the environment. |

### `docs/` — why the runs look the way they do (written in German)

| File | Purpose |
|---|---|
| `SESSION_HANDOFF.md` | State of the extension to general shapes before the vocabulary was fixed, kept as the prehistory of the current pipeline. |
| `experiments/01-gap-bottleneck-plan.md` | Diagnosis of the gap regression of the earlier surrogate and the plan drawn from it. |
| `experiments/02-shape-extension-plan.md` | The roadmap from the box-only pipeline to general shapes. |
| `experiments/03-least-volume-latent.md` | The Least Volume experiment: why the earlier latent failed the repair, and how the compression was set up. |
| `experiments/04-vocab-and-targets.md` | The decision for the fourteen shapes and the richer targets, and the autoencoder training that followed. Appendix A of the thesis cites this file. |
| `experiments/05-sim-dataset-generation.md` | Generation of the dataset in the simulator: the two-part model, the shapes, the labelled failure modes. |
| `experiments/06-gnn-repair-adaptation.md` | Adaptation of the surrogate and of the repair to that dataset, step by step. |

### Repository root — the cluster entry points and the environment

| File | Purpose |
|---|---|
| `autoencoder_general.slurm` | Entry point. Submits the autoencoder training on the cluster; the sweep of step 5 runs this same script with environment overrides. |
| `sim_gnn_train.slurm` | Entry point. Submits the surrogate training; it names the smaller of the two datasets, which is the adjustment the note on step 9 describes. |
| `env_setup.sh` | Modules, virtual environment and variables of the cluster, sourced by both submit scripts. |
| `environment.yaml` | The local conda environment, the one the *Requirements* section below describes. |

Remaining files: `.gitignore`, and the empty `__init__.py` of `src/` and `config/` that make the
two folders importable. The folders `data/`, `encoder_decoder_model/`, `gnn_models/`, `logs/` and
`wandb/` hold what a run reads and writes and are ignored by git; see *Not in the repository*.

## Requirements

Conda environment from `environment.yaml`, or `env_setup.sh` on the ETH Euler cluster
(Python 3.11, PyTorch 2.2.2, PyTorch Geometric, trimesh, scikit-image, dash, plotly, wandb).
Training the autoencoder and the surrogate needs a GPU with 24 GB; the repair, the evaluation
and the dashboard run on a CPU. Re-export `environment.yaml` before handing the code on: the
committed file is a Windows CPU-only export and does not resolve on the cluster.

## Order of execution

| Step | Command | Produces |
|---|---|---|
| 1 | `python -m tools.verify_shape_vocab` | gate: every shape of the vocabulary reconstructs from its distance field |
| 2 | `python -m tools.export_shape_objs --scale 100` | the meshes the simulator spawns |
| 3 | `python -m tools.preflight_autoencoder` | GPU memory estimate, mandatory before a cluster submit |
| 4 | `sbatch autoencoder_general.slurm` | the uncompressed autoencoder |
| 5 | the same script with `LV_S3_ONLY=1`, `LV_SPECTRAL=1`, `LV_LAMBDA_VOL=0.005/0.01/0.05/0.1` | the four compressed autoencoders of the sweep |
| 6 | `python -m tools.validate_vocab_recon`, `python -m tools.codec_quality_report` | reconstruction per shape, and under code truncation |
| 7 | simulation repository | the labelled CSV |
| 8 | `python -m tools.precompute_shape_latents`, `python -m tools.sim_csv_to_txt` | the code table and the training file |
| 9 | `sbatch sim_gnn_train.slurm` | the feasibility surrogate |
| 10 | `python -m tools.sim_gnn_eval_by_shape` | surrogate metrics, also per shape |
| 11 | `python -m tools.sim_repair_demo` | the three repair modes |
| 12 | `python -m tools.sim_repair_export` | `repair_configs.json` for the replay |
| 13 | simulation repository | the verdict under physics |
| 14 | `python -m tools.export_shape_repair_views`, then `python -m tools.shape_repair_dashboard` | the before-and-after figures |

Note on step 9: `sim_gnn_train.slurm` points at the smaller of the two datasets. The result
reported in the thesis was trained on the continuum file; adjust the path before submitting.

## Where the numbers of the thesis come from

| Statement in the thesis | Source |
|---|---|
| The fourteen shapes and their categories | `src/shape_primitives.py`, `src/enc_dec_dataset_generation.py` |
| Canonical normalisation onto a largest extent of 0.9 | `to_canonical` in `src/validate_shape_encoder.py` |
| Encoder 333,472 and decoder 806,913 parameters | `src/enc_pointnet.py`, `src/dec_sdf.py` |
| Least Volume regulariser and the active-dimension count | `src/least_volume.py` |
| The sweep 30 / 24 / 9 / 4 active dimensions | `autoencoder_general.slurm` with the environment overrides of step 5 |
| Reconstruction per shape, and under truncation | `tools/validate_vocab_recon.py`, `tools/codec_quality_report.py` |
| Dataset sizes and the rates per failure mode | `tools/sim_csv_to_txt.py`, `src/sim_gnn_dataset.py` |
| Surrogate architecture, 131,652 parameters, the four heads | `src/gnn.py` (`SimAssemblyGNN`) |
| Surrogate metrics | `src/sim_gnn_training.py`, re-derivable with `tools/sim_gnn_eval_by_shape.py` |
| Repair constants, the contact height, the mask of active coordinates | `src/sim_repair_optimizer.py` |
| The three repair modes | `tools/sim_repair_demo.py` |
| The verdict under physics | `tools/sim_repair_export.py` and the replay in the simulation repository |
| Figures 11 and 12 | `tools/export_shape_repair_views.py`, `tools/shape_repair_dashboard.py` |

`docs/experiments/03` to `06` record the decisions behind these runs, including the ones that
were discarded. Appendix A of the thesis cites `04` by name.

## What is shipped with the code

The artifacts the results of this part rest on are in the repository, so that the repair and
the figures can be reproduced without access to the cluster or to Isaac Sim:

| File | What it is |
|---|---|
| `data/sim_shape_dataset_1613.txt` | The labelled configurations over the fourteen shapes: 508,704 rows, 30.6 % of them feasible. |
| `data/sim_shape_continuum_2027.txt` | The same over the shape continuum: 580,640 rows, 31.7 % feasible. This is the dataset the reported surrogate was trained on. |
| `encoder_decoder_model/best_encoder_decoder_general_canon_lv_v14_lam005_latentdim32_active9.pth` | The compressed autoencoder of the thesis: fourteen shapes, volume penalty 0.05, a code of 32 entries of which 9 stay active. |
| `encoder_decoder_model/shape_latents_lv_v14_lam005.pt` | The codes of the fourteen shapes, looked up by mesh name. |
| `encoder_decoder_model/latents_continuum.pth` | The codes of the 777 continuum shapes. |
| `gnn_models/sim_gnn_shape_node37_32.pth` | The feasibility surrogate. **See the warning below.** |
| `results/repair_configs.json` | The 200 repaired configurations of the measurement run, 100 per variant, with the verdict of the geometric check. Table 12 of the thesis is derived from this file. |

They are held against the ignore rules by explicit exceptions in `.gitignore`, and `.gitattributes`
keeps git from rewriting the line endings of the datasets — a CRLF checkout would change the
files and with them every number derived from them.

**Warning about the surrogate checkpoint.** The file in this repository is an artifact of a smoke
test, not the model the thesis reports. Re-measured on the discrete test split it reaches an
OVERLAP AUROC of 0.744 against the 0.953 stated in the thesis, and the other three targets are
correspondingly lower. It loads and runs, and the repair pipeline works end to end with it, but
the numbers it produces are not the reported ones. The trained model exists only on the ETH Euler
cluster; the metrics of the thesis are reproducible from the Weights & Biases runs named in
`docs/experiments/06-gnn-repair-adaptation.md`, and `results/repair_configs.json` carries the
verdicts of the real model for the measurement run.

## Not in the repository

* The replay CSV of the physics validation (35 MB), from which the 35 % against 5 % of the final
  validation is computed.
* The bundle read by the dashboard, which `tools/export_shape_repair_views.py` rebuilds from the
  export above.
* The meshes exported to the simulator, which `tools/export_shape_objs.py` regenerates.

## Known limitations

* The capsule and the sphere stay at about 1.2 mm of reconstruction error. This is a limit of
  a point-cloud encoder built on a maximum over per-point features, not of the compression.
* The cone drops out of the shape continuum: its tip makes every perturbed mesh non-watertight.
* The thickness criterion is a threshold on one input channel, so its perfect score is not a
  modelling achievement.
* The repair and physics numbers are a single run of one hundred configurations per mode,
  without repetition across seeds.
* `config.yaml` carries `lambda_vol: 0.01`, while the selected setting of the thesis is 0.05;
  the sweep was run through environment overrides.
