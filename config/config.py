"""Typed schema for ``config.yaml``, and the single configuration object the whole code reads.

Every hyperparameter of Part II is declared here as a field with a type and loaded from
``config/config.yaml`` at import. Nothing has a default: a key missing from the file is an
error at import rather than a silent fallback, so a run cannot proceed on a value nobody chose.

The exception is deliberate and documented at its site: the repair constants at the top of
``src/repair_optimizer.py`` and ``pipeline/repair_strategies.py`` are not in this file.
"""

from confz.base_config import BaseConfigMetaclass, BaseConfig
from confz import FileSource, FileFormat
from pathlib import Path


class GeneralConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Settings that apply to every stage.

    ``random_seed`` is what makes the train, validation and test split reproducible across the
    separate entry points that each rebuild it.
    """

    random_seed: int
    device: str


class DataConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Dataset location and the sampling of the shapes the autoencoder is trained on.

    ``sampling_min`` and ``sampling_max`` bound the half-extent per axis in metres and are
    drawn from log-uniformly, so every decade of aspect ratio is covered equally.
    """

    data_file: str
    label_names: list[str]
    n_uniform: int
    n_surface: int
    sigma: float
    data_folder: str
    sdf_output_dir: str
    sampling_min: float
    sampling_max: float


class AutoEncoderConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Shape and size of the autoencoder training set, and the width of the latent code.

    ``latent_dim`` propagates: it fixes the width of the encoder's output, of the decoder's
    input, and of the latent part of a node feature in the surrogate.
    """

    latent_dim: int
    n_shapes: int
    n_surface: int
    n_query: int
    batch_size: int
    autoencoder_folder: str
    encoder_point_mlp_dims: list[int]
    encoder_fc_dims: list[int]


class BoxEncoderConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Architecture of the encoder from three half-extents to a code.

    ``num_layers`` counts the output layer, so the number of hidden layers is one less. The
    output width is not here; it is ``autoencoder.latent_dim``.
    """

    hidden_dim: int
    num_layers: int


class sdfDecoderConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Architecture of the auxiliary signed-distance decoder of stage 1.

    Not the decoder used in the repair loop. That one maps a code to three half-extents, is
    defined in ``src/dec_box.py``, and takes its widths from its own defaults rather than from
    this file.
    """

    latent_dim: int
    hidden_dim: int
    num_layers: int
    point_dim: int
    output_dim: int
    sdf_folder: str
    use_siren: bool
    omega: float


class TrainingConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Splits and optimiser settings, shared by the autoencoder and the surrogate.

    The ``stage1_``, ``stage2_`` and ``stage3_`` entries belong to the three stages of
    ``src/enc_dec_training.py``; the remaining entries are the surrogate's.
    """

    lambda_reg: float
    sdf_clamp_delta: (
        float  # DeepSDF clamping: loss computed only in [-delta, delta] band
    )
    train_split: float
    val_split: float
    test_split: float
    batch_size_decoder: int
    batch_size_gnn: int
    epochs: int
    learning_rate: float
    patience: int
    stage1_epochs: int
    stage1_lr: float
    stage1_patience: int
    stage2_epochs: int
    stage2_lr: float
    stage2_patience: int
    stage3_epochs: int
    stage3_lr: float


class GNNConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Surrogate architecture, training monitor, and the two screwdriving criteria in metres.

    ``node_dim`` is what separates the two branches: the latent formulation of Part II has the
    code plus the two-entry role indicator, the parameter formulation of Part I the three edge
    lengths plus the same indicator. ``thresh_overlap_min`` and ``thresh_thickness_max`` are
    duplicated by hand in ``src/analytical_metrics.py``, which must not follow this file
    silently.
    """

    node_dim: int
    edge_dim: int
    hidden_dim: int
    heads: int
    head_hidden: int
    dropout: float
    output_dim: int
    gnn_folder: str
    alpha: float
    monitor_metric: str
    monitor_mode: str
    thresh_overlap_min: float
    thresh_thickness_max: float


class DesignRepairConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Repair objective and step budget. Lengths are in metres.

    ``lambda_shape`` and ``lambda_position`` weight the drift terms, which is what stops a
    repair from moving a block further than it has to. The bounds and margins the optimiser
    enforces are not all here; the rest are stated at the top of ``src/repair_optimizer.py``.
    """

    lambda_shape: float
    lambda_position: float
    lambda_reg: float
    max_z_drift: float
    max_pos_drift: float
    hinge_scale: float
    hinge_weights: list[float]
    num_steps: int
    learning_rate: float
    target_p: float
    optimizer_failed_only: bool


class ConfluenceConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """The whole configuration, grouped by the stage of the pipeline each section belongs to."""

    general: GeneralConfig
    data: DataConfig
    autoencoder: AutoEncoderConfig
    box_encoder: BoxEncoderConfig
    sdf_decoder: sdfDecoderConfig
    training: TrainingConfig
    gnn: GNNConfig
    design_repair: DesignRepairConfig


yaml_path = Path(__file__).parent / "config.yaml"

if not yaml_path.exists():
    print("Config file not found.")
else:
    config = ConfluenceConfig(
        config_sources=FileSource(file=yaml_path, format=FileFormat.YAML)
    )
