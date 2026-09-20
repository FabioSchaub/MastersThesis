"""Typed schema for ``config.yaml``, and the single configuration object the whole code reads.

Every hyperparameter is declared here as a field with a type and loaded from
``config/config.yaml`` at import. Only the five Least Volume entries of :class:`TrainingConfig`
carry a default; for every other field a key missing from the file is an error at import rather
than a silent fallback, so a run cannot proceed on a value nobody chose.

Two groups of settings live elsewhere on purpose. The Least Volume sweep passes its settings as
environment variables, which ``src/enc_dec_training.py`` reads before falling back to the fields
here, so ``lambda_vol`` in the file is not the value the thesis reports. And the repair is
configured by the default arguments of ``repair_pair`` in ``src/sim_repair_optimizer.py``, not
by the ``design_repair`` section below.

The schema is shared with the earlier parts of the thesis so that their branches can read one
file. Several sections therefore describe the box pipeline and are not read here; each says so
at the class that declares it.
"""

from confz.base_config import BaseConfigMetaclass, BaseConfig
from confz import FileSource, FileFormat
from pathlib import Path


class GeneralConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Settings that are not tied to one stage.

    ``random_seed`` seeds the autoencoder training and the split it draws. The surrogate takes
    its seed from the command line instead, and ``device`` is not read on this branch: every
    entry point selects the card or the processor by availability.
    """

    random_seed: int
    device: str


class DataConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Dataset location and shape sampling of the box pipeline of the earlier parts.

    Not read on this branch. The shapes of the vocabulary are drawn from the dimensionless
    ranges declared in ``src/enc_dec_dataset_generation.py``, and the files produced by the
    simulator are named on the command line of the tools that read them.
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
    """Size of the autoencoder training set, its sampling per shape, and the width of the code.

    ``latent_dim`` fixes the width of the encoder's output and has to agree with
    ``sdf_decoder.latent_dim``, which fixes the width the decoder expects. ``n_shapes``,
    ``n_surface``, ``n_query`` and ``batch_size`` are the quantities the memory estimate of
    ``tools/preflight_autoencoder.py`` is computed from.
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
    """Architecture of the encoder from three half extents to a code.

    Read only by ``src/enc_box.py``, the encoder of the earlier parts; the shapes of this
    branch are encoded from surface points instead. ``num_layers`` counts the output layer, so
    the number of hidden layers is one less. The output width is not here; it is
    ``autoencoder.latent_dim``.
    """

    hidden_dim: int
    num_layers: int


class sdfDecoderConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Architecture of the signed distance decoder, which is the decoder of this branch.

    ``use_siren`` and ``omega`` select the sinusoidal variant. The spectral variant that the
    Least Volume compression needs is not chosen here but through the environment of the sweep,
    which is why the checkpoint records which variant was active.
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
    """Splits, the settings of the three autoencoder stages, and the Least Volume compression.

    The splits, ``lambda_reg``, ``sdf_clamp_delta`` and the ``stage1_``, ``stage2_`` and
    ``stage3_`` entries are what ``src/enc_dec_training.py`` reads. The remaining optimiser
    entries belong to the earlier parts; the surrogate of this branch takes its epochs, batch
    size, learning rate and patience from the command line.

    The five Least Volume entries are the only fields of this file with a default, so that a
    configuration written before the compression existed still loads, and the sweep overrides
    them per run through the environment.
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
    # Least Volume (Chen & Fuge, ICLR 2024): compress the Stage-1 latent codes
    # onto their intrinsic dimension. 0.0 = off (default = unchanged behaviour).
    lambda_vol: float = 0.0
    vol_eta: float = 0.1
    # Spectral norm = the paper's anti-collapse mechanism: a Lipschitz-bounded
    # decoder cannot scale up to compensate for sigma->0, so the volume penalty
    # cannot trivially collapse the latent. lv_enc_spectral is the prof's extra
    # (bi-Lipschitz encoder). lipschitz_k scales the 1-Lipschitz decoder output
    # back to SDF range. All off by default (baseline codec unchanged).
    lv_spectral: bool = False
    lv_enc_spectral: bool = False
    lipschitz_k: float = 1.0


class GNNConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Surrogate architecture, the training monitor, and the two screwdriving criteria in metres.

    ``edge_dim``, ``hidden_dim``, ``heads``, ``head_hidden`` and ``dropout`` size both models of
    ``src/gnn.py``. ``node_dim``, ``output_dim``, ``alpha`` and the two ``monitor_`` entries
    describe the surrogate of the earlier parts: the one here reads the width of a node from the
    table of codes and has a fixed set of heads. ``thresh_overlap_min`` and
    ``thresh_thickness_max`` are the values the simulator labelled with and are read by
    ``src/sim_repair_optimizer.py``, which is what keeps its geometric check consistent with the
    labels the surrogate was trained on.
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
    """Repair objective and step budget of the earlier parts. Lengths are in metres.

    Not read on this branch. The repair of Part III is configured by the default arguments of
    ``repair_pair`` in ``src/sim_repair_optimizer.py``, and the two geometric thresholds it
    checks against come from ``gnn`` above.
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


# Main Config Class
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
