"""Load ``config.yaml`` into one typed object that the rest of the code imports.

Every hyperparameter and every threshold lives in the YAML file; these classes only declare
what it has to contain. Declaring the fields makes a missing or misspelled key fail at import
rather than halfway through a run, and it is what lets the thresholds be read from the same
place by the training objective, the repair and the offline analysis.

The module exposes a single ``config`` object. Sections that belong to the other parts of the
thesis, such as the autoencoder and the decoder, are still declared because the file is shared
across branches. Part I reads ``general``, ``data``, ``training``, ``gnn``, and four keys of
``design_repair``; the remaining repair keys are declared so the file validates, but the
values the repair actually uses are the constants at the top of ``src/repair_optimizer.py``.
"""

from confz.base_config import BaseConfigMetaclass, BaseConfig
from confz import FileSource, FileFormat
from pathlib import Path


class GeneralConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Seed and device. The seed also fixes the train, validation and test split."""

    random_seed: int
    device: str


class DataConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Which dataset file is used, and how shapes are sampled for the other parts."""

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
    """Shape autoencoder: code width, how many shapes are sampled, and the encoder layers.

    Declared so the shared file validates. Part I describes a part by its three edge lengths
    and never encodes anything, so nothing on this branch reads these values.
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
    """Width and depth of the box encoder. Not read on this branch; it belongs to Part II."""

    hidden_dim: int
    num_layers: int


class sdfDecoderConfig(BaseConfig, metaclass=BaseConfigMetaclass):
    """Distance-field decoder: code width, layer sizes, and the periodic activation settings.

    Declared so the shared file validates. Not read on this branch.
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
    """Optimiser settings and the split ratios, shared by every trained model of the thesis."""

    lambda_reg: float
    sdf_clamp_delta: (
        float
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
    """Shape of the surrogate, and the two screwdriving limits in metres.

    The limits belong here rather than in the model because they are read by the training
    objective, the repair and the analysis alike, and all three have to agree on them.
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
    """Settings of the repair. Only ``lambda_shape``, ``lambda_position``, ``num_steps`` and
    ``learning_rate`` reach the optimiser as defaults; the rest are declared so the file
    validates and are not read on this branch."""

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
    """The whole configuration file."""

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
