"""Graph attention surrogate that predicts whether a two-part configuration is buildable.

The model reads the configuration graph built by the dataset modules -- node 0 the frozen
anchor, node 1 the part under test -- and returns the two screwdriving quantities as physical
lengths together with a feasibility logit. It is the only learned component of Part I: the
dataset labels it, the training objective fits it, and the repair descends its gradients.

Two attention layers are shared, then the network forks into one more attention layer per
group of heads, and the read-out is taken from the part under test alone. With the widths of
``config/config.yaml`` the weights amount to 131,813 parameters, plus the two standardisation
buffers that the trainer registers on the instance.
"""

import sys
from pathlib import Path
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config


class GNN(nn.Module):
    """Feasibility surrogate of Part I, with a regression and a classification head.

    A node carries ``[size_x, size_y, size_z, role_0, role_1]``: the three edge lengths of a
    block in metres, followed by the one-hot role that separates the frozen anchor from the
    part under test. An edge carries ``[dx, dy, dz, dist_xy, dist_xz, dist_yz]``, the
    centre-to-centre offset and the three axis-pair distances, also in metres. Nothing is
    normalised on the way in; the attention layers absorb the magnitude in their weights.

    The regression head predicts the two quantities the screwdriving constraints are stated
    on, and the model is queried for both: the repair reads the regression outputs to see how
    far a design is from the thresholds, and the feasibility logit to see whether the surrogate
    calls the design buildable at all.

    The mode head is auxiliary supervision only. The repair reads the regression outputs and
    the feasibility logit and ignores it.

    The standardisation of the regression targets is not part of this class. The trainer
    registers ``target_mean`` and ``target_std`` as buffers on the instance so that they are
    saved with the weights, and every consumer de-standardises through
    :func:`src.repair_optimizer.destandardize`.

    Attributes:
        dropout_p: Dropout probability, applied between the trunk layers as well as inside the
            attention layers and the heads.
    """

    def __init__(
        self,
        node_dim: int = config.gnn.node_dim,
        edge_dim: int = config.gnn.edge_dim,
        hidden_dim: int = config.gnn.hidden_dim,
        heads: int = config.gnn.heads,
        dropout: float = config.gnn.dropout,
        head_hidden: int = config.gnn.head_hidden,
    ):
        """Build the trunk, the fork and the three heads.

        Args:
            node_dim: Width of a node feature, five: three edge lengths and the two-entry
                role indicator.
            edge_dim: Width of an edge feature, six.
            hidden_dim: Width of a node embedding after each attention layer. Each of the
                ``heads`` attention heads produces ``hidden_dim // heads`` channels, which are
                concatenated back to ``hidden_dim``.
            heads: Number of attention heads.
            dropout: Dropout probability.
            head_hidden: Width of the first layer of the regression head.
        """
        super().__init__()
        self.dropout_p = dropout

        # Two attention layers are shared by every head, so that the representation the
        # regression and the classification path start from is the same.
        self.gat1 = GATv2Conv(
            node_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.gat2 = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        # The fork is late rather than at the input: predicting a distance in millimetres and
        # deciding feasibility need different features in the last layer, but the same
        # geometric relations underneath.
        self.gat3_reg = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.gat3_cls = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.mlp_reg = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Linear(head_hidden // 2, 2),
        )

        self.mlp_binary = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.mlp_modes = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x, edge_index, edge_attr):
        """Run one batch of configuration graphs through trunk, fork and heads.

        Args:
            x: Node features, shape ``(2 * B, node_dim)``, two nodes per configuration.
            edge_index: Edge list of the batched graph, shape ``(2, 2 * B)``.
            edge_attr: Edge features, shape ``(2 * B, edge_dim)``.

        Returns:
            A tuple ``(reg_out, binary_out, mode_logits)`` for the ``B`` parts under test:

            * ``reg_out`` of shape ``(B, 2)``, the standardised overlap and thickness. The
              first column lives in logarithmic space and is recovered with an exponential;
              the second is standardised directly.
            * ``binary_out`` of shape ``(B, 1)``, the feasibility logit.
            * ``mode_logits`` of shape ``(B, 2)``, the auxiliary failure mode logits in the
              order insufficient overlap, excessive thickness.
        """
        x = F.gelu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.gelu(self.gat2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x_reg = F.gelu(self.gat3_reg(x, edge_index, edge_attr))
        x_cls = F.gelu(self.gat3_cls(x, edge_index, edge_attr))

        # Read out the part under test only. It is node 1 of every two-node graph, so the odd
        # indices of the batched tensor.
        x_reg_b1 = x_reg[1::2]
        x_cls_b1 = x_cls[1::2]

        reg_out = self.mlp_reg(x_reg_b1)
        binary_out = self.mlp_binary(x_cls_b1)
        mode_logits = self.mlp_modes(x_cls_b1)

        return reg_out, binary_out, mode_logits
