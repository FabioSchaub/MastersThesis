"""Graph attention surrogate that predicts whether a two-part configuration is buildable.

Two graph attention layers over the configuration graph, then a late fork into one more
attention layer per group of heads, and a read-out taken from the part under test alone. The
model is the one of Part I with a single substitution: a node carries the learned code of a
block instead of its three edge lengths. Widths and head counts are unchanged, so a difference
in behaviour is attributable to the node feature and not to a differently sized network.

It is trained by ``src/gnn_training.py`` on the graphs of ``src/gnn_dataset_preparation.py``,
and it is the function whose gradients the repair of ``src/repair_optimizer.py`` follows.
"""

import sys
from pathlib import Path
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config


class GNN(nn.Module):
    """Feasibility surrogate of the latent branch, with a regression and a classification head.

    The node feature is ``[z_0..z_7, node_type_0, node_type_1]``: the code of the block
    produced by the frozen :class:`~src.enc_box.BoxEncoder` from its half-extents, followed by
    a two-entry one-hot marking base against part under test. The parameter branch of Part I
    uses ``[size_x, size_y, size_z, node_type_0, node_type_1]`` in the same slot.

    The edge feature holds the centre-to-centre offset and the three axis-pair distances,
    always in metres.

    The mode head is auxiliary supervision only. The repair reads the regression outputs and
    the feasibility logit and ignores it.

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
            node_dim: Width of a node feature. Ten in the latent formulation, five in the
                parameter one.
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
            * ``mode_logits`` of shape ``(B, 2)``, the auxiliary logits for insufficient
              overlap and excessive thickness.
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
