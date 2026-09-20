"""Graph attention surrogates that predict whether a two-part configuration is buildable.

Both models share the same shape: two graph attention layers over the configuration graph,
then a late fork into one more attention layer per group of heads, and a read-out taken from
the active part alone. They differ only in what a node carries and in what the heads predict,
which is the substitution the three parts of the thesis are built around.

    GNN               Parts I and II. A node carries either the three edge lengths of a block
                      or a learned code of its size. Predicts overlap and thickness as
                      physical quantities, plus a feasibility logit.
    SimAssemblyGNN    Part III. A node carries a learned shape code, the measured bounding box
                      and the role indicator. Predicts one probability per failure mode and one
                      overall verdict, and standardises its inputs itself.

`GNN` is not used on this branch; it is kept so that the three branches share one model file.
Both models read the graphs built by the dataset modules, in which node 0 is the frozen base
and node 1 the part under test, so the read-out takes the odd indices of a batched tensor.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config


class GNN(nn.Module):
    """Feasibility surrogate of Parts I and II, with a regression and a classification head.

    The node feature is either ``[size_x, size_y, size_z, role_0, role_1]`` in the parameter
    formulation of Part I or ``[z_0..z_7, role_0, role_1]`` in the latent formulation of
    Part II. Everything else, including the widths and the number of heads, is identical in
    both, so that a difference in behaviour is attributable to the node feature and not to a
    differently sized network.

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
            node_dim: Width of a node feature. Five in the parameter formulation, ten in the
                latent one.
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
            A tuple ``(reg_out, binary_out, mode_logits)`` for the ``B`` active parts:

            * ``reg_out`` of shape ``(B, 2)``, the standardised overlap and thickness. The
              first column lives in logarithmic space and is recovered with an exponential;
              the second is standardised directly.
            * ``binary_out`` of shape ``(B, 1)``, the feasibility logit.
            * ``mode_logits`` of shape ``(B, 2)``, the auxiliary failure mode logits.
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


class SimAssemblyGNN(nn.Module):
    """Feasibility surrogate of Part III, trained on configurations labelled in Isaac Sim.

    A node carries the shape code, the measured bounding box and the role indicator, so that
    form is described by something learned and dimensionless while size stays physical and
    observable. With the defaults of this branch and a node width of 37 the model has 131,652
    parameters, which is essentially the size of the surrogates of Parts I and II: the change
    of representation is not accompanied by a change of capacity.

    Nothing is regressed here. Overlap and thickness are not defined for the curved members of
    the shape vocabulary, so a head that regressed them would have to invent a value for the
    shapes on which the quantity does not exist. The model predicts one logit per failure mode
    and one for the overall verdict instead.

    Attributes:
        dropout_p: Dropout probability.
        x_mean, x_std, e_mean, e_std: Standardisation of the node and edge features, held as
            buffers so that they travel with the weights. They are the identity until
            :meth:`set_input_stats` is called.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = config.gnn.edge_dim,
        hidden_dim: int = config.gnn.hidden_dim,
        heads: int = config.gnn.heads,
        dropout: float = config.gnn.dropout,
        head_hidden: int = config.gnn.head_hidden,
        num_fail: int = 3,
    ):
        """Build the trunk, the fork, the two heads and the standardisation buffers.

        Args:
            node_dim: Width of a node feature: the shape code, three bounding box entries and
                the two-entry role indicator, so 37 for a code of width 32.
            edge_dim: Width of an edge feature, six.
            hidden_dim: Width of a node embedding after each attention layer, split over the
                heads as in :class:`GNN`.
            heads: Number of attention heads.
            dropout: Dropout probability.
            head_hidden: Width of the first layer of the failure mode head.
            num_fail: Number of failure modes, three: insufficient overlap, excessive
                thickness and tipping.
        """
        super().__init__()
        self.dropout_p = dropout

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

        self.gat3_fail = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.gat3_good = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.mlp_fail = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Linear(head_hidden // 2, num_fail),
        )

        self.mlp_good = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # The standardisation lives inside the model rather than in whatever code prepares the
        # data. The repair builds its inputs itself, and because these buffers are saved with
        # the weights it cannot end up querying the model under a different normalisation than
        # the one it was trained under.
        self.register_buffer("x_mean", torch.zeros(node_dim))
        self.register_buffer("x_std", torch.ones(node_dim))
        self.register_buffer("e_mean", torch.zeros(edge_dim))
        self.register_buffer("e_std", torch.ones(edge_dim))

    @torch.no_grad()
    def set_input_stats(self, x_mean, x_std, e_mean, e_std):
        """Fill the standardisation buffers from statistics of the training split.

        Args:
            x_mean: Mean per node feature entry, shape ``(node_dim,)``.
            x_std: Standard deviation per node feature entry, shape ``(node_dim,)``.
            e_mean: Mean per edge feature entry, shape ``(edge_dim,)``.
            e_std: Standard deviation per edge feature entry, shape ``(edge_dim,)``.
        """
        self.x_mean.copy_(x_mean)
        self.x_std.copy_(x_std)
        self.e_mean.copy_(e_mean)
        self.e_std.copy_(e_std)

    def forward(self, x, edge_index, edge_attr):
        """Run one batch of configuration graphs through trunk, fork and heads.

        Args:
            x: Raw node features, shape ``(2 * B, node_dim)``. They are standardised here, so
                callers pass unnormalised values.
            edge_index: Edge list of the batched graph, shape ``(2, 2 * B)``.
            edge_attr: Raw edge features, shape ``(2 * B, edge_dim)``.

        Returns:
            A tuple ``(failure_logits, good_logit)`` for the ``B`` parts under test, both raw
            logits: ``(B, num_fail)`` in the order insufficient overlap, excessive thickness,
            tipping, and ``(B, 1)`` for the overall verdict.
        """
        # A node mixes quantities of very different magnitude: the size entries are lengths in
        # metres, the code entries are of order one. Presented raw, the shape information is
        # numerically swamped and the network effectively ignores it.
        x = (x - self.x_mean) / self.x_std
        edge_attr = (edge_attr - self.e_mean) / self.e_std

        x = F.gelu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.gelu(self.gat2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x_fail = F.gelu(self.gat3_fail(x, edge_index, edge_attr))
        x_good = F.gelu(self.gat3_good(x, edge_index, edge_attr))

        # Read out the part under test only, node 1 of every two-node graph.
        x_fail_b1 = x_fail[1::2]
        x_good_b1 = x_good[1::2]

        failure_logits = self.mlp_fail(x_fail_b1)
        good_logit = self.mlp_good(x_good_b1)

        return failure_logits, good_logit
