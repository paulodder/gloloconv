import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, Dropout, Linear


class GloLoConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads,
        edge_dim,
        graph_dropout,
        linear_dropout,
        num_nodes,
    ):
        super(GloLoConv, self).__init__()

        self.gatv2 = GATv2Conv(  # graph part
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            edge_dim=edge_dim,
            aggr="mean",
            dropout=graph_dropout,
        )
        self.dropout = (
            Dropout(p=linear_dropout) if linear_dropout > 0 else None,
        )
        self.lin = Linear(
            in_features=(num_nodes * out_channels * heads,),
            out_features=(num_nodes * out_channels,),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gatv2(x, edge_index, edge_attr)
        if self.dropout is not None:
            x = self.dropout(x)
        x = flatten(x, batch)
        x = self.lin(x)
        x = matricize(x, batch)
        return x


def flatten(x, batch):
    """
    Flattens the data to arrive at a batch_size x num_features tensor.  x is of
    shape (batch_size x num_nodes) x num_features and batch is of shape
    (batch_size x num_nodes).
    """
    if batch is None:
        return x.view(1, -1)
    batch_size = batch.max().item() + 1
    x = x.view(batch_size, -1)
    return x


def matricize(x, batch):
    """
    Matricizes the data to arrive at a batch_size x num_nodes x num_features
    tensor. x is of shape batch_size x num_features and batch is of shape
    batch_size.
    """
    num_nodes = (batch == batch.min()).sum()
    num_features = x.size(1) // num_nodes
    x = x.view(-1, num_features)
    return x
