"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl.nn import GATConv
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.layers.append(nn.Linear(n_hidden, n_classes))
    

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
                h = F.relu(h)
            h = layer(h)
        return h


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        if n_layers == 0:
            self.layers = nn.ModuleList()
            self.layers.append(GraphConv(in_feats, n_classes, allow_zero_in_degree=True))
        else:
            self.layers = nn.ModuleList()
            # input layer
            self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
            # output layer
            self.layers.append(GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        if num_layers == 0:
            self.gat_layers.append(GATConv(
                in_dim, num_classes, heads[0],
                feat_drop, attn_drop, negative_slope, False, None, allow_zero_in_degree=True))        
        else:
            # input projection (no residual)
            self.gat_layers.append(GATConv(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree=True))
            # hidden layers
            for l in range(1, num_layers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation, allow_zero_in_degree=True))
            # output projection
            self.gat_layers.append(GATConv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None, allow_zero_in_degree=True))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits
    

class DROPEDGE(nn.Module):
    def __init__(self,
                 args,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(DROPEDGE, self).__init__()
        from dgl import DropEdge
        self.g = g
        if n_layers == 0:
            self.layers = nn.ModuleList()
            self.layers.append(GraphConv(in_feats, n_classes, allow_zero_in_degree=True))
        else:
            self.layers = nn.ModuleList()
            # input layer
            self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
            # hidden layers
            for i in range(n_layers - 1):
                self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
            # output layer
            self.layers.append(GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)
        self.dropedge_fn = DropEdge(p=args.dropedge) # Probability of an edge to be dropped.

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
                
            if self.training:
                new_g = self.dropedge_fn(self.g.clone())
            else:
                new_g = self.g.clone()
            h = layer(new_g, h)
        return h