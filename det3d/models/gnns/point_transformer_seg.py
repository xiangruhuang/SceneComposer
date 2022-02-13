import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from .point_transformer import (
    TransformerBlock,
    TransitionDown
)
from torch_cluster import knn_graph

import torch_geometric.transforms as T
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.utils import intersection_and_union as i_and_u

from ..utils import build_norm_layer, build_mlp_layer
from ..registry import GNNS

class TransitionUp(torch.nn.Module):
    '''
        Reduce features dimensionnality and interpolate back to higher
        resolution and cardinality
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = build_mlp_layer([in_channels, out_channels])
        self.mlp = build_mlp_layer([out_channels, out_channels])

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
                                         batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x

@GNNS.register_module
class PointTransformerSeg(nn.Module):
    def __init__(self, in_channels, out_channels, dim_model,
                 up_transf_layers, down_transf_layers,
                 k=16, point_dim=3):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = build_mlp_layer(
                             [in_channels, dim_model[0]]
                         )

        #self.transformer_input = TransformerBlock(
        #                             in_channels=dim_model[0],
        #                             out_channels=dim_model[0],
        #                             point_dim=point_dim,
        #                         )

        # backbone layers
        self.transformers_up = nn.ModuleList()
        self.transformers_down = nn.ModuleList()
        self.transition_up = nn.ModuleList()
        self.transition_down = nn.ModuleList()
        self.up_transf_layers = up_transf_layers
        self.down_transf_layers = down_transf_layers
        for i in range(0, len(dim_model) - 1):
            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k)
            )

            if (i in down_transf_layers):
                self.transformers_down.append(
                    TransformerBlock(in_channels=dim_model[i + 1],
                                     out_channels=dim_model[i + 1],
                                     point_dim=point_dim)
                )
            else:
                self.transformers_down.append(
                    nn.Identity()
                )

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(in_channels=dim_model[i + 1],
                             out_channels=dim_model[i])
            )

            if i in up_transf_layers:
                self.transformers_up.append(
                    TransformerBlock(in_channels=dim_model[i],
                                     out_channels=dim_model[i],
                                     point_dim=point_dim)
                )
            else:
                self.transformers_up.append(
                    nn.Identity()
                )

        # summit layers
        self.mlp_summit = build_mlp_layer(
                              [dim_model[-1], dim_model[-1]],
                              batch_norm=False
                          )

        self.transformer_summit = TransformerBlock(
                                      in_channels=dim_model[-1],
                                      out_channels=dim_model[-1],
                                      point_dim=point_dim
                                  )

        # class score computation
        self.mlp_output = nn.Sequential(
            nn.Linear(dim_model[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, pos, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1)).to(pos.get_device())

        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)
        #edge_index = knn_graph(pos, k=self.k, batch=batch)
        #x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            if i in self.down_transf_layers:
                edge_index = knn_graph(pos, k=self.k, batch=batch)
                x = self.transformers_down[i](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

        # summit
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.transformers_down)
        for i in range(n):
            x = self.transition_up[-i - 1](
                    x=out_x[-i - 2],
                    x_sub=x,
                    pos=out_pos[-i - 2],
                    pos_sub=out_pos[-i - 1],
                    batch_sub=out_batch[-i - 1],
                    batch=out_batch[-i - 2]
                )

            if (len(self.transformers_up)-1-i) in self.up_transf_layers:
                edge_index = knn_graph(out_pos[-i - 2], k=self.k,
                                       batch=out_batch[-i - 2])
                x = self.transformers_up[-i - 1](x, out_pos[-i - 2], edge_index)

        out = self.mlp_output(x)

        return out
