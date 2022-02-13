import numpy as np

from torch import nn
from torch.nn import functional as F

from torch_geometric.nn.conv import PointTransformerConv
from torch_cluster import fps, knn_graph
from torch_scatter import scatter_max
from torch_geometric.nn.pool import knn
from torch_geometric.nn import global_mean_pool

from ..registry import GNNS
from ..utils import build_norm_layer, build_mlp_layer

@GNNS.register_module
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, point_dim=3):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, in_channels)
        self.lin_out = nn.Linear(out_channels, out_channels)

        self.pos_nn = build_mlp_layer([point_dim, 64, out_channels], batch_norm=False)

        self.attn_nn = build_mlp_layer([out_channels, 64, out_channels],
                                       batch_norm=False)

        self.transformer = PointTransformerConv(
                               in_channels,
                               out_channels,
                               pos_nn=self.pos_nn,
                               attn_nn=self.attn_nn
                           )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = build_mlp_layer([in_channels, out_channels])

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)
        
        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(x[id_k_neighbor[1]], id_k_neighbor[0],
        dim_size=id_clusters.size(0), dim=0)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


@GNNS.register_module
class PointTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = build_mlp_layer([in_channels, dim_model[0]])

        self.transformer_input = TransformerBlock(in_channels=dim_model[0],
                                                  out_channels=dim_model[0])
        # backbone layers
        self.transformers_down = nn.ModuleList()
        self.transition_down = nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                out_channels=dim_model[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                out_channels=dim_model[i + 1]))

        # class score computation
        self.mlp_output = nn.Sequential(
                              nn.Linear(dim_model[-1], 64),
                              nn.ReLU(),
                              nn.Linear(64, 64),
                              nn.ReLU(),
                              nn.Linear(64, out_channels)
                          )

    def forward(self, x, pos, batch=None):
        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())
                                                                                                                                                                                                                   # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)
                                                                                                                                                                                                                   # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)
        
        # GlobalAveragePooling
        x = global_mean_pool(x, batch)
        
        out = self.mlp_output(x)

        return out

