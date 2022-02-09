import numpy as np

from torch import nn
from torch.nn import functional as F

from ..registry import BACKBONES
from ..utils import build_norm_layer, build_mlp_layer

@BACKBONES.register_module
class PointTransformer(nn.Module):
    """Feature extraction for a set of point clouds.
    Not fixing the number of points.
    
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        add_self_loops=True,
        **kwargs
    ):
        super(PointTransformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_nn = build_mlp_layer([out_channels, 64, out_channels])
        self.pos_nn = build_mlp_layer([3, 64, out_channels])
        self.lin_in = nn.Linear(in_channels, in_channels)
        self.lin_out = nn.Linear(out_channels, out_channels)

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_src = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, out_channels, bias=False)

        #self.reset_paramters()
        
    def forward(self, points):
        
        obj_features = self.mlp(points)
        
