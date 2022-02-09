import warnings

import torch.nn as nn
from det3d.torchie.cnn import constant_init, kaiming_init

class MLP(nn.Module):
    def __init__(self, channels, batch_norm=True):
        super(MLP, self).__init__()
        layers = []
        for i in range(1, len(channels)):
            linear = nn.Linear(channels[i-1], channels[i])
            if batch_norm:
                bn = nn.BatchNorm1d(channels[i])
            else:
                bn = nn.Identity()
            act = nn.ReLU()
            layers.append(nn.Sequential(linear, bn, act))
        self.mlp = nn.Sequential(*layers)
          
    def forward(self, x):
        return self.mlp(x)

    def init_weights(self):
        pass


def build_mlp_layer(channels, cfg=dict(), **kwargs):
    """ build MLP layer

    Returns:
        mlp (nn.Module): created MLP layer
    """
    batch_norm = cfg.get('batch_norm', True)

    mlp = MLP(channels, batch_norm)

    return mlp
