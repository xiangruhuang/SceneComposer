import numpy as np
try:
    import spconv.pytorch as spconv 
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv3d, SubMConv3d
except: 
    import spconv 
    from spconv import ops
    from spconv import SparseConv3d, SubMConv3d

from torch import nn
from torch.nn import functional as F

from ..registry import BACKBONES
from ..utils import build_norm_layer

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out
@BACKBONES.register_module
class TestBackbone(nn.Module):
    def __init__(
        self,
        num_input_features=128,
        channels=[16, 32, 64, 128],
        norm_cfg=None,
        name="TestBackbone",
        **kwargs
    ):
        super(TestBackbone, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        c0, c1, c2, c3 = channels

        # input: # [1600, 1200, 41]
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, c0, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, c0)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(c0, c0, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(c0, c0, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(
                c0, c1, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, c1)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(c1, c1, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(c1, c1, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(
                c1, c2, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, c2)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(c2, c2, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(c2, c2, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(
                c2, c3, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, c3)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(c3, c3, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(c3, c3, norm_cfg=norm_cfg, indice_key="res3"),
        )
        
        #self.conv5 = spconv.SparseSequential(
        #    SparseConv3d(
        #        128, 128, 3, (1, 2, 2), padding=[1, 1, 1], bias=False
        #    ),  # [188, 188, 5] -> [93, 93, 5]
        #    build_norm_layer(norm_cfg, 128)[1],
        #    nn.ReLU(inplace=True),
        #    SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
        #    SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
        #)

        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                c3, c3, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, c3)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):

        # input: # [41, 1600, 1408]
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        ret = self.extra_conv(x_conv4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret, multi_scale_voxel_features
