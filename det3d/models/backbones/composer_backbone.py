import numpy as np

from torch import nn
import torch
from torch.nn import functional as F

from .. import builder
from ..registry import BACKBONES
from ..utils import build_norm_layer, build_mlp_layer

class BackgroundFeatureModule(nn.Module):
    def __init__(
        self,
        reader,
        backbone,
        neck,
    ):
        super(BackgroundFeatureModule, self).__init__()
        self.reader = builder.build_reader(reader)
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)

    def forward(self, data):
        
        data = dict(
            features=data['voxels'],
            num_voxels=data["num_points"],
            coors=data["coordinates"],
            batch_size=len(data['points']),
            input_shape=data["shape"][0],
        )
        input_features = self.reader(data["features"], data['num_voxels'])

        x, voxel_feature = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        x = self.neck(x)

        # transpose to [batch, x, y, feat_dim]
        x = x.transpose(1, 3)
        
        # find 2D coordinates (ignoring z dimension) of grids 
        x_dim, y_dim = torch.tensor(data['input_shape'], dtype=torch.long)[:2]
        assert x_dim * x.shape[2] == y_dim * x.shape[1]
        factor = torch.div(x_dim, x.shape[1], rounding_mode='floor')
        batch, x_idx, y_idx = torch.meshgrid(
                                  torch.arange(x.shape[0]),
                                  torch.arange(x.shape[1]),
                                  torch.arange(x.shape[2]),
                              )
        
        coord = torch.stack([x_idx, y_idx], dim=-1).view(-1, 2) * factor
        batch = batch.reshape(-1)

        return x, batch.to(x.device), coord.to(x.device)

class ObjectFeatureModule(nn.Module):
    """ Shared feature extraction.

    """
    def __init__(
        self,
        point_gnn,
        box_mlp,
        train_cfg=None,
    ):
        super(ObjectFeatureModule, self).__init__()
        if point_gnn is not None:
            self.point_gnn = builder.build_gnn(point_gnn)
        else:
            self.point_gnn = None

        self.box_mlp = build_mlp_layer(box_mlp['channels'])
        self.num_classes = box_mlp['num_classes']
    
    def forward(self, objects):

        boxes = objects['boxes']
        num_objects = boxes.shape[0] 
        labels = objects['labels']
        labels_one_hot = F.one_hot(labels, self.num_classes)
        boxes = torch.cat([boxes[:, :6], boxes[:, -1:]], dim=-1)
        box_attr = torch.cat([boxes, labels_one_hot], dim=-1)
        obj_box_feature = self.box_mlp(box_attr)

        if (objects['points'] is not None) and (self.point_gnn is not None):
            points = objects['points']
            pos = points[:, :3]
            x = points[:, 3:]
            batch = objects['batch'].long()
            obj_point_feature = self.point_gnn(x, pos, batch)
        else:
            obj_point_feature = torch.empty(num_objects, 0).to(obj_box_feature)

        obj_feature = torch.cat([obj_box_feature, obj_point_feature], dim=-1)

        obj_batch = objects['coord'][:, -1].long()
        obj_coord = objects['coord'][:, :2]

        return obj_feature, obj_batch, obj_coord

@BACKBONES.register_module
class ComposerBackbone(nn.Module):
    def __init__(
        self,
        bg_feat_module,
        obj_feat_module,
        feat_prop_module,
        **kwargs
    ):
        super(ComposerBackbone, self).__init__()

        self.bg_feat_module = BackgroundFeatureModule(**bg_feat_module)

        self.obj_feat_module = ObjectFeatureModule(**obj_feat_module)

        self.feat_prop_module = builder.build_gnn(feat_prop_module)

    def forward(self, data, gt_objects, pred_objects=None, test_cfg=None):
        
        x1, batch1, coord1 = self.bg_feat_module(data)
        x1_shape = x1.shape
        x1 = x1.reshape(-1, x1.shape[-1])

        x2, batch2, coord2 = self.obj_feat_module(gt_objects)
        if pred_objects is not None:
            x3, batch3, coord3 = self.obj_feat_module(pred_objects)
            num_preds = x3.shape[0]
            
            x2 = torch.cat([x2, x3], dim=0)
            batch2 = torch.cat([batch2, batch3], dim=0)
            coord2 = torch.cat([coord2, coord3], dim=0)
        else:
            num_preds = 0
        
        with torch.no_grad():
            e1, e2 = torch.meshgrid(
                        torch.arange(x1.shape[0]),
                        torch.arange(x2.shape[0]),
                     )
            e1, e2 = e1.reshape(-1), e2.reshape(-1)
            edge_index = torch.stack([e1, e2], dim=0)
            edge_index = edge_index[:, batch1[e1] == batch2[e2]].to(x1.device)
            edge_index[1, :] += x1.shape[0]

        x = torch.cat([x1, x2], dim=0)
        batch = torch.cat([batch1, batch2], dim=0)
        coord = torch.cat([coord1, coord2], dim=0).float()
        x_out = self.feat_prop_module(x, coord, batch)
        x1_out = x_out[:x1.shape[0]]
        x1_out = x1_out.reshape(*x1_shape[:3], -1)
        x1_out = x1_out.transpose(1, 3)

        x2_out = x_out[x1.shape[0]:(x.shape[0]-num_preds)]
        x3_out = x_out[(x.shape[0]-num_preds):]

        return x1_out, x2_out, x3_out
