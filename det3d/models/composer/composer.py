import torch.nn as nn
import torch

from .. import builder
from ..registry import COMPOSERS

class Discriminator(nn.Module):
    def __init__(
        self,
        backbone,
        heads,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(Discriminator, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        if heads['point'] is not None:
            self.point_head = builder.build_head(heads['point'])
        else:
            self.point_head = None
        self.box_head = builder.build_head(heads['box'])
        self.crit = nn.MSELoss()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self, data, test_cfg):

        gt_objects = data['objects']
        pred_objects = data['pred_objects']
        
        voxel_feat, gt_obj_feat, pred_obj_feat = self.backbone(
                                                     data,
                                                     gt_objects,
                                                     pred_objects,
                                                     test_cfg
                                                 )

        preds_on_gt = self.box_head(gt_obj_feat).squeeze(-1)
        preds_on_fake = self.box_head(pred_obj_feat).squeeze(-1)

        return torch.cat([preds_on_gt, preds_on_fake], dim=0)

    def loss(self, preds, data):
        
        gt_objects = data['objects']
        pred_objects = data['pred_objects']
        gt = torch.cat([gt_objects['gt'], pred_objects['gt']], dim=0)
        
        return self.crit(preds, gt)

class Generator(nn.Module):
    def __init__(
        self,
        backbone,
        heads,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(Generator, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        if heads['point'] is not None:
            self.point_head = builder.build_head(heads['point'])
        else:
            self.point_head = None
        self.box_head = builder.build_head(heads['box'])

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
    def forward(self, data, test_cfg, **kwargs):
        
        gt_objects = data['objects']
        voxel_feat, gt_obj_feat, _ = self.backbone(
                                         data,
                                         gt_objects,
                                         None,
                                         test_cfg
                                     )

        preds_dicts, _ = self.box_head(voxel_feat)

        # regress boxes
        box_preds = self.box_head.predict(data, preds_dicts, test_cfg)

        obj_preds = dict(
            boxes=box_preds['boxes'],
            labels=box_preds['labels'],
            points=None,
            coord=box_preds['coord'],
            gt=torch.zeros(box_preds['boxes'].shape[0],
                           dtype=torch.long).to(box_preds['boxes']),
        )
        
        # regress points
        if self.point_head is not None:
            assert False, "Not Implemented Yet"

        return obj_preds

@COMPOSERS.register_module
class Composer(nn.Module):
    def __init__(
        self,
        generator,
        discriminator,
        train_cfg=None,
        test_cfg=None,
    ):
        super(Composer, self).__init__()

        self.generator = Generator(
            **generator,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self.discriminator = Discriminator(
            **discriminator,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

        self.test_cfg = test_cfg
    
    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))

    def forward(self, data, return_loss=True, **kwargs):
        pred_objects = self.generator(data, self.test_cfg, **kwargs)

        data['pred_objects'] = pred_objects

        preds = self.discriminator(data, self.test_cfg, **kwargs)
        loss = self.discriminator.loss(preds, data)

        return dict(loss=[loss])
