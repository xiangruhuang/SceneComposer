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

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self, data, objects, test_cfg):
        
        voxel_feat, obj_feat = self.backbone(
                                   data,
                                   objects,
                                   test_cfg
                               )

        return self.box_head(obj_feat).squeeze(-1)

    def loss(self, preds, gt):

        gt = gt.long()
        loss = -(preds[gt == 1].log().mean() + (1-preds[gt == 0]).log().mean())
        acc = (preds.round().long() == gt).float().mean()

        return dict(
            loss=loss,
            dsc_acc=[acc],
        )

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
        
    def forward(self, data, gt_objects, test_cfg, **kwargs):
        
        voxel_feat, obj_feat = self.backbone(
                                   data,
                                   gt_objects,
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

    def loss(self, scores):

        acc = scores.round().mean()

        return dict(
            loss=(1 - scores).log().mean(),
            gen_acc=[acc],
        )

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
        gt_objects = data.pop('objects')

        pred_objects = self.generator(data, gt_objects, self.test_cfg, **kwargs)

        all_objects = {}
        for attr in ['boxes', 'coord', 'labels', 'gt']:
            all_objects[attr] = torch.cat([gt_objects[attr], pred_objects[attr]], dim=0)
        all_objects['points'] = None
        all_objects['batch'] = None

        preds = self.discriminator(data, all_objects, self.test_cfg, **kwargs)
        
        preds_on_fake = preds[all_objects['gt'].long()==1]

        loss_gen = self.generator.loss(preds_on_fake)
        loss_dsc = self.discriminator.loss(preds, all_objects['gt'])

        rets = dict(loss=[loss_gen.pop('loss')+loss_dsc.pop('loss')])
        rets.update(loss_gen)
        rets.update(loss_dsc)

        return rets
