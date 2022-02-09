import torch.nn as nn

from .. import builder
from ..registry import COMPOSERS

class Generator(nn.Module):
    def __init__(
        self,
        reader,
        backbone,
        obj_backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(Generator, self).__init__()
        self.reader = builder.build_reader(reader)
        self.backbone = builder.build_backbone(backbone)
        self.obj_backbone = builder.build_backbone(obj_backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_background_feat(self, data):
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

        return x, voxel_feature

    def extract_object_feat(self, data):
        import ipdb; ipdb.set_trace()

        objects = data['objects']
        for class_name in objects.keys():
            objects_cls = objects[class_name]

        self.obj_backbone(
                data['objects'],

            )
        
    def forward(self, example, **kwargs):

        x1, _ = self.extract_background_feat(example)
        x2 = self.extract_object_feat(example)
        import ipdb; ipdb.set_trace()

        preds, _ = self.bbox_head(x)

        pass
    
    def forward_train(self):
        pass

@COMPOSERS.register_module
class Composer(nn.Module):
    def __init__(
        self,
        reader,
        backbone,
        obj_backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(Composer, self).__init__()
        self.generator = Generator(
            reader,
            backbone,
            obj_backbone,
            neck,
            bbox_head,
            train_cfg,
            test_cfg,
            pretrained
        )

        self.init_weights(pretrained=pretrained)
    
    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))

    def loss(self):
        pass

    def forward(self, example, return_loss=True, **kwargs):
        preds = self.generator(example, **kwargs)
