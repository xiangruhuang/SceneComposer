from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 
from det3d.core import Visualizer
from det3d.core.bbox import box_np_ops

@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        visualize=False,
        render=False
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        self.visualize = visualize
        self.render = render
        
    def extract_feat(self, data):
        if 'voxels' not in data:
            output = self.reader(data['points'])    
            voxels, coors, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels
            )
            input_features = voxels
        else:
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

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def render_examples(self, example):
        vis = Visualizer(
                  self.test_cfg.voxel_size,
                  self.test_cfg.pc_range,
                  self.test_cfg.out_size_factor
              )
        for i, points in enumerate(example['points']):
            points = points.detach().cpu()
            vis.clear()
            vis.pointcloud('points', points[:, :3])
            mask = example['mask'][0][i]
            gt_boxes_and_cls = example['gt_boxes_and_cls'][i, mask.bool()]
            gt_boxes = gt_boxes_and_cls[:, :-3].detach().cpu().numpy()
            cls = gt_boxes_and_cls[:, -1].detach().cpu().long()
            gt_corners = box_np_ops.center_to_corner_box3d(
                             gt_boxes[:, :3],
                             gt_boxes[:, 3:6],
                             gt_boxes[:, -1],
                             axis=2)
            vis.boxes('boxes', gt_corners, cls)
            dims = (points.max(0)[0][:2] - points.min(0)[0][:2])/4.0
            center = (points.max(0)[0][:3] + points.min(0)[0][:3]).detach().cpu()/2.0
            token = example['metadata'][i]['token'].split('.')[0]
            seq_id, frame_id = int(token.split('_')[1]), int(token.split('_')[3])
            #for dx in [0, 1]:
            #    for dy in [0, 1]:
            #        suffix = f'{dx}{dy}'
            #        center_ = center.clone()
            #        center_[:2] -= dims
            #        center_[0] += dims[0]*2*dx
            #        center_[1] += dims[1]*2*dy
            vis.look_at(center)
            vis.heatmap('hm', example['hm'][0][i, 0].detach().cpu())
            vis.heatmap('visibility', example['visibility'][i].detach().cpu().T)
            vis.heatmap('occupancy', example['occupancy'][i].detach().cpu().T)
            if self.visualize:
                vis.show()
                import ipdb; ipdb.set_trace()
            if self.render:
                vis.screenshot(f'figures/seq_{seq_id:03d}_frame_{frame_id:03d}.png')


    def forward(self, example, return_loss=True, **kwargs):
        x, _ = self.extract_feat(example)

        if self.render or self.visualize:
            self.render_examples(example)

        preds, _ = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None 
