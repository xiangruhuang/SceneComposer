import torch 
from PIL import Image
from collections import defaultdict
from copy import deepcopy 
import os
import numpy as np

from det3d.core import Visualizer
from det3d.core.bbox import box_np_ops
from det3d.torchie.trainer import load_checkpoint

from ..registry import DETECTORS
from .single_stage import SingleStageDetector

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
        self.epoch_dict = defaultdict(lambda: 0)
        self.visualize = visualize
        self.render = render
        
    def extract_feat(self, example):
        if 'voxels' not in example:
            output = self.reader(example['points'])
            voxels, coors, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(example['points']),
                input_shape=shape,
                voxels=voxels
            )
            input_features = voxels
        else:
            data = dict(
                features=example['voxels'],
                num_voxels=example["num_points"],
                coors=example["coordinates"],
                batch_size=len(example['points']),
                input_shape=example["shape"][0],
            )
            input_features = self.reader(data["features"], data['num_voxels'])

        x, voxel_feature = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        #visibility = example['visibility'].unsqueeze(1).transpose(-1, -2).to(x.device)
        #occupancy = example['occupancy'].unsqueeze(1).transpose(-1, -2).to(x.device)
        #x = torch.cat([x, visibility, occupancy], dim=1)

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def render_examples(self, example, preds):
        vis = Visualizer(
                  self.test_cfg.voxel_size,
                  self.test_cfg.pc_range,
                  self.test_cfg.out_size_factor
              )
        num_samples = len(example['points'])

        for i in range(num_samples):
            token = example['metadata'][i]['token'].split('.')[0]
            seq_id, frame_id = int(token.split('_')[1]), int(token.split('_')[3])

            visit_time = self.epoch_dict[token]
            self.epoch_dict[token] += 1

            points = example['points'][i].detach().cpu()

            vis.clear()
            #vis.pointcloud('points', points[:, :3])

            # get boxes
            if 'mask' in example:
                mask = example['mask'][0][i]
                gt_boxes_and_cls = example['gt_boxes_and_cls'][i, mask.bool()]
                gt_boxes = gt_boxes_and_cls[:, :-3].detach().cpu().numpy()
                cls = gt_boxes_and_cls[:, -1].detach().cpu().long()
            else:
                batch = example['gt_boxes_and_cls_batch']
                mask = (batch == i)
                gt_boxes_and_cls = example['gt_boxes_and_cls'][mask]
                gt_boxes = gt_boxes_and_cls[:, :-3].detach().cpu().numpy()
                cls = gt_boxes_and_cls[:, -1].detach().cpu().long()

            # draw boxes
            gt_corners = box_np_ops.center_to_corner_box3d(
                             gt_boxes[:, :3],
                             gt_boxes[:, 3:6],
                             gt_boxes[:, -1],
                             axis=2)
            #vis.boxes('boxes', gt_corners, cls)

            # compute visualizer setting
            camera_center = (points.max(0)[0][:3] + points.min(0)[0][:3]).detach().cpu()/2.0
            token = example['metadata'][i]['token'].split('.')[0]
            seq_id, frame_id = int(token.split('_')[1]), int(token.split('_')[3])
            vis.look_at(camera_center, distance=200)

            if self.visualize:
                vis.pointcloud('points', example['points'][0][:, :3].detach().cpu())
                vis.boxes_from_attr('boxes', gt_boxes, cls-1)
                import ipdb; ipdb.set_trace()
                vis.show()
            # draw heat map
            vis.heatmap('hm', example['hm'][0][i, 0].detach().cpu())
            if self.render:
                folder = f'figures/heatmap/seq_{seq_id:03d}_frame_{frame_id:03d}'
                os.makedirs(folder, exist_ok=True)
                path = f'{folder}/{visit_time:05d}.png'
                vis.screenshot(path)
                img1 = np.array(Image.open(path))
                pred_hm = preds[0]['hm'][i, 0].detach().cpu()
                pred_hm = self.bbox_head._sigmoid(pred_hm)
                vis.heatmap('hm', pred_hm)
                vis.screenshot(path)
                img2 = np.array(Image.open(path))
                img = np.concatenate([img1, img2], axis=0)
                Image.fromarray(img).save(path)

    def forward(self, example, return_loss=True, **kwargs):
        x, _ = self.extract_feat(example)

        preds, _ = self.bbox_head(x)
            
        local_rank = 0
        if "LOCAL_RANK" in os.environ:
            # distributed 
            local_rank = int(os.environ["LOCAL_RANK"])
        
        if local_rank == 0:
            if self.render or self.visualize:
                self.render_examples(example, preds)

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
