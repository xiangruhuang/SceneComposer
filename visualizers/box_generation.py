import argparse
import numpy as np
from det3d.core import Visualizer
import pickle
import os, glob
from det3d.structures import Frame 
from det3d.core.bbox import box_np_ops
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        """Visualize Generated Boxes Results.

        example usage:

            python visualizers/box_generation.py work_dirs/waymo_6e_imaginary_box_generator/prediction.pkl --split train_50
           
        """
        )
    parser.add_argument('pred', help='generation results file (.pkl)')
    parser.add_argument('--split', default='train_50', type=str)
    args = parser.parse_args()
    
    return args

def parse_boxes(boxes):
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()
    if boxes.shape[-1] == 8:
        sint = boxes[:, -2:-1]
        cost = boxes[:, -1:]
        theta = torch.atan2(sint, cost).numpy()[:, 0]
    else:
        theta = -boxes[:, -1]

    corners = box_np_ops.center_to_corner_box3d(
                  boxes[:, :3],
                  boxes[:, 3:6],
                  theta,
                  axis=2
              )

    return corners

if __name__ == '__main__':
    args = parse_args()

    vis = Visualizer()
    with open(args.pred, 'rb') as fin:
        predictions = pickle.load(fin)
        for token, val in predictions.items():
            seq_id = token.split('.')[0].split('_')[1]
            frame_id = token.split('.')[0].split('_')[3]
            frame = Frame.from_index(seq_id, frame_id, args.split)
            boxes = val['box3d_lidar']
            labels = val['label_preds']

            gt_boxes = frame.boxes
            gt_labels = frame.classes
            gt_corners = parse_boxes(gt_boxes)
            
            pred_boxes = boxes
            pred_labels = labels.long()

            pred_corners = parse_boxes(pred_boxes)

            vis.boxes('gt-box', gt_corners, gt_labels, color=(0,1,0))
            vis.boxes('pred-box', pred_corners, pred_labels, color=(1,0,0))

            surfaces = box_np_ops.corner_to_surfaces_3d(gt_corners)
            indices = box_np_ops.points_in_convex_polygon_3d_jit(frame.points, surfaces)
            points = frame.points[indices.any(-1) == False]

            vis.pointcloud('points', points, color=(75, 75, 75))

            vis.show()

