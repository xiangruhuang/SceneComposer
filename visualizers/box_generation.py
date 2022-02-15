import numpy as np
import argparse
from det3d.core import Visualizer
import pickle
import os, glob
from det3d.structures import Frame 
from det3d.core.bbox import box_np_ops
import torch


def parse_args():
    parser = argparse.ArgumentParser('Visualize Generated Boxes Results')
    parser.add_argument('pred', help='generation results file (.pkl)')
    parser.add_argument('--split', default='train_50', type=str)
    args = parser.parse_args()
    
    return args

def parse_boxes(boxes):
    sint = boxes[:, -2:-1]
    cost = boxes[:, -1:]
    theta = torch.atan2(sint, cost).numpy()[:, 0]
    boxes = boxes.numpy()

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
            boxes = val['boxes']
            labels = val['labels']
            gt = val['gt'].long()
            gt_boxes = boxes[gt == 1]
            gt_labels = labels[gt == 1].long()
            gt_corners = parse_boxes(gt_boxes)
            
            pred_boxes = boxes[gt == 0]
            pred_labels = labels[gt == 0].long()
            pred_corners = parse_boxes(pred_boxes)

            vis.boxes('gt-box', gt_corners, gt_labels)
            vis.boxes('pred-box', pred_corners, pred_labels)

            vis.pointcloud('points', frame.points)
            import ipdb; ipdb.set_trace()

            vis.show()

