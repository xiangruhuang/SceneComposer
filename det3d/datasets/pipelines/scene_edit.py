import numpy as np
import torch

from det3d.core.bbox import box_np_ops
from det3d.core.bbox.geometry import points_in_convex_polygon_3d_jit
from det3d.builder import build_dbsampler

from ..registry import PIPELINES

def remove_empty_boxes(indices, attr_dict):
    not_empty_box = indices.any(0)

    new_attr_dict = {}
    for key, val in attr_dict.items():
        new_attr_dict[key] = val[not_empty_box]
    indices = indices[:, not_empty_box]

    return indices, new_attr_dict

@PIPELINES.register_module
class ComputeVisibility(object):
    """For all grid, compute if they are visible by the camera.

    """
    def __init__(self, cfg, **kwargs):
        self.voxel_size = np.array(cfg.voxel_size)
        self.pc_range = np.array(cfg.pc_range)
        self.size_factor = cfg.out_size_factor

    def __call__(self, res, info):
        points = res['lidar']['points']
        points_xy = points[:, :2]
        grid_size = np.round(
                        np.divide(
                            self.pc_range[-3:-1] - self.pc_range[:2],
                            self.voxel_size[:2]*self.size_factor)
                        ).astype(np.int32)
        visible = np.zeros(grid_size[::-1], dtype=np.float32)
        for ratio in np.linspace(0, 1, 100):
            x, y = np.round(
                       np.divide(
                           points_xy[:, :2]*ratio - self.pc_range[:2],
                           self.voxel_size[:2]*self.size_factor)
                       ).astype(np.int32).T
            visible[(y, x)] += 0.2

        visible = visible.clip(0, 1)
        from det3d.core import Visualizer
        vis = Visualizer(self.voxel_size, self.pc_range, self.size_factor)
        vis.heatmap('visible', visible)
        vis.pointcloud('points', points[:, :3])
        vis.show()

        return res, info

@PIPELINES.register_module
class ComputeOccupancy(object):
    """For all grid, compute if they are visible by the camera.

    """
    def __init__(self, cfg, **kwargs):
        self.voxel_size = np.array(cfg.voxel_size)
        self.pc_range = np.array(cfg.pc_range)
        self.size_factor = cfg.out_size_factor

    def __call__(self, res, info):
        points = res['lidar']['points']
        points_xy = points[:, :2]
        grid_size = np.round(
                        np.divide(
                            self.pc_range[-3:-1] - self.pc_range[:2],
                            self.voxel_size[:2]*self.size_factor)
                        ).astype(np.int32)
        occupancy = np.zeros(grid_size[::-1], dtype=np.float32)
        x, y = np.round(
                   np.divide(
                       points_xy[:, :2] - self.pc_range[:2],
                       self.voxel_size[:2]*self.size_factor)
                   ).astype(np.int32).T
        occupancy[(y, x)] = 1 

        from det3d.core import Visualizer
        vis = Visualizer(self.voxel_size, self.pc_range, self.size_factor)
        vis.heatmap('occupancy', occupancy)
        vis.pointcloud('points', points[:, :3])
        vis.show()

        return res, info

@PIPELINES.register_module
class SeparateForeground(object):
    """Remove all points in the boxes.
    
    """
    def __init__(self, cfg, **kwargs):
        self.mode = cfg.mode
        self.return_objects = cfg.get("return_objects", False)
        self.ignore_empty_boxes = cfg.get("ignore_empty_boxes", False)

    def __call__(self, res, info):

        assert self.mode == "train"
        
        points = res["lidar"]["points"]
        gt_dict = res["lidar"]["annotations"]
        gt_boxes = gt_dict["gt_boxes"]

        if gt_boxes.shape[0] > 0:
            # find points that are in the boxes
            gt_corners = box_np_ops.center_to_corner_box3d(
                             gt_boxes[:, :3],
                             gt_boxes[:, 3:6],
                             gt_boxes[:, -1],
                             axis=2
                         )
            surfaces = box_np_ops.corner_to_surfaces_3d(gt_corners)
            indices = points_in_convex_polygon_3d_jit(
                          points[:, :3], surfaces
                      ) # ([num_points, num_boxes], bool)

            if self.return_objects:
                # extract objects, boxes and class labels
                gt_classes = gt_dict["gt_classes"]
                    
                if self.ignore_empty_boxes:
                    indices, gt_dict = remove_empty_boxes(indices, gt_dict)
                    gt_boxes = gt_dict["gt_boxes"]
                    gt_classes = gt_dict["gt_classes"]
                    

                # extract objects and boxes
                is_in_box = indices.any(-1)
                obj_points = points[is_in_box]
                batch = indices[is_in_box].astype(
                            np.int32).argmax(-1).astype(np.int32)

                anno_boxes = np.concatenate([gt_boxes[:, :6],
                                             np.sin(gt_boxes[:, -1:]),
                                             np.cos(gt_boxes[:, -1:])],
                                            axis=1)
                objects = dict(
                    points=obj_points,
                    batch=batch,
                    boxes=anno_boxes,
                    classes=gt_classes,
                )
                res["lidar"]["objects"] = objects
            
            # get background points
            points = points[indices.any(-1) == False]

            res["lidar"]["points"] = points

        return res, info


