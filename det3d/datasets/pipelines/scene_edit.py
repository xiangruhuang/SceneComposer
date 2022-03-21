import numpy as np
import torch
import pickle

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

def get_obj(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)

class Projector(object):
    def __init__(self, voxel_size, pc_range):
        self.voxel_size = voxel_size
        self.pc_range = pc_range

    def __call__(self, points):
        return np.floor(
                   np.divide(
                       points[:, :2] - self.pc_range[:2],
                       self.voxel_size[:2]
                   )
               ).astype(np.int32)

@PIPELINES.register_module
class ComputeVisibility(object):
    """For all grid, compute if they are visible by the camera.

    """
    def __init__(self, cfg, **kwargs):
        self.voxel_size = np.array(cfg.voxel_size)
        self.pc_range = np.array(cfg.pc_range)
        self.size_factor = cfg.out_size_factor
        self.projector = Projector(self.voxel_size*self.size_factor, self.pc_range)

    def __call__(self, res, info):
        points = res['lidar']['points']
        points_xy = points[:, :2]
        grid_size = self.projector(self.pc_range[np.newaxis, -3:-1])[0]
        visibility = np.zeros(grid_size, dtype=np.float32)
        for ratio in np.linspace(0, 1, 100):
            x, y = self.projector(points_xy*ratio).T
            x = x.clip(0, grid_size[0]-1)
            y = y.clip(0, grid_size[1]-1)
            visibility[(x, y)] += 0.2

        res['lidar']['visibility'] = visibility.clip(0, 1)

        return res, info


@PIPELINES.register_module
class ComputeGroundPlaneMask(object):
    def __init__(self, threshold, **kwargs):
        self.threshold = threshold 
    
    def __call__(self, res, info):
        
        # transform points into world coordinate system
        annos = get_obj(info['anno_path'])
        T = annos['veh_to_global'].reshape(4, 4)
        points_xyz = res['lidar']['points'][:, :3]
        points_xyz = points_xyz @ T[:3, :3].T + T[:3, 3]

        # load ground plane (in world coord. system)
        tokens = info['anno_path'].split('/')
        seq_id = int(tokens[-1].split('_')[1])
        tokens[-1] = f'seq_{seq_id}.pkl'
        gp_path = '/'.join(tokens).replace('annos', 'ground_plane')
        gp_dict = get_obj(gp_path)
        ground_plane = gp_dict["ground_plane"]
        projector = Projector(gp_dict["voxel_size"]*gp_dict["size_factor"],
                              gp_dict["pc_range"])

        # find relative height of each point
        vx, vy = projector(points_xyz[:, :2]).T
        ground_z = ground_plane[(vx, vy, 2)]
        height = points_xyz[:, 2] - ground_z

        is_ground = (height < self.threshold)[:, np.newaxis].astype(np.float32)
        res["lidar"]["points"] = np.concatenate(
                                     [res["lidar"]["points"], is_ground],
                                     axis=-1)

        return res, info


@PIPELINES.register_module
class ComputeOccupancy(object):
    """For all grid, compute if they are occupied by the background.

    """
    def __init__(self, cfg, **kwargs):
        self.voxel_size = np.array(cfg.voxel_size)
        self.pc_range = np.array(cfg.pc_range)
        self.size_factor = cfg.out_size_factor
        self.projector = Projector(self.voxel_size*self.size_factor, self.pc_range)

    def __call__(self, res, info):
        points = res['lidar']['points']
        points_xy = points[:, :2]
        grid_size = self.projector(self.pc_range[np.newaxis, -3:-1])[0]
        occupancy = np.zeros(grid_size, dtype=np.float32)
        is_ground = points[:, -1].astype(np.int32)
        
        x, y = self.projector(points_xy[is_ground == False]).T
        x = x.clip(0, grid_size[0]-1)
        y = y.clip(0, grid_size[1]-1)
        occupancy[(x, y)] = 1
        visibility = res["lidar"]["visibility"]
        res["lidar"]["points"] = res["lidar"]["points"][:, :-1]

        res["lidar"]["occupancy"] = occupancy

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
        gt_dict = res["lidar"]["box_annotations"]
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


