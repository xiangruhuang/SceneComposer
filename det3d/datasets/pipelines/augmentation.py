import numpy as np
import pickle

from det3d.core.bbox import box_np_ops
from det3d.core.bbox.geometry import (
    points_in_convex_polygon_3d_jit
)
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


@PIPELINES.register_module
class ReplaceAug(object):
    """Replace Augmentation.

    For each ground truth box (object), with probability `replace_prob`,
    replace the box with a random one of the same class.
    Properly orient the box.

    """
    def __init__(self, cfg=None, **kwargs):
        self.mode = cfg.mode
        if self.mode == "train":
            self.class_names = cfg.class_names
        dbinfo_path = cfg.get('dbinfo_path', None)
        if dbinfo_path is not None:
            with open(dbinfo_path, 'rb') as fin:
                self._dbinfos = pickle.load(fin)

        self.replace_prob = cfg.get('replace_prob', 0.5)
        self.keep_orientation = cfg.get('keep_orientation', True)

    def __call__(self, res, info):
        
        points = res["lidar"]["points"]
        gt_dict = res["lidar"]["annotations"]

        gt_boxes = gt_dict['gt_boxes']
        gt_names = gt_dict['gt_names']
        num_boxes = gt_boxes.shape[0]
        removed_box_indices = []
        new_box_infos = []
        
        # decide the boxes to be removed and the new box attributes
        for class_name in self.class_names:
            indices = np.array([i for i, gt_name in enumerate(gt_names)
                                if gt_name == class_name])
            num_boxes = indices.shape[0]
            if num_boxes == 0:
                continue
            boxes = gt_boxes[indices]
            names = gt_names[indices]
            
            mask = np.random.uniform(size=(num_boxes)) < self.replace_prob
            if mask.astype(np.int32).sum() == 0:
                continue

            replaced_boxes = boxes[mask]
            num_replaces = replaced_boxes.shape[0]
            
            class_infos = self._dbinfos[class_name]
            rand_indices = np.random.permutation(len(class_infos))[:num_replaces]
            new_box_infos += [class_infos[i] for i in rand_indices]
            removed_box_indices.append(indices[mask])
        
        if len(removed_box_indices) > 0:
            removed_box_indices = np.concatenate(removed_box_indices, axis=0)
            removed_boxes = gt_boxes[removed_box_indices]

            new_boxes = np.stack(
                            [info['box3d_lidar'] for info in new_box_infos]
                        )
            new_boxes[:, :3] = removed_boxes[:, :3]

            new_points = []
            for i, info in enumerate(new_box_infos):
                path = 'data/Waymo/'+info['path']
                cluster = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
                box_i = removed_boxes[i:i+1]
                new_box_i = new_boxes[i:i+1]

                if self.keep_orientation:
                    # rotate the points to follow
                    # the orientation of the old object
                    theta = new_box_i[0, -1] - box_i[0, -1]
                    R = np.array([np.cos(theta), -np.sin(theta),
                                  np.sin(theta),  np.cos(theta)]).reshape(2, 2)
                    cluster[:, :2] = cluster[:, :2] @ R.T
                cluster[:, :3] += new_boxes[i, :3]

                new_points.append(cluster)

            if self.keep_orientation:
                # rotate the new boxs to follow
                # the orientation of the old ones
                new_boxes[:, -1] = removed_boxes[:, -1]
            new_points = np.concatenate(new_points, axis=0)
            involved_boxes = np.concatenate(
                                 [new_boxes, removed_boxes], axis=0,
                             )
            
            corners = box_np_ops.center_to_corner_box3d(
                involved_boxes[:, :3],
                involved_boxes[:, 3:6],
                involved_boxes[:, -1],
                axis=2,
            )
            surfaces = box_np_ops.corner_to_surfaces_3d(corners)
            indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
            valid_mask = indices.any(-1) == False
            points = points[valid_mask]

            points = np.concatenate([points, new_points], axis=0)
            gt_boxes[removed_box_indices] = new_boxes
            gt_dict["gt_boxes"] = gt_boxes

        res["lidar"]["points"] = points
        res["lidar"]["annotations"] = gt_dict

        return res, info


@PIPELINES.register_module
class GTAug(object):
    def __init__(self, cfg=None, **kwargs):
        self.mode = cfg.mode
        if self.mode == "train":
            self.class_names = cfg.class_names
            if cfg.db_sampler is not None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None

    def __call__(self, res, info):

        assert self.mode == "train", "For training only."

        points = res["lidar"]["points"]
        gt_dict = res["lidar"]["annotations"]
            
        if self.db_sampler:
            sampled_dict = self.db_sampler.sample_all(
                res["metadata"]["image_prefix"],
                gt_dict["gt_boxes"],
                gt_dict["gt_names"],
                res["metadata"]["num_point_features"],
                False,
                gt_group_ids=None,
                calib=None,
                road_planes=None
            )

            if sampled_dict is not None:
                sampled_gt_masks = sampled_dict["gt_masks"]
                sampled_gt_names = sampled_dict["gt_names"][sampled_gt_masks]
                sampled_gt_boxes = sampled_dict["gt_boxes"][sampled_gt_masks]
                sampled_points = sampled_dict["points"]

                gt_dict["gt_names"] = np.concatenate(
                    [gt_dict["gt_names"], sampled_gt_names], axis=0
                )
                gt_dict["gt_boxes"] = np.concatenate(
                    [gt_dict["gt_boxes"], sampled_gt_boxes]
                )

                points = np.concatenate([sampled_points, points], axis=0)

        gt_classes = np.array(
            [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
            dtype=np.int32
        )
        gt_dict["gt_classes"] = gt_classes

        res["lidar"]["annotations"] = gt_dict
        res["lidar"]["points"] = points

        return res, info


@PIPELINES.register_module
class AffineAug(object):
    def __init__(self, cfg=None, **kwargs):
        self.mode = cfg.mode
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_std = cfg.get('global_translate_std', 0)

    def __call__(self, res, info):

        assert self.mode == "train", "For training only."

        points = res["lidar"]["points"]
        gt_dict = res["lidar"]["annotations"]
            
        gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)

        gt_dict["gt_boxes"], points = prep.global_rotation(
            gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
        )
        gt_dict["gt_boxes"], points = prep.global_scaling_v2(
            gt_dict["gt_boxes"], points, *self.global_scaling_noise
        )
        gt_dict["gt_boxes"], points = prep.global_translate_(
            gt_dict["gt_boxes"], points, noise_translate_std=self.global_translate_std
        )

        res["lidar"]["annotations"] = gt_dict
        res["lidar"]["points"] = points

        return res, info
