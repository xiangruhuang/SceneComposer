import numpy as np

from det3d.core.bbox import box_np_ops
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
