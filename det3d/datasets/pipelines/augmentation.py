import numpy as np
import pickle
import glob, os

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
from det3d.structures import Sequence, get_sequence_id, get_frame_id
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

@PIPELINES.register_module
class SceneAug(object):
    def __init__(self, split, cfg, **kwargs):
        nsweeps = cfg.get('nsweeps', 1)
        root_path = cfg.root_path
        ANNO_PATH = os.path.join(
                        root_path,
                        split,
                        'annos',
                        'seq_*_frame_0.pkl'
                    )
        seq_paths = glob.glob(ANNO_PATH)
        seq_ids = list(map(get_sequence_id, seq_paths))

        seq_ = {}
        for seq_id in seq_ids:
            seq = Sequence.from_index(
                      seq_id,
                      root_path,
                      split,
                      no_points=True,
                      class_names=cfg.class_names,
                  )
            seq_[seq_id] = seq

        self.compress_static = cfg.get("compress_static", True)
        
        self.seq_ = seq_
        self.class_names = cfg.class_names

        self.nsweeps = nsweeps
        
    def __call__(self, res, info):
        if self.nsweeps == 1:
            return res, info
        
        seq_id = get_sequence_id(info['anno_path'])
        frame_id = get_frame_id(info['anno_path'])
        
        seq = self.seq_[seq_id]

        seq.toframe(frame_id)

        start_frame_id = max(frame_id-self.nsweeps+1, 0)
        end_frame_id = min(frame_id+self.nsweeps, len(seq.frames))
        seq.set_scope(start_frame_id, end_frame_id)

        boxes = seq.boxes_with_velo
            
        uids = seq.uids
        classes = seq.classes.astype(np.int64)
        names = np.array([self.class_names[cls] for cls in classes]).astype(str)
        classes = classes + 1

        if self.compress_static:
            mask = np.ones(uids.shape[0], dtype=bool)
            unique_ids = np.unique(uids)
            velo = np.linalg.norm(boxes[:, 6:8], axis=-1, ord=2)
            velo_dict = {u: velo[uids == u].mean() for u in unique_ids}

            for u in unique_ids:
                u_indices = np.where(uids == u)[0]
                umask = np.ones(u_indices.shape[0], dtype=bool)
                for i, idx in enumerate(u_indices):
                    if not umask[i]:
                        continue
                    for j, idx2 in enumerate(u_indices[i+1:]):
                        dist = np.linalg.norm(boxes[idx, :3] - boxes[idx2, :3], ord=2)
                        if dist < 0.1:
                            umask[j] = False
                mask[u_indices] = umask
                    
            uids = uids[mask]
            classes = classes[mask]
            names = names[mask]
            boxes = boxes[mask]
        
        info['gt_names'] = names
        info['gt_boxes'] = boxes
        info['unique_ids'] = uids

        res["lidar"]["annotations"]["gt_boxes"] = boxes 
        res["lidar"]["annotations"]["gt_names"] = names
        res["lidar"]["annotations"]["gt_classes"] = classes

        return res, info

@PIPELINES.register_module
class SemanticAug(object):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.root_path = cfg.get("root_path", 'data/Waymo')
        self.num_sample = cfg.get("num_sample", None)
        self.ratio = cfg.get("ratio", 1.5)
        with open(cfg.info_path, "rb") as fin:
            self.objects = pickle.load(fin)

    def _retrieve_obj(self, objects, num_sample=1):
        num_objects = len(objects)

        indices = np.random.permutation(num_objects)[:num_sample]
        
        boxes, point_clouds = [], []
        for index in indices:
            path = os.path.join(self.root_path,
                                objects[index]['path'])
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
            box = objects[index]['box3d_lidar']
            num_points = objects[index]["num_points_in_gt"]
            boxes.append(box)
            point_clouds.append(points)
        boxes = np.stack(boxes, axis=0)

        return boxes, point_clouds

    def _sample_locations(self, walkable_points, num_sample=1):
        num_points = walkable_points.shape[0]
        
        indices = np.random.permutation(num_points)[:num_sample]

        locations = []
        for index in indices:
            box_location = walkable_points[index, :3]
            locations.append(box_location)
        locations = np.stack(locations, axis=0)
            
        return locations

    def __call__(self, res, info):

        points = res["lidar"]["points"]
        num_sample = self.num_sample["PEDESTRIAN"]
        seg_labels = res["lidar"]["annotations"].pop("seg_labels")

        # find marked points
        points = points[:seg_labels.shape[0]]
        mask = (seg_labels != 0).any(-1)
        points = points[mask]
        seg_labels = seg_labels[mask]

        # find walkable regions
        road_region = seg_labels[:, 1] > 17
        walkable_points = points[road_region]
        non_walkable_points = points[road_region == False]

        if walkable_points.shape[0] == 0:
            return res, info

        # sample objects and locations to place them
        boxes, point_clouds = self._retrieve_obj(self.objects['PEDESTRIAN'],
                                                 num_sample)
        locations = self._sample_locations(walkable_points, num_sample)

        # create fake boxes that are larger by a ratio r
        boxes[:, :3] = locations
        boxes[:, 2] += boxes[:, 5] / 2
        fake_boxes = boxes.copy()
        fake_boxes[:, 3:6] *= self.ratio
        boxes_all = np.concatenate([boxes, fake_boxes], axis=0)
        
        # compute overlap
        corners = box_np_ops.center_to_corner_box3d(
                      boxes_all[:, :3],
                      boxes_all[:, 3:6],
                      boxes_all[:, -1],
                      axis=2)
        surfaces = box_np_ops.corner_to_surfaces_3d(corners)
        box_overlap = box_np_ops.points_in_convex_polygon_3d_jit(
                          non_walkable_points, surfaces
                      ).any(0)

        obj_valid_indices = np.where(
                                np.logical_not(box_overlap[:boxes.shape[0]]) & \
                                box_overlap[boxes.shape[0]:]
                            )[0]
        
        # check validity
        valid_point_clouds, valid_boxes = [], []
        for idx in obj_valid_indices:
            sampled_points_this = point_clouds[idx]
            sampled_points_this[:, :3] += boxes[idx, :3]
            valid_point_clouds.append(sampled_points_this)
            valid_boxes.append(boxes[idx])

        num_sampled = len(valid_point_clouds)
        if num_sampled > 0:
            sampled_points = np.concatenate(valid_point_clouds, axis=0)
            sampled_boxes = np.stack(valid_boxes, axis=0)

            #from det3d.core import Visualizer
            #vis = Visualizer()
            #vis.pointcloud('w-points', walkable_points[:, :3])
            #vis.pointcloud('points', non_walkable_points[:, :3])
            #vis.boxes_from_attr('sampled_boxes', sampled_boxes, np.ones(sampled_boxes.shape[0]).astype(np.int32))

            #vis.pointcloud('sampled', sampled_points[:, :3])

            res["lidar"]["points"] = np.concatenate([
                                         res["lidar"]["points"],
                                         sampled_points], axis=0)

            gt_dict = res["lidar"]["annotations"]
            gt_boxes = gt_dict["gt_boxes"]
            gt_names = gt_dict["gt_names"]
            gt_classes = gt_dict["gt_classes"]

            sampled_names = np.array(["PEDESTRIAN" for i in range(num_sampled)]
                                    ).astype(str)
            sampled_classes = np.ones(num_sampled).astype(np.int32) + 1

            gt_boxes = np.concatenate([gt_boxes, sampled_boxes], axis=0)
            gt_names = np.concatenate([gt_names, sampled_names], axis=0)
            gt_classes = np.concatenate([gt_classes, sampled_classes], axis=0)

            gt_dict = dict(
                gt_boxes=gt_boxes,
                gt_names=gt_names,
                gt_classes=gt_classes,
            )

            res["lidar"]["annotations"] = gt_dict

        return res, info
