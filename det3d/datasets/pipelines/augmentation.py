import numpy as np
import pickle
import glob, os
from collections import defaultdict

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
        self.class_names = cfg.class_names

        with open(cfg.info_path, "rb") as fin:
            self.objects = pickle.load(fin)

        self.distance_to_npoints = {}
        for key, obj in self.objects.items():
            self.objects[key] = [o for o in obj if o['num_points_in_gt'] >= 5]
            dists = [np.linalg.norm(o['box3d_lidar'][:3]) for o in obj]
            num_points = [o['num_points_in_gt'] for o in obj]

            num_points_by_dist = {i: [] for i in range(0, 100)}
            for d, n in zip(dists, num_points):
                d_int = int(np.round(d))
                num_points_by_dist[d_int].append(n)
            last_npoints = 1000000
            distance_to_npoints_this = {}
            for d in range(0, 100):
                many = []
                for di in range(max(d-5, 0), min(d+5, 100)):
                    many += num_points_by_dist[di]
                if len(many) == 0:
                    npoints = last_npoints
                else:
                    npoints = np.mean(many)
                    if (npoints > last_npoints) or (np.isnan(npoints)):
                        npoints = last_npoints
                    else:
                        last_npoints = npoints
                distance_to_npoints_this[d] = int(npoints)
            self.distance_to_npoints[key] = distance_to_npoints_this

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
            assert points.shape[0] == num_points
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

    def _subsample_by_dist(self, boxes, point_clouds, classes):
        dists = np.linalg.norm(boxes[:, :3], axis=-1, ord=2)
        for i, pc in enumerate(point_clouds):
            dist = int(np.round(dists[i]))
            key = self.class_names[classes[i]-1]
            npoints = self.distance_to_npoints[key][dist]
            if pc.shape[0] > npoints:
                rand_idx = np.random.permutation(pc.shape[0])[:npoints]
                point_clouds[i] = point_clouds[i][rand_idx]
            
        return boxes, point_clouds, classes

    def _update_objects(self, boxes, point_clouds, locations):
        """
        Args:
            boxes ([N, 9])
            point_clouds list([N, 3])
            locations [N, 3]

        Returns:
            boxes (rotated)
            point_clouds (rotated)
        """
        
        box_x, box_y = boxes[:, :2].T
        box_theta = np.arctan2(box_y, box_x)
        loc_x, loc_y = locations[:, :2].T
        loc_theta = np.arctan2(loc_y, loc_x)
        rot_theta = loc_theta - box_theta
        
        boxes[:, -1] -= rot_theta
        rot_sint, rot_cost = np.sin(rot_theta), np.cos(rot_theta)
        # counterclockwise
        R = np.stack([rot_cost, -rot_sint, rot_sint, rot_cost], axis=-1).reshape(-1, 2, 2)
        for i in range(len(point_clouds)):
            point_clouds[i][:, :2] = point_clouds[i][:, :2] @ R[i].T

        boxes[:, :3] = locations
        boxes[:, 2] += boxes[:, 5] / 2

        return boxes, point_clouds

    def _reject_by_collision(self, boxes, point_clouds, classes):
        num_boxes = boxes.shape[0]

        # bird eye view box corners
        boxes_bv = box_np_ops.center_to_corner_box2d(
                       boxes[:, :2], boxes[:, 3:5], boxes[:, -1]
                   )

        # compute collision matrix
        coll_mat = prep.box_collision_test(boxes_bv, boxes_bv)
        diag = np.arange(boxes_bv.shape[0])
        coll_mat[diag, diag] = False

        # find valid boxes (greedy)
        visited = np.zeros(boxes_bv.shape[0], dtype=bool)
        sampled_boxes, sampled_point_clouds, sampled_classes = [], [], []
        for i in range(num_boxes):
            if not visited[i]:
                visited[i] = True
                sampled_boxes.append(boxes[i])
                sampled_point_clouds.append(point_clouds[i])
                sampled_classes.append(classes[i])
                visited[coll_mat[i, :]] = True
        sampled_boxes = np.array(sampled_boxes).reshape(-1, 9)
        sampled_classes = np.array(sampled_classes).reshape(-1).astype(np.int32)
        
        return sampled_boxes, sampled_point_clouds, sampled_classes

    def _keep_interacting_objects(self, boxes, point_clouds, non_walkable_points):

        # create fake boxes that are larger by a ratio r
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
        
        # keep only interacting objects
        valid_point_clouds, valid_boxes = [], []
        for idx in obj_valid_indices:
            sampled_points_this = point_clouds[idx]
            valid_point_clouds.append(sampled_points_this)
            valid_boxes.append(boxes[idx])
        valid_boxes = np.array(valid_boxes).reshape(-1, 9)

        return valid_boxes, valid_point_clouds

    def sample_class(self, non_walkable_points, walkable_points, key, num_sample):

        # sample objects and locations to place them
        boxes, point_clouds = self._retrieve_obj(self.objects[key],
                                                 num_sample)
        locations = self._sample_locations(walkable_points, num_sample)

        # orient and update boxes and points based on locations
        boxes, point_clouds = self._update_objects(boxes, point_clouds, locations)

        # filter objects 
        boxes, point_clouds = self._keep_interacting_objects(
                                  boxes, point_clouds,
                                  non_walkable_points)

        classes = np.array([self.class_names.index(key) + 1 for i in range(boxes.shape[0])], dtype=np.int32)

        return boxes, point_clouds, classes

    def __call__(self, res, info):

        points = res["lidar"]["points"]
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

        sampled_boxes, sampled_point_clouds, sampled_classes = [], [], []

        for key, num_sample in self.num_sample.items():
            if num_sample == 0:
                continue
            sampled_class_objects = self.sample_class(non_walkable_points, walkable_points, key, num_sample)
            
            sampled_boxes.append(sampled_class_objects[0])
            sampled_point_clouds += sampled_class_objects[1]
            sampled_classes.append(sampled_class_objects[2])
            
        sampled_boxes = np.concatenate(sampled_boxes, axis=0)
        sampled_classes = np.concatenate(sampled_classes, axis=0)

        sampled_objects = (sampled_boxes, sampled_point_clouds, sampled_classes)

        sampled_objects = self._reject_by_collision(*sampled_objects)
            
        sampled_objects = self._subsample_by_dist(*sampled_objects)

        # update res
        sampled_boxes, sampled_point_clouds, sampled_classes = sampled_objects
        num_sampled = len(sampled_point_clouds)
        if num_sampled > 0:
            for i, pc in enumerate(sampled_point_clouds):
                sampled_point_clouds[i][:, :3] += sampled_boxes[i, :3]
            sampled_points = np.concatenate(sampled_point_clouds, axis=0)

            res["lidar"]["points"] = np.concatenate([
                                         res["lidar"]["points"],
                                         sampled_points], axis=0)

            gt_dict = res["lidar"]["annotations"]
            gt_boxes = gt_dict["gt_boxes"]
            gt_names = gt_dict["gt_names"]
            gt_classes = gt_dict["gt_classes"]

            sampled_names = np.array([self.class_names[cls-1] for cls in sampled_classes]
                                    ).astype(str)

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
