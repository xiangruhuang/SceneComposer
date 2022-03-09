import numpy as np

from det3d.core.bbox import box_np_ops
from det3d.core.utils.center_utils import (
    draw_umich_gaussian as draw_gaussian
)
from ..registry import PIPELINES

def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

def flatten(box):
    return np.concatenate(box, axis=0)

def merge_multi_group_label(gt_classes, num_classes_by_task): 
    num_task = len(gt_classes)
    offset = 0 

    for i in range(num_task):
        gt_classes[i] += offset 
        offset += num_classes_by_task[i]

    return flatten(gt_classes)

def organize_tasks(gt_dict, class_names_by_task):
    task_masks = []
    offset = 0
    for class_name in class_names_by_task:
        task_masks.append(
            [
                np.where(
                    gt_dict["gt_classes"] == class_name.index(i) + 1 + offset
                )
                for i in class_name
            ]
        )
        offset += len(class_name)

    task_boxes = []
    task_classes = []
    task_names = []
    offset2 = 0
    for idx, mask in enumerate(task_masks):
        task_box = []
        task_class = []
        task_name = []
        for m in mask:
            task_box.append(gt_dict["gt_boxes"][m])
            task_class.append(gt_dict["gt_classes"][m] - offset2)
            task_name.append(gt_dict["gt_names"][m])
        task_boxes.append(np.concatenate(task_box, axis=0))
        task_classes.append(np.concatenate(task_class))
        task_names.append(np.concatenate(task_name))
        offset2 += len(mask)

    return task_boxes, task_classes, task_names
       
def gaussian_radius(dims, min_overlap=0.5):
    l, w = dims.T

    a1  = 1
    b1  = (l + w)
    c1  = w * l * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (l + w)
    c2  = (1 - min_overlap) * w * l
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (l + w)
    c3  = (min_overlap - 1) * w * l
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2

    return np.minimum(r1, r2, r3)

class HeatmapGenerator(object):
    def __init__(self, cfg, **kwargs):
        self.out_size_factor = cfg.out_size_factor
        self.gaussian_overlap = cfg.gaussian_overlap
        self._min_radius = cfg.min_radius

    def __call__(self, num_classes, voxel_cfg, boxes, classes,
                 visibility=None):
        """
        Args:
            num_classes (int)
            voxel_cfg (dict)
            boxes (np.ndarray, [N, 7])
            classes (np.ndarray, [N])
            visibility (np.ndarray, [L, W])

        Returns:
            hm (np.ndarray, [num_classes, L, W])
            anno_box (np.ndarray, [N_out, 10])
            indices (np.ndarray, [N_out])
            classes (np.ndarray, [N_out])

        """

        # size of heatmap and voxel
        grid_size = voxel_cfg["shape"]
        pc_range = voxel_cfg["range"]
        voxel_size = voxel_cfg["size"]
        feature_map_size = grid_size[:2] // self.out_size_factor
        effective_voxel_size = voxel_size[:2] * self.out_size_factor
        hm = np.zeros((num_classes,
                       feature_map_size[1],
                       feature_map_size[0]),
                      dtype=np.float32)

        # remove small boxes
        mask = (boxes[:, 3:5] / effective_voxel_size > 0).all(-1)
        boxes, classes = boxes[mask], classes[mask]
        
        # compute coordinates of box centers
        coors = (boxes[:, :2] - pc_range[:2]) / effective_voxel_size
        coors_int = coors.astype(np.int32)

        # remove out of boundary objects/boxes
        mask2 = ((coors_int[:, :2] < feature_map_size[:2]) \
                 & (coors_int[:, :2] >= 0)).all(-1)
        coors, coors_int = coors[mask2], coors_int[mask2]
        boxes, classes = boxes[mask2], classes[mask2]

        # remove invisible boxes
        if visibility is not None:
            mask3 = visibility[(coors_int[:, 1], coors_int[:, 0])] > 0.05
            coors, coors_int = coors[mask3], coors_int[mask3]
            boxes, classes = boxes[mask3], classes[mask3]
        
        # compute radius of gaussian
        _box_dims = boxes[:, 3:5] / effective_voxel_size
        radius = gaussian_radius(_box_dims, min_overlap=self.gaussian_overlap)
        radius = np.maximum(self._min_radius, radius.astype(np.int32))

        # draw gaussian distribution
        for cls_id, ct, rad in zip(classes, coors_int, radius):
            draw_gaussian(hm[cls_id-1], ct[:2], rad)
        
        # record indices of peak
        #pos_indices = np.where(hm > 0)
        #classes = pos_indices[0]
        #indices = pos_indices[1] * feature_map_size[0] + pos_indices[2]
        indices = coors_int[:, 1] * feature_map_size[0] + coors_int[:, 0]

        velo, rot = boxes[:, 6:8], boxes[:, -1]

        # remove unique indices
        unique_indices, reverse_indices = np.unique(indices, return_index=True)
        coors, coors_int = coors[reverse_indices], coors_int[reverse_indices]
        boxes, classes = boxes[reverse_indices], classes[reverse_indices]
        velo, rot = velo[reverse_indices], rot[reverse_indices]
        indices = unique_indices

        anno_box = np.concatenate(
                [coors[:, :2] - coors_int[:, :2],
                 boxes[:, 2:3],
                 np.log(boxes[:, 3:6]),
                 velo,
                 np.sin(rot)[:, np.newaxis],
                 np.cos(rot)[:, np.newaxis]],
                axis=-1)

        assert anno_box.shape[-1] == 10

        return hm, anno_box, indices.astype(np.int64), classes


@PIPELINES.register_module
class AssignLabel2(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.cfg = assigner_cfg
        self.heatmap_generator = HeatmapGenerator(assigner_cfg)

    def __call__(self, res, info):
        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]

        example = {}

        if res["mode"] == "train":

            gt_dict = res["lidar"]["annotations"]

            # reorganize the gt_dict by tasks
            task_boxes, task_classes, task_names = \
                        organize_tasks(gt_dict, class_names_by_task)
            
            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict
            
            # compute heatmap and box annotations by task
            hms, anno_boxs, indices, cats, neg_indices = [], [], [], [], []
            visibility = res["lidar"]["visibility"]
            occupancy = res["lidar"]["occupancy"]

            for idx, task in enumerate(self.tasks):
                num_classes = len(class_names_by_task[idx])

                hm, anno_box, index, cat = self.heatmap_generator(
                                               num_classes,
                                               res["lidar"]["voxels"],
                                               gt_dict["gt_boxes"][idx],
                                               gt_dict["gt_classes"][idx] - 1,
                                               visibility.T
                                           )
                neg_mask = (visibility.T < 0.05) | (occupancy.T == 1)
                neg_index = np.where(neg_mask)
                for cls in range(hm.shape[0]):
                    hm[(cls, *neg_index)] = -1.0
                hm = (hm + 1.0)/2.0
                neg_index = neg_index[1] * hm.shape[2] + neg_index[0]
                neg_index = neg_index.astype(np.int64)

                hms.append(hm)
                
                anno_boxs.append(anno_box)
                indices.append(index)
                cats.append(cat)
                neg_indices.append(neg_index)

            # used for two stage code 
            boxes = flatten(gt_dict['gt_boxes'])
            classes = merge_multi_group_label(gt_dict['gt_classes'], num_classes_by_task)

            boxes_and_cls = np.concatenate(
                                (boxes, 
                                 classes.reshape(-1, 1).astype(np.float32)),
                                axis=1)

            # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
            boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
            
            # anno_boxs[task_id] = [N, 10]
            # collated anno_boxs[task_id][batch_id] = [N, 10]

            data = dict(
                gt_boxes_and_cls=boxes_and_cls,
                anno_box=anno_boxs,
                ind=indices,
                cat=cats,
                neg_ind=neg_indices
            )

            example.update({'batch_data': data, 'hm': hms})
        else:
            pass

        res["lidar"]["targets"] = example

        return res, info
