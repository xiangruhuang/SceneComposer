import numpy as np
import pickle
import torch
import os
import open3d as o3d

from det3d.core.bbox import box_np_ops
from det3d.core.bbox.geometry import points_in_convex_polygon_3d_jit

def get_pickle(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)

def get_frame_id(path):
    return int(path.split('/')[-1].split('.')[0].split('_')[3])

def get_sequence_id(path):
    return int(path.split('/')[-1].split('.')[0].split('_')[1])

class Frame:
    def __init__(
        self,
        path,
        dtype=np.float64,
        no_points=False,
        compute_normal=False,
        class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST'],
    ):
        self.path = path
        self.dtype = dtype
        self.frame_id = get_frame_id(path)
        self.compute_normal = compute_normal

        self.load_annos(class_names)

        if not no_points:
            self.load_points()
        else:
            self.points = np.zeros((0, 3)).astype(dtype)
            self.normals = np.zeros((0, 3)).astype(dtype)

        self.pose = np.eye(4).astype(self.dtype)
        self.camera_loc = np.zeros(3).astype(self.dtype)

    @classmethod
    def from_index(
        cls,
        seq_id,
        frame_id,
        root_path='data/Waymo',
        split='train',
        dtype=np.float64,
        no_points=False,
        class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST'],
    ):
        path = os.path.join(
                   root_path,
                   split,
                   'lidar',
                   f'seq_{seq_id}_frame_{frame_id}.pkl',
               )
        return cls(path,
                   dtype=dtype,
                   no_points=no_points,
                   class_names=class_names
               )
    
    def transform(self, T):
        self.pose = T @ self.pose

        if self.points is not None:
            self.points = self.points @ T[:3, :3].T + T[:3, 3]

        if self.normals is not None:
            self.normals = self.normals @ T[:3, :3].T

        self.box_centers = self.box_centers @ T[:3, :3].T + T[:3, 3]
        self.ori3d = T[:3, :3] @ self.ori3d

        # approximate 2D orientation
        cost = (self.ori3d[:, 0, 0] + self.ori3d[:, 1, 1])/2.0
        sint = (self.ori3d[:, 0, 1] - self.ori3d[:, 1, 0])/2.0
        self.ori2d = np.arctan2(sint, cost)

        self.corners = (self.corners.reshape(-1, 3) @ T[:3, :3].T \
                        + T[:3, 3]).reshape(-1, 8, 3)

        self.camera_loc = T[:3, :3] @ self.camera_loc + T[:3, 3]
    
    def load_points(self):
        lidars = get_pickle(self.path)['lidars']
        self.points = lidars['points_xyz'].astype(self.dtype)
        self.feats = lidars['points_feature'].astype(self.dtype)
        self.mask = np.ones(self.points.shape[0], dtype=bool)
        if self.compute_normal:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                             radius=0.5, max_nn=30))
            self.normals = np.array(pcd.normals, dtype=self.dtype)
        else:
            self.normals = None

    def load_annos(self, class_names):
        anno_dict = get_pickle(self.path.replace('lidar', 'annos'))

        # wall clock time in second.
        self.time = 1e-6 * int(anno_dict['frame_name'].split("_")[-1])

        # load the pose (from this) to global
        self.T = anno_dict['veh_to_global'].reshape(4, 4).astype(self.dtype)

        # used as a unique scene name
        self.scene_name = anno_dict['scene_name']

        # ignore objects of no interest
        objects = list(filter(lambda o: o['class_name'] in class_names,
                              anno_dict['objects']))
        
        # unique object ids
        self.uids = np.array([o['name'] for o in objects]).astype(str)

        # parse boxes
        boxes = np.array(
                    [o['box'] for o in objects]
                ).reshape(-1, 9).astype(self.dtype)
        
        self.classes = np.array([class_names.index(o['class_name']) for o in objects])
            
        # remove boxes that are empty or of ignored classes
        num_points = np.array([o['num_points'] for o in objects])
        mask = num_points > 0
        self.uids = self.uids[mask]
        boxes = boxes[mask]
        self.classes = self.classes[mask]
        
        self.num_objs = boxes.shape[0]
        if self.num_objs != 0:
            # transform from Waymo to KITTI coordinate 
            # Waymo: x, y, z, length, width, height, rotation 
            #    from positive x axis clockwisely
            # KITTI: x, y, z, width, length, height, rotation 
            #    from negative y axis counterclockwisely 
            boxes[:, -1] = -np.pi / 2 - boxes[:, -1]
            boxes[:, [3, 4]] = boxes[:, [4, 3]]
            
        # manage box attributes
        self.box_centers = boxes[:, :3]
        self.box_dims = boxes[:, 3:6]
        theta = boxes[:, -1]
        zeros, ones = np.zeros(self.num_objs), np.ones(self.num_objs)
        cost, sint = np.cos(theta), np.sin(theta)

        # orientation of box (in 3D)
        self.ori3d = np.array(
                         [[cost,  -sint, zeros],
                          [sint,   cost, zeros],
                          [zeros, zeros,  ones]],
                     ).transpose(2, 1, 0)

        self._boxes = boxes
        
        # approximate 2D orientation
        _cost = (self.ori3d[:, 0, 0] + self.ori3d[:, 1, 1])/2.0
        _sint = (self.ori3d[:, 1, 0] - self.ori3d[:, 0, 1])/2.0
        self.ori2d = np.arctan2(_sint, _cost)
        
        if self.num_objs != 0:
            if objects[0].get('global_speed', None) is not None:
                self.global_speed = np.array(
                                        [o['global_speed'] for o in objects],
                                        dtype=self.dtype
                                    ).reshape(-1, 2)[mask]
            if objects[0].get('global_accel', None) is not None:
                self.global_accel = np.array(
                                        [o['global_accel'] for o in objects],
                                        dtype=self.dtype
                                    ).reshape(-1, 2)[mask]

        # for visualization purpose, we also keep track of the box corners
        if boxes.shape[0] > 0:
            self.corners = box_np_ops.center_to_corner_box3d(
                               boxes[:, :3],
                               boxes[:, 3:6],
                               boxes[:, -1],
                               axis=2
                           )
        else:
            self.corners = np.zeros((0, 8, 3), dtype=self.dtype)

    @property
    def boxes(self):
        return np.concatenate(
                   [self.box_centers,
                    self.box_dims,
                    self.ori2d[:, np.newaxis]],
                   axis=-1,
               )
    
    @property
    def boxes_with_velo(self):
        if hasattr(self, 'global_speed'):
            global_speed = self.global_speed
        else:
            global_speed = np.zeros((np.box_centers.shape[0], 2))
        return np.concatenate(
                   [self.box_centers,
                    self.box_dims,
                    global_speed,
                    self.ori2d[:, np.newaxis]],
                   axis=-1,
               )

    def toglobal(self):
        T = self.T @ np.linalg.inv(self.pose)
        self.transform(T)

    def tolocal(self):
        T = np.linalg.inv(self.pose)
        self.transform(T)

    def filter(self, mask):
        """
        Args:
            mask (N): the same size as the current number of points

        """
        premask = self.mask == True
        newmask = self.mask[premask]
        newmask[:] = False
        newmask[mask] = True

        self.mask[premask] = newmask
        self.feats = self.feats[mask]
        
        if self.points is not None:
            self.points = self.points[mask]

        if self.normals is not None:
            self.normals = self.normals[mask]

    def points_in_box(self, uids = None):
        """Return all points in the boxes. If uids is not None,
        only use boxes specified by uids, otherwise, use all boxes.
        
        Args:

            uids (list): list of object unique IDs.

        Returns:
            points (np.ndarray, [N_in, 3]): points in boxes.
            cls (np.ndarray, [N_in]): the class of box for each point.
            point_uids (np.ndarray, [N_in]): object uIDs (str) for each point.

        """
        if self.points is None:
            raise ValueError('No Points Loaded.')
        box_uids = []
        if uids is not None:
            sel_indices = []
            for i, uid in enumerate(self.uids):
                if uid in uids:
                    sel_indices.append(i)
                    box_uids.append(uid)
            sel_indices = np.array(sel_indices).astype(np.int32)
            if len(box_uids) == 0:
                box_uids = None
            else:
                box_uids = np.array(box_uids)
        else:
            sel_indices = np.arange(self.corners.shape[0])
            box_uids = None
        
        surfaces = box_np_ops.corner_to_surfaces_3d(self.corners[sel_indices])
        box_classes = self.classes[sel_indices]
        indices = points_in_convex_polygon_3d_jit(self.points[:, :3], surfaces)
        box_indices = torch.tensor(indices).long().argmax(-1)[indices.any(axis=-1)]
        points = self.points[indices.any(axis=-1)]
        cls = box_classes[box_indices]
        if box_uids is not None:
            point_uids = box_uids[box_indices]
        else:
            point_uids = None
        
        return points, cls, point_uids

    def points_not_in_box(self, uids = None):
        """Return all points **not** in the boxes. If uids is not None,
        only use boxes specified by uids, otherwise, use all boxes.
        
        Args:

            uids (list): list of object unique IDs.

        Returns:
            points (np.ndarray, [N_in, 3]): points not in any selected box.

        """
        if self.points is None:
            raise ValueError('No Points Loaded')
        box_uids = []
        if uids is not None:
            sel_indices = []
            for i, uid in enumerate(self.uids):
                if uid in uids:
                    sel_indices.append(i)
                    box_uids.append(uid)
            sel_indices = np.array(sel_indices).astype(np.int32)
            if len(box_uids) == 0:
                box_uids = None
            else:
                box_uids = np.array(box_uids)
        else:
            sel_indices = np.arange(self.corners.shape[0])
            box_uids = None
        
        surfaces = box_np_ops.corner_to_surfaces_3d(self.corners[sel_indices])
        box_classes = self.classes[sel_indices]
        indices = points_in_convex_polygon_3d_jit(self.points[:, :3], surfaces)
        box_indices = torch.tensor(indices).long().argmax(-1)[indices.any(axis=-1)]
        points = self.points[indices.any(axis=-1) == False]
        
        return points
