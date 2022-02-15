import pickle
import numpy as np
from det3d.core.bbox import box_np_ops
from det3d.core.bbox.geometry import points_in_convex_polygon_3d_jit
import torch
import open3d as o3d

def get_pickle(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)

def get_frame_id(path):
    return int(path.split('/')[-1].split('.')[0].split('_')[3])

class Frame:
    def __init__(self, path, dtype=np.float64, no_points=False):
        self.path = path
        self.dtype = dtype
        self.frame_id = get_frame_id(path)
        self.load_annos()
        if not no_points:
            self.load_points()
        else:
            self.points = np.zeros((0, 3)).astype(dtype)
            self.normals = np.zeros((0, 3)).astype(dtype)
        self.pose = np.eye(4).astype(self.dtype)
        self.camera_loc = np.zeros(3).astype(self.dtype)

    @classmethod
    def from_index(cls, seq_id, frame_id,
                   split='train',
                   dtype=np.float64,
                   no_points=False):
        path = f'data/Waymo/{split}/lidar/seq_{seq_id}_frame_{frame_id}.pkl'
        return cls(path, dtype=dtype, no_points=no_points)
    
    def transform(self, T):
        self.pose = T @ self.pose
        self.points = self.points @ T[:3, :3].T + T[:3, 3]
        self.normals = self.normals @ T[:3, :3].T
        self.corners = (
                self.corners.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
                ).reshape(-1, 8, 3)
        self.camera_loc = T[:3, :3] @ self.camera_loc + T[:3, 3]
    
    def load_points(self):
        lidars = get_pickle(self.path)['lidars']
        self.points = lidars['points_xyz'].astype(self.dtype)
        self.feats = lidars['points_feature'].astype(self.dtype)
        self.mask = np.ones(self.points.shape[0], dtype=bool)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                         radius=0.5, max_nn=30))
        self.normals = np.array(pcd.normals, dtype=self.dtype)

    def load_annos(self):
        anno_dict = get_pickle(self.path.replace('lidar', 'annos'))
        self.T = anno_dict['veh_to_global'].reshape(4, 4).astype(self.dtype)
        self.frame_name = anno_dict['frame_name']
        self.scene_name = anno_dict['scene_name']
        cls_map = {'VEHICLE':0, 'PEDESTRIAN':1, 'CYCLIST':2}
        label_map = {0:-1, 1:0, 2:1, 3:-1, 4:2}
        reverse_label_map = {0: 1, 1: 2, 2: 4}
        self.boxes = np.array([o['box'] for o in anno_dict['objects']],
                              dtype=self.dtype).reshape(-1, 9)
        if anno_dict['objects'][0].get('global_speed', None) is not None:
            self.global_speed = np.array([o['global_speed']
                                         for o in anno_dict['objects']],
                                         dtype=self.dtype).reshape(-1, 2)
        if anno_dict['objects'][0].get('global_accel', None) is not None:
            self.global_accel = np.array([o['global_accel']
                                         for o in anno_dict['objects']],
                                         dtype=self.dtype).reshape(-1, 2)
        self.tokens = np.array([o['name'] for o in anno_dict['objects']]
                               ).astype(str)
        cls = [label_map[o['label']] if o['num_points'] > 0 else -1 \
               for o in anno_dict['objects']]
        self.classes = np.array(cls)
        if self.boxes.shape[0] > 0:
            self.corners = box_np_ops.center_to_corner_box3d(
                self.boxes[:, :3], self.boxes[:, 3:6], -self.boxes[:, -1],
                origin=(0.5, 0.5, 0.3), axis=2)
        else:
            self.corners = np.zeros((0, 8, 3), dtype=self.dtype)
        mask = (self.classes != -1)
        self.corners = self.corners[mask]
        self.classes = self.classes[mask]
        self.origin_classes = np.array([reverse_label_map[c]
                                        for c in self.classes])
        self.boxes = self.boxes[mask]
        self.tokens = self.tokens[mask]

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
        self.points = self.points[mask]
        self.feats = self.feats[mask]
        self.normals = self.normals[mask]

    def points_in_box(self, tokens = None):
        box_tokens = []
        if tokens is not None:
            sel_indices = []
            for i, token in enumerate(self.tokens):
                if token in tokens:
                    sel_indices.append(i)
                    box_tokens.append(token)
            sel_indices = np.array(sel_indices).astype(np.int32)
            if len(box_tokens) == 0:
                box_tokens = None
            else:
                box_tokens = np.array(box_tokens)
        else:
            sel_indices = np.arange(self.corners.shape[0])
            box_tokens = None
        
        surfaces = box_np_ops.corner_to_surfaces_3d(self.corners[sel_indices])
        box_classes = self.classes[sel_indices]
        indices = points_in_convex_polygon_3d_jit(self.points[:, :3], surfaces)
        box_indices = torch.tensor(indices).long().argmax(-1)[indices.any(axis=-1)]
        points = self.points[indices.any(axis=-1)]
        cls = box_classes[box_indices]
        if box_tokens is not None:
            point_tokens = box_tokens[box_indices]
        else:
            point_tokens = None
        
        return points, cls, point_tokens

    def points_not_in_box(self, tokens = None):
        box_tokens = []
        if tokens is not None:
            sel_indices = []
            for i, token in enumerate(self.tokens):
                if token in tokens:
                    sel_indices.append(i)
                    box_tokens.append(token)
            sel_indices = np.array(sel_indices).astype(np.int32)
            if len(box_tokens) == 0:
                box_tokens = None
            else:
                box_tokens = np.array(box_tokens)
        else:
            sel_indices = np.arange(self.corners.shape[0])
            box_tokens = None
        
        surfaces = box_np_ops.corner_to_surfaces_3d(self.corners[sel_indices])
        box_classes = self.classes[sel_indices]
        indices = points_in_convex_polygon_3d_jit(self.points[:, :3], surfaces)
        box_indices = torch.tensor(indices).long().argmax(-1)[indices.any(axis=-1)]
        points = self.points[indices.any(axis=-1) == False]
        
        return points
