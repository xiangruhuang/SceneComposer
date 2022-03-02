import numpy as np
import time
import glob, os
import multiprocessing.pool

from det3d.core.bbox import box_np_ops
from det3d.core.bbox.geometry import points_in_convex_polygon_3d_jit

from det3d.structures.frame import Frame, get_frame_id, get_sequence_id

class Sequence:
    def __init__(
        self,
        seq_id,
        frames,
        dtype=np.float64,
    ):
        self.frames = frames
        self.dtype = dtype
        self.seq_id = seq_id
        self.set_scope()

    @classmethod
    def from_info(
        self,
        info,
        dtype=np.float64,
        no_points=False,
        class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST'],
        num_processes=8,
    ):
        """Load Sequence from info dict.

        Args:
            info (dict): metadata about this sequence
            no_points: if False, only load annotations 
            num_processes: multi-thread loading

        """

        frame_paths = [info['path']]
        for s in info['sweeps']:
            frame_paths.append(s['path'])
        for s in info['reverse_sweeps']:
            frame_paths.append(s['path'])
        frame_paths = sorted(frame_paths, key=get_frame_id)

        seq_id = get_sequence_id(frame_paths[0])

        pool = multiprocessing.pool.ThreadPool(processes=8)
        frames = pool.map(
                     lambda x : Frame(x, dtype, no_points,
                                      class_names=class_names),
                     frame_paths,
                     chunksize=100
                 )
        pool.close()
        return cls(seq_id, frames, dtype)

    @classmethod
    def from_index(
        cls,
        seq_id,
        root_path='data/Waymo',
        split='train',
        dtype=np.float32,
        no_points=False,
        class_names=['VEHICLE', 'PEDESTRIAN', 'CYCLIST'],
        num_processes=8,
    ):
        """Load sequence from sequence id.
        
        """

        LIDARPATH = os.path.join(
                        root_path,
                        split,
                        'lidar',
                        'seq_{}_frame_{}.pkl',
                    )
        frame_paths = glob.glob(LIDARPATH.format(seq_id, '*'))
        frame_paths = sorted(frame_paths, key=get_frame_id)

        pool = multiprocessing.pool.ThreadPool(processes=8)
        frames = pool.map(
                     lambda x : Frame(x, dtype, no_points,
                                      class_names=class_names),
                     frame_paths,
                     chunksize=100
                 )
        pool.close()

        return cls(seq_id, frames, dtype)

    def set_scope(self, start_frame=0, end_frame=-1):
        self.start_frame = start_frame
        self.end_frame = end_frame
   
    def toglobal(self):
        for frame in self.frames:
            frame.toglobal()
    
    def tolocal(self):
        for frame in self.frames:
            frame.tolocal()

    def toframe(self, frame_id):
        T = None
        for frame in self.frames:
            if frame.frame_id == frame_id:
                T = frame.T
        
        if T is None:
            raise ValueError('No frame match.')

        Tinv = np.linalg.inv(T)
        for frame in self.frames:
            frame.toglobal()
            frame.transform(Tinv)

    def center(self):
        points = self.points4d()
        self.scene_center = scene_center = points.mean(0)[:3]
        T = np.eye(4).astype(self.dtype)
        T[:3, 3] = -scene_center
        for frame in self.frames:
            frame.transform(T)

    def _gather(
        self,
        key,
        start_frame=0,
        end_frame=-1,
        attach_frame_id=False,
        return_list=False,
    ):
        if end_frame == -1:
            end_frame = len(self.frames)

        vals = []
        for f in self.frames[start_frame:end_frame]:
            val = getattr(f, key)
            if attach_frame_id:
                frame_id = np.ones((val.shape[0], 1)) * f.frame_id
                val = np.concatenate([val, frame_id], axis=-1)
            vals.append(val)

        if return_list:
            return vals
        else:
            return np.concatenate(vals, axis=0)
    
    def filter_points(self, mask):
        offset = 0
        for i, f in enumerate(self.frames):
            num_points = f.points.shape[0]
            mask_i = mask[offset:(offset+num_points)]
            f.filter(mask_i)
            offset += num_points

    def points_in_box(self, uids=None):
        points = []
        classes = []
        point_uids = []
        for frame in self.frames:
            p, cls, point_uid = frame.points_in_box(uids=uids)
            frame_id = np.ones((p.shape[0], 1)) * frame.frame_id
            p = np.concatenate([p, frame_id], axis=-1)
            cls = cls.reshape(-1)
            points.append(p)
            classes.append(cls)
            if point_uid is not None:
                point_uid = point_uid.reshape(-1)
                point_uids.append(point_uid)
        
        points = np.concatenate(points, axis=0)
        classes = np.concatenate(classes, axis=0)
        point_uids = np.concatenate(point_uids, axis=0)
        return points, classes, point_uids

    def points_in_boxes(self, moving=True):
        trace_dict = {}
        for f in self.frames:
            surfaces = box_np_ops.corner_to_surfaces_3d(f.corners)
            indices = points_in_convex_polygon_3d_jit(f.points[:, :3], surfaces)
            num_points_in_boxes = indices.astype(np.int32).sum(0)
            for box_id, uid in enumerate(f.uids):
                cls = f.classes[box_id]
                if trace_dict.get(uid, None) is None:
                    trace_dict[uid] = []
                box_dict = dict(frame_id = f.frame_id,
                                corners = f.corners[box_id],
                                cls = f.classes[box_id],
                                num_points = num_points_in_boxes[box_id])
                trace_dict[uid].append(box_dict)

        moving_uids = []
        for uid in trace_dict.keys():
            box_trace = trace_dict[uid]
            abs_travel_dist = 0
            cls = box_trace[0]['cls']

            # check if moving
            for i in range(1, len(box_trace)):
                last_box = box_trace[i-1]
                box = box_trace[i]
                abs_travel_dist += np.linalg.norm(
                                       last_box['corners'] - box['corners'],
                                       ord=2, axis=-1).mean()

            if moving and (abs_travel_dist <= 1.5):
                continue
            moving_uids.append(uid)
        
        return self.points_in_box(moving_uids)
    
    def points_not_in_box(self, uids=None):
        points = []
        for frame in self.frames:
            p = frame.points_not_in_box(uids=uids)
            frame_id = np.ones((p.shape[0], 1)) * frame.frame_id
            p = np.concatenate([p, frame_id], axis=-1)
            points.append(p)
        
        points = np.concatenate(points, axis=0)
        return points
    
    def points_not_in_boxes(self, moving=True):
        trace_dict = {}
        for f in self.frames:
            surfaces = box_np_ops.corner_to_surfaces_3d(f.corners)
            indices = points_in_convex_polygon_3d_jit(f.points[:, :3], surfaces)
            num_points_in_boxes = indices.astype(np.int32).sum(0)
            for box_id, uid in enumerate(f.uids):
                cls = f.classes[box_id]
                if trace_dict.get(uid, None) is None:
                    trace_dict[uid] = []
                box_dict = dict(frame_id = f.frame_id,
                                corners = f.corners[box_id],
                                cls = f.classes[box_id],
                                num_points = num_points_in_boxes[box_id])
                trace_dict[uid].append(box_dict)

        moving_uids = []
        for uid in trace_dict.keys():
            box_trace = trace_dict[uid]
            abs_travel_dist = 0
            cls = box_trace[0]['cls']

            # check if moving
            for i in range(1, len(box_trace)):
                last_box = box_trace[i-1]
                box = box_trace[i]
                abs_travel_dist += np.linalg.norm(
                                       last_box['corners'] - box['corners'],
                                       ord=2, axis=-1).mean()

            if moving and (abs_travel_dist <= 1.5):
                continue
            moving_uids.append(uid)
        
        return self.points_not_in_box(moving_uids)
    
    @property
    def camera_trajectory(self):
        return self._gather(
                   'camera_loc',
                   self.start_frame,
                   self.end_frame,
                   return_list=True,
               )

    @property
    def points(self):
        return self._gather(
                   'points',
                   self.start_frame,
                   self.end_frame,
                   attach_frame_id=True,
                   return_list=True,
               )
    
    @property
    def ori3d(self):
        return self._gather(
                   'ori3d',
                   self.start_frame,
                   self.end_frame,
               )

    @property
    def points4d(self):
        return self._gather(
                   'points',
                   self.start_frame,
                   self.end_frame,
                   attach_frame_id=True,
               )

    @property
    def normals(self):
        return self._gather(
                   'normals',
                   self.start_frame,
                   self.end_frame,
               )

    @property
    def boxes(self):
        return self._gather(
                   'boxes',
                   self.start_frame,
                   self.end_frame,
               )
    
    @property
    def boxes_with_velo(self):
        return self._gather(
                   'boxes_with_velo',
                   self.start_frame,
                   self.end_frame,
               )
    
    @property
    def corners(self):
        return self._gather(
                   'corners',
                   self.start_frame,
                   self.end_frame,
               )

    @property
    def velocity(self):
        start_frame, end_frame = self.start_frame, self.end_frame
        velocity = []
        if end_frame == -1:
            end_frame = len(self.frames)
        for f in self.frames[start_frame:end_frame]:
            velocity.append(f.velocity/f.vweight)
        velocity = np.concatenate(velocity, axis=0)
        return velocity
    
    @property
    def classes(self):
        return self._gather(
                   'classes',
                   self.start_frame,
                   self.end_frame,
               )
    
    @property
    def origin_classes(self):
        return self._gather(
                   'origin_classes',
                   self.start_frame,
                   self.end_frame,
               )

    @property
    def uids(self):
        return self._gather(
                   'uids',
                   self.start_frame,
                   self.end_frame,
               )

    @property
    def box_centers_4d(self):
        return self._gather(
                   'box_centers',
                   self.start_frame,
                   self.end_frame,
                   attach_frame_id=True,
               )

    @property
    def object_traces(self):
        start_frame, end_frame = self.start_frame, self.end_frame
        if end_frame == -1:
            end_frame = len(self.frames)
        object_pool = {}
        for fid, f in enumerate(self.frames[start_frame:end_frame]):
            for tid, uid in enumerate(f.uids):
                obj_box = (f.frame_id,
                           f.boxes[tid],
                           f.corners[tid],
                           f.origin_classes[tid],
                           f.global_speed[tid],
                           f.global_accel[tid],
                           uid)
                trace = object_pool.get(uid, [])
                trace.append(obj_box)
                object_pool[uid] = trace

        object_traces = []
        for uid, trace in object_pool.items():
            frame_ids, boxes, corners, classes, global_speed, global_accel, \
                uids = [[] for i in range(7)]
            for t in trace:
                frame_ids.append(t[0])
                boxes.append(t[1])
                corners.append(t[2])
                classes.append(t[3])
                global_speed.append(t[4])
                global_accel.append(t[5])
                uids.append(t[6])
            frame_ids = np.array(frame_ids)
            boxes = np.stack(boxes, axis=0)
            corners = np.stack(corners, axis=0)
            classes = np.array(classes)
            uids = np.array(uids).astype(str)
            global_speed = np.stack(global_speed, axis=0)
            global_accel = np.stack(global_accel, axis=0)
            trace_dict = dict(
                frame_ids=frame_ids,
                boxes=boxes,
                corners=corners,
                classes=classes,
                global_speed=global_speed,
                global_accel=global_accel,
                uids=uids)
            object_traces.append(trace_dict)

        return object_traces

