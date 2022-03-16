import copy
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

import argparse, os
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from det3d.datasets.waymo import waymo_common as waymo_ds

def _get_available_frames(root, split, use_frames):

    def sort_frame(frames):
        indices = [] 

        for f in frames:
            seq_id = int(f.split("_")[1])
            frame_id= int(f.split("_")[3][:-4])

            idx = seq_id * 1000 + frame_id
            indices.append(idx)

        rank = list(np.argsort(np.array(indices)))

        frames = [frames[r] for r in rank]
        return frames

    if use_frames == 'all':
        dir_path = os.path.join(root, split, 'lidar')
    elif use_frames == 'seg':
        dir_path = os.path.join(root, split, 'seg3d')
    else:
        raise ValueError("Unknown use_frames mode")

    available_frames = list(os.listdir(dir_path))

    sorted_frames = sort_frame(available_frames)

    print(split, " split ", "exist frame num:", len(available_frames))
    return sorted_frames

def _robust_inverse(T):
    inv_T = transform_matrix(
        T[:3, 3], Quaternion(matrix=T[:3, :3]), inverse=True
    )
    return inv_T

def _get_obj(path):
    with open(path, "rb") as fin:
        return pickle.load(fin)

def _fill_infos(root_path, frames, split='train', nsweeps=1):
    # load all train infos
    infos = []
    for frame_name in tqdm(frames):  # global id
        lidar_path = os.path.join(root_path, split, 'lidar', frame_name)
        ref_path = os.path.join(root_path, split, 'annos', frame_name)
        seg_path = os.path.join(root_path, split, 'seg3d', frame_name)
        if not os.path.exists(seg_path):
            seg_path = None

        ref_obj = _get_obj(ref_path)
        ref_time = 1e-6 * int(ref_obj['frame_name'].split("_")[-1])

        ref_pose = np.reshape(ref_obj['veh_to_global'], [4, 4])

        ref_from_global = _robust_inverse(ref_pose)

        info = {
            "path": lidar_path,
            "anno_path": ref_path,
            'seg_path': seg_path,
            "token": frame_name,
            "timestamp": ref_time,
        }

        sequence_id = int(frame_name.split("_")[1])
        frame_id = int(frame_name.split("_")[3][:-4]) # remove .pkl

        if split != 'test':
            # read boxes 
            TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
            annos = ref_obj['objects']
            num_points_in_gt = np.array([ann['num_points'] for ann in annos])
            gt_boxes = np.array([ann['box'] for ann in annos]).reshape(-1, 9)
            unique_ids = np.array([ann['name'] for ann in annos]).reshape(-1)
            
            if len(gt_boxes) != 0:
                # transform from Waymo to KITTI coordinate 
                # Waymo: x, y, z, length, width, height, rotation from positive x axis clockwisely
                # KITTI: x, y, z, width, length, height, rotation from negative y axis counterclockwisely 
                gt_boxes[:, -1] = -np.pi / 2 - gt_boxes[:, -1]
                gt_boxes[:, [3, 4]] = gt_boxes[:, [4, 3]]

            gt_names = np.array([TYPE_LIST[ann['label']] for ann in annos])
            mask_not_zero = (num_points_in_gt > 0).reshape(-1)    

            # filter boxes without lidar points 
            info['gt_boxes'] = gt_boxes[mask_not_zero, :].astype(np.float32)
            info['gt_names'] = gt_names[mask_not_zero].astype(str)
            info['unique_ids'] = unique_ids[mask_not_zero].astype(str)

        infos.append(info)
    return infos

def parse_waymo_info(root_path, split, use_frames, nsweeps=1):

    frames = _get_available_frames(root_path, split, use_frames)

    waymo_infos = _fill_infos(root_path, frames, split, nsweeps)
    
    print(f"sample: {len(waymo_infos)}")

    info_name = f"infos_{split}_{use_frames}.pkl"
    with open(os.path.join(root_path, info_name), "wb") as fout:
        pickle.dump(waymo_infos, fout)

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
                    example:
                        python tools/waymo_info_parser.py waymo/training train seg
                    """,

        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('root_path', type=str)
    parser.add_argument('split', type=str)
    parser.add_argument('use_frames', type=str, help="{'all', 'seg'}")
    parser.add_argument('--nsweeps', type=int, default=1)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    parse_waymo_info(args.root_path, args.split, args.use_frames, args.nsweeps)
