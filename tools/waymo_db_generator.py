import copy
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

import argparse, os
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database
from det3d.datasets.waymo import waymo_common as waymo_ds

from det3d.core import box_np_ops
from det3d.datasets.dataset_factory import get_dataset
from tqdm import tqdm

dataset_name_map = {
    "WAYMO": "WaymoDataset"
}

def generate_waymo_db(
    data_path,
    split,
    info_path=None,
    used_classes=None,
    db_path=None,
    dbinfo_path=None,
    relative_path=True,
    virtual=False,
    **kwargs,
):
    dataset_class_name = "WAYMO"
    pipeline = [
        {
            "type": "LoadPointCloudFromFile",
            "dataset": dataset_name_map[dataset_class_name],
        },
        {"type": "LoadPointCloudAnnotations", "with_bbox": True},
    ]

    dataset = get_dataset(dataset_class_name)(
        info_path=info_path, root_path=data_path, test_mode=True, pipeline=pipeline
    )

    root_path = Path(data_path)

    if dataset_class_name in ["WAYMO"]: 
        if db_path is None:
            db_path = root_path / f"gt_database_{split}_1sweeps_withvelo"
        if dbinfo_path is None:
            dbinfo_path = root_path / f"dbinfos_{split}_1sweeps_withvelo.pkl"
    else:
        raise NotImplementedError()

    db_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    group_counter = 0
    unique_id_dict = {}
    unique_id_count = 0

    for index in tqdm(range(len(dataset))):
        image_idx = index
        # modified to nuscenes
        sensor_data = dataset.get_sensor_data(index)
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]

        points = sensor_data["lidar"]["points"]
            
        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]
        unique_ids = annos["unique_ids"]

        if dataset_class_name == 'WAYMO':
            # waymo dataset contains millions of objects and it is not possible to store
            # all of them into a single folder
            # we randomly sample a few objects for gt augmentation
            # We keep all cyclist as they are rare 
            if index % 4 != 0:
                mask = (names == 'VEHICLE') 
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]
                unique_ids = unique_ids[mask]

            if index % 2 != 0:
                mask = (names == 'PEDESTRIAN')
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]
                unique_ids = unique_ids[mask]

        group_dict = {}
        # why group ids
        group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            continue 
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            unique_id = unique_ids[i]
            if (used_classes is None) or names[i] in used_classes:
                if unique_id_dict.get(unique_id, None) is None:
                    unique_id_dict[unique_id] = unique_id_count
                    unique_idx = unique_id_count
                    unique_id_count += 1
                else:
                    unique_idx = unique_id_dict[unique_id]
                filename = f"{image_idx}_{names[i]}_{i}_u{unique_idx}.bin"
                dirpath = os.path.join(str(db_path), names[i])
                os.makedirs(dirpath, exist_ok=True)

                filepath = os.path.join(str(db_path), names[i], filename)
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "w") as f:
                    try:
                        gt_points.tofile(f)
                    except:
                        print("process {} files".format(index))
                        break

            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_dump_path = os.path.join(db_path.stem, names[i], filename)
                else:
                    db_dump_path = str(filepath)

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "unique_id": unique_ids[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    print("dataset length: ", len(dataset))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, "wb") as f:
        pickle.dump(all_db_infos, f)

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
                    example:
                        python tools/waymo_info_parser.py waymo/training train
                    """,

        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('root_path', type=str)
    parser.add_argument('split', type=str)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    nsweeps = 1
    info_path = Path(args.root_path) / f"infos_{args.split}_{nsweeps:02d}sweeps_filter_zero_gt.pkl"
    generate_waymo_db(
        args.root_path, args.split, info_path,
        used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
        nsweeps=1
    )
