import argparse
import os, pickle
import numpy as np
from multiprocessing import Pool
import glob
import zlib
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import box_utils
      
TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

args = None

def convert_range_image_to_point_cloud_labels(frame,
        range_images,
        segmentation_labels,
        ri_index=0):
    """Convert segmentation labels from range images to point clouds.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      segmentation_labels: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      ri_index: 0 for the first return, 1 for the second return.

    Returns:
      point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
      range_image = range_images[c.name][ri_index]
      range_image_tensor = tf.reshape(
          tf.convert_to_tensor(range_image.data), range_image.shape.dims)
      range_image_mask = range_image_tensor[..., 0] > 0

      if c.name in segmentation_labels:
        assert c.name == dataset_pb2.LaserName.TOP
        sl = segmentation_labels[c.name][ri_index]
        sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
        sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        point_labels.append(sl_points_tensor.numpy())

    return point_labels

def decode_frame(frame, frame_id):
    """Extract points, segmentation labels and objects from frame.

    Args:
        frame: frame proto buffer
        frame_id: the index of this frame in the sequence

    Returns:
        lidar3d: lidar points
        seg3d: segmentation labels
        det3d: detection labels
    
    """
    (range_images, camera_projections, segmentation_labels,
     range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
        frame)

    veh_to_global = np.array(frame.pose.transform)
        
    #Load points in vehicle frames. Sort by lidar cameras so that 
    # points from lidar.TOP goes first and aligns with segmentation labels
    # (Segmentation labels is occurs twice per second and is annotated only for 
    # lidar.TOP)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose,
        keep_polar_features=True)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections,
        range_image_top_pose, ri_index=1, keep_polar_features=True)
    points = [np.concatenate([p1, p2], axis=0) \
                for p1, p2 in zip(points, points_ri2)]
    points = np.concatenate(points, axis=0)
    points_xyz = points[:, 3:]
    points_feature = points[:, 1:3]

    # load segmentation labels
    if frame.lasers[0].ri_return1.segmentation_label_compressed:
        assert frame.lasers[0].ri_return2.segmentation_label_compressed
        point_labels = convert_range_image_to_point_cloud_labels(
            frame, range_images, segmentation_labels)
        point_labels_ri2 = convert_range_image_to_point_cloud_labels(
            frame, range_images, segmentation_labels, ri_index=1)
        point_labels_all = np.concatenate(point_labels, axis=0)
        point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
        point_labels = np.concatenate([point_labels_all, point_labels_all_ri2],
                                      axis=0)
    else:
        point_labels = None

    # load objects
    objects, num_obj = [], 0
    for label in frame.laser_labels:
        box = label.box
        velo = np.array([label.metadata.speed_x, label.metadata.speed_y])
        accel = np.array([label.metadata.accel_x, label.metadata.accel_y])
        obj_class = label.type
        uid = label.id
        num_lidar_points_in_box = label.num_lidar_points_in_box

        #difficulty level is 0 if labeler did not say this was LEVEL_2.
        # Set difficulty level of "999" for boxes with no points in box.
        if num_lidar_points_in_box <= 0:
            combined_difficulty_level = 999
        if label.detection_difficulty_level == 0:
            # Use points in box to compute difficulty level.
            if num_lidar_points_in_box >= 5:
                combined_difficulty_level = 1
            else:
                combined_difficulty_level = 2
        else:
            combined_difficulty_level = label.detection_difficulty_level

        # get rotation to global coordinate system
        veh_to_global = np.array(frame.pose.transform).reshape(4, 4)
        global_rot = np.copy(veh_to_global)
        global_rot[3, :3] = 0

        # compute velocity relative to frame 
        ref_velo = box_utils.transform_point(
                       np.array([[velo[0], velo[1], 0]]),
                       global_rot.T, np.eye(4)).numpy()[0, :2]
        
        assert 0 <= obj_class < 5, "found unidentified object class"
        obj = dict(
            id=num_obj,
            name=uid,
            label=obj_class,
            box=np.array([box.center_x, box.center_y, box.center_z,
                          box.length, box.width, box.height, ref_velo[0],
                          ref_velo[1], box.heading], dtype=np.float32),
            num_points=num_lidar_points_in_box,
            detection_difficulty_level=label.detection_difficulty_level,
            combined_difficulty_level=combined_difficulty_level,
            global_speed=np.array(velo, dtype=np.float32),
            global_accel=np.array(accel, dtype=np.float32),
            class_name=TYPE_LIST[obj_class],
        )
        num_obj+=1
        objects.append(obj)

    frame_name = '{scene_name}_{location}_{time_of_day}_{timestamp}'.format(
        scene_name=frame.context.name,
        location=frame.context.stats.location,
        time_of_day=frame.context.stats.time_of_day,
        timestamp=frame.timestamp_micros)

    lidar3d = dict(
        scene_name=frame.context.name,
        frame_name=frame_name,
        frame_id=frame_id,
        lidars=dict(
            points_xyz=points_xyz,
            points_feature=points_feature,
        ),
    )

    if point_labels is None:
        seg3d = None
    else:
        seg3d = dict(
            scene_name=frame.context.name,
            frame_name=frame_name,
            frame_id=frame_id,
            point_labels=point_labels,
        )

    det3d = dict(
        scene_name=frame.context.name,
        frame_name=frame_name,
        frame_id=frame_id,
        veh_to_global=veh_to_global,
        objects=objects,
    )

    return lidar3d, seg3d, det3d


def convert_sequence(filename):
    global args
    with open(args.tfrecord_names, 'r') as fin:
        lines = fin.readlines()
        name_dict = {line.strip(): i for i, line in enumerate(lines)}
    assert name_dict.get(filename.split('/')[-1], None) is not None
    idx = name_dict[filename.split('/')[-1]]
    if args.dry_run:
        return
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    for frame_id, data in enumerate(dataset):

        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        lidar3d, seg3d, det3d = decode_frame(frame, frame_id)

        for prefix, data in zip(
            ['lidar', 'seg3d', 'annos'], [lidar3d, seg3d, det3d]
        ):
            folder = os.path.join(
                         args.output_path,
                         prefix
                     )
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(
                       args.output_path,
                       prefix,
                       'seq_{}_frame_{}.pkl'.format(idx, frame_id)
                   )
            if data is not None:
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
    if args.delete_tfrecord:
        print('deleting {filename}')
        os.system(f'rm {filename}')

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
                    Run this script to convert .tfrecord file into
                        three type of .pkl files, containing
                        
                        1) lidar point clouds,
                        2) 3D point-wise segmentation labels,
                        3) objects (boxes and classes)

                    example:
                        python tools/waymo_convertor.py <input_path> <output_path>
                    """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('tfrecord_path', help='path to tfrecord files',
                        type=str)
    parser.add_argument('output_path', help='path to store converted files',
                        type=str)
    parser.add_argument('--num_processes', type=int, default=8)
    parser.add_argument('--chunksize', type=int, default=10)
    parser.add_argument('--delete_tfrecord', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    return args

def main():
    global args
    args = parse_args()
    
    tfrecord_files = sorted(glob.glob(f'{args.tfrecord_path}/*.tfrecord'))
    args.tfrecord_names = f'{args.output_path}/tfrecord_names.txt'
    
    print(args)
    if not os.path.exists(args.tfrecord_names):
        with open(args.tfrecord_names, 'w') as fout:
            for tfrecord_file in tfrecord_files:
                fout.write(tfrecord_file+'\n')
    
    num_sequences = len(tfrecord_files)
    with Pool(args.num_processes) as pool: # change according to your cpu
        pool.map(
                convert_sequence,
                tfrecord_files,
                #chunksize=args.chunksize,
            )

if __name__ == '__main__':
    main()
