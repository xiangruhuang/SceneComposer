import numpy as np
import pickle
import numpy as np
import glob
from det3d.core import Visualizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Visualize PC Segmentation Labels')
    parser.add_argument('info_path', type=str)
    args = parser.parse_args()
    
    return args

def get_obj(path):
    with open(path, "rb") as fin:
        return pickle.load(fin)
    
TYPE_LIST = [
    "UNDEFINED",
    "CAR",
    "TRUCK",
    "BUS",
    # Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction
    # vehicles, RV, limo, tram).
    "OTHER_VEHICLE",
    "MOTORCYCLIST",
    "BICYCLIST",
    "PEDESTRIAN",
    "SIGN",
    "TRAFFIC_LIGHT",
    # Lamp post, traffic sign pole etc.
    "POLE",
    # Construction cone/pole.
    "CONSTRUCTION_POLE",
    "BICYCLE",
    "MOTORCYCLE",
    "BUILDING",
    # Bushes, tree branches, tall grasses, flowers etc.
    "VEGETATION",
    "TREE_TRUNK",
    # Curb on the edge of roads. This does not include road boundaries if
    # there’s no curb.
    "CURB",
    # Surface a vehicle could drive on. This include the driveway connecting
    # parking lot and road over a section of sidewalk.
    "ROAD",
    # Marking on the road that’s specifically for defining lanes such as
    # single/double white/yellow lines.
    "LANE_MARKER",
    # Marking on the road other than lane markers, bumps, cateyes, railtracks
    # etc.
    "OTHER_GROUND",
    # Most horizontal surface that’s not drivable, e.g. grassy hill,
    # pedestrian walkway stairs etc.
    "WALKABLE",
    # Nicely paved walkable surface when pedestrians most likely to walk on.
    "SIDEWALK",
]

def plot(infos, labels):
    colors = [c for l, c in labels]
    labels = [l for l, c in labels]
    label_str= str(labels)
    indices = [TYPE_LIST.index(label) for label in labels]
    vis = Visualizer()

    for info in infos:
        points = get_obj(info['path'])["lidars"]["points_xyz"]
        seg_labels = get_obj(info['seg_path'])["point_labels"]
        instance, seg_labels = seg_labels.T
        points = points[:instance.shape[0]]
        valid_indices = np.where((instance != 0) | (seg_labels != 0))[0]
        instance, seg_labels = instance[valid_indices], seg_labels[valid_indices]
        masks = [seg_labels == idx for idx in indices]
        mask_all = masks[0]
        for m in masks[1:]:
            mask_all = mask_all | m
        if mask_all.any():
            vis.pointcloud("points-all", points, color=(75, 75, 75))
            ps_l = vis.pointcloud("labeled-points", points[valid_indices], color=(75, 75, 75))

            ps_l.add_scalar_quantity('instance', instance)
            ps_l.add_scalar_quantity('seg', seg_labels)
            #vis.pointcloud(f'{label_str}', points[valid_indices[mask_all]], color=(1,0,0), radius=2e)
            for i, mask in enumerate(masks):
                vis.pointcloud(f'{labels[i]}', points[valid_indices[mask]], radius=2.5e-4, color=(colors[i]))
            vis.show()

if __name__ == '__main__':
    args = parse_args()


    infos = get_obj(args.info_path) 
    infos = [info for info in infos if info['seg_path'] is not None]

    labels = [
        # Curb on the edge of roads. This does not include road boundaries if
        # there’s no curb.
        ("CURB", (1,0,0)),
        # Surface a vehicle could drive on. This include the driveway connecting
        # parking lot and road over a section of sidewalk.
        ("ROAD", (0,0,1)),
        # Marking on the road that’s specifically for defining lanes such as
        # single/double white/yellow lines.
        ("LANE_MARKER", (0,1,1)),
        # Marking on the road other than lane markers, bumps, cateyes, railtracks
        # etc.
        ("OTHER_GROUND", (0,0,1)),
        # Most horizontal surface that’s not drivable, e.g. grassy hill,
        # pedestrian walkway stairs etc.
        ("WALKABLE", (0,1,0)),
        # Nicely paved walkable surface when pedestrians most likely to walk on.
        ("SIDEWALK", (0,1,0)),
        ("BUILDING", (128,22,0)),
        ("PEDESTRIAN", (1,0,0)),
    ]

    plot(infos, labels)
