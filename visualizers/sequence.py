import numpy as np
import pickle
import numpy as np
import glob
from det3d.core import Visualizer
import argparse
import open3d as o3d

from det3d.structures import Sequence as Seq

def parse_args():
    parser = argparse.ArgumentParser('Visualize Point Cloud Sequences')
    parser.add_argument('--root_path', help='root path of data', default='data/Waymo')
    parser.add_argument('--split', help='data split', default='train_50')
    parser.add_argument('--seq_id', help='index of sequence (0 to 797)', default=0)
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    seq = Seq.from_index(args.seq_id, args.root_path, args.split, no_points=True)
    seq.toglobal()
    seq.set_scope(0, 30)
    vis = Visualizer()
    vis.boxes_from_attr('boxes', seq.boxes, seq.classes)

    vis.show()
