import argparse
from det3d.core import Visualizer
from det3d.core.bbox import box_np_ops
import pickle, numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', type=str)
    args = parser.parse_args()

    return args

def rotate_box_and_points(box, points, target_angles):
    points[:, :3] += box[0, :3]
    
    

if __name__ == '__main__':
    args = parse_args()

    vis = Visualizer()
    with open('data/Waymo/dbinfos_train_50_1sweeps_withvelo.pkl', 'rb') as fin:
        data = pickle.load(fin)
        clusters = data[args.cls]
        for i, cluster in enumerate(clusters):
            path = 'data/Waymo/'+cluster['path']
            box = cluster['box3d_lidar']
            box = box.reshape(-1, 9)
            corners = box_np_ops.center_to_corner_box3d(box[:, :3],
                                                        box[:, 3:6],
                                                        box[:, -1],
                                                        axis=2)
            labels = np.zeros(1).astype(np.int32)+0
            vis.boxes('box', corners, labels, radius=2e-2)
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
            if points.shape[0] < 1000:
                continue
            print(cluster)
            points[:, :3] += corners.mean(1)[0]
            print(corners.mean(1)[0], points.mean(0))
            vis.pointcloud('points', points[:, :3], radius=2e-2)
            vis.look_at(points[:, :3].mean(0)/2, 100)
            filename = f'{args.cls}_{i:03d}.png'
            origin = np.zeros((100, 3))
            vis.pointcloud('zero', origin, radius=2e-1)
            angle = -np.pi/2 - box[0, -1] 
            vec = np.array([np.cos(angle), np.sin(angle), 0])
            vec = np.linspace(0, 10, 100).reshape(-1, 1)*vec
            vis.pointcloud('vec', vec+corners.mean(1)[0], radius=3e-2)


            vis.screenshot(filename)
            os.system(f'mv {filename} ~/public_html/figures/')

            if i > 10:
                break





        



    
