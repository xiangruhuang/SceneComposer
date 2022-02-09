import pickle
import numpy as np
import glob
from det3d.core import Visualizer
import argparse
import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser('Visualize Object Traces')
    parser.add_argument('info_path', help='object traces info file (.pkl)')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()

    class_info_files = glob.glob(f'{args.info_path}/VEHICLE.pkl')
    vis = Visualizer()
    for class_info_file in class_info_files:
        with open(class_info_file, 'rb') as fin:
            infos = pickle.load(fin)
        for key, info in infos.items():
            points, origins = info['points'], info['origins']
            for i, (point, origin) in enumerate(zip(points, origins)):
                vis.pointcloud(f'frame-{i}', point[:, :3], radius=4e-4, enabled=False)
                vis.pointcloud(f'origin-{i}', origin[np.newaxis, :3], radius=3e-2, enabled=False)

            points = np.concatenate(info['points'], axis=0)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=100)
            )
            normals = np.array(pcd.normals)
            
            origins = np.stack(info['origins'], axis=0)
            vis.pointcloud('points', points[:, :3], radius=4e-5)
            ps_sub = vis.pointcloud('sub-points', points[::1000, :3], radius=4e-5)
            ps_sub.add_vector_quantity('sub-normals', normals[::1000])
            vis.pointcloud('origins', origins[:, :3], radius=3e-3)
            vis.show()
