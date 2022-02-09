import argparse
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser('Extract Object Traces from database of objects')
    parser.add_argument('dbinfo', help='database info file (.pkl)')
    parser.add_argument('--result_path', help='folder to save results under')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()

    with open(args.dbinfo, 'rb') as fin:
        dbinfos = pickle.load(fin)

    for class_name in dbinfos.keys():
        # VEHICLE, PEDESTRIAN, CYCLIST
        dbinfos_cls = dbinfos[class_name]
        
        traces = {}
        for info in dbinfos_cls:
            uid = info['unique_id']
            points = np.fromfile('data/Waymo/'+info['path'],
                                 dtype=np.float32).reshape(-1, 5)
            box = info['box3d_lidar']
            theta = box[-1]
            R = np.array([np.cos(theta), -np.sin(theta),
                          np.sin(theta), np.cos(theta)],
                         dtype=np.float32).reshape(2, 2)

            points[:, :2] = points[:, :2] @ R.T
            origin = -box[:3]
            origin[:2] = origin[:2] @ R.T
            
            if traces.get(uid, None) is None:
                traces[uid] = dict(points=[], origins=[])
            traces[uid]['points'].append(points)
            traces[uid]['origins'].append(origin)
        with open(f'{args.result_path}/{class_name}.pkl', 'wb') as fout:
            pickle.dump(traces, fout)

