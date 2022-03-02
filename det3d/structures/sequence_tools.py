import numpy as np
import torch
import scipy
import pickle

from torch_scatter import scatter
from torch.nn import functional as F
from torch_cluster import radius_graph
from scipy.sparse import csr_matrix
import fire, os
from multiprocessing import Pool
from tqdm import tqdm

from det3d.structures.sequence import Sequence
from det3d.core import Visualizer

def get_grid_laplacian(dims):
    """Get a laplacian of a 2d grid graph
    
    Args:
        dims (2): number of grids in x, y dimension

    Returns:
        L (scipy.sparse.csr_matrix): sparse Laplacian

    """
    num_grids = dims[0] * dims[1]
    grids = np.meshgrid(
                np.arange(dims[0]),
                np.arange(dims[1]),
                indexing="ij")
    grids = np.stack(grids, axis=-1).reshape(-1, 2)
    grids = torch.tensor(grids)
    e0, e1 = radius_graph(grids, r=1.5, loop=False)
    grids_1d = grids[:, 0]*dims[1] + grids[:, 1]
    deg = scatter(torch.ones_like(e0), e0, dim=0,
                  dim_size=num_grids, reduce='add')
    
    data, row, col = [], [], []
    # edges
    data.append(-np.ones_like(e0))
    row.append(e0.numpy())
    col.append(e1.numpy())

    # diagonal
    data.append(deg.numpy())
    row.append(np.arange(num_grids))
    col.append(np.arange(num_grids))
    
    # merge
    data = np.concatenate(data)
    row = np.concatenate(row)
    col = np.concatenate(col)
    
    L = csr_matrix((data, (row, col)), shape=(num_grids, num_grids))
    
    return L

def optimize_ground_plane(
        z_min,
        z_mask,
        b_min,
        b_mask,
        lamb=100,
    ):
    z_mask = z_mask.astype(np.float64)
    b_mask = b_mask.astype(np.float64) * 1000.0

    size_x, size_y = z_min.shape
    num_grids = size_x * size_y
    L = get_grid_laplacian([size_x, size_y])
    Wz = scipy.sparse.dia_matrix((z_mask.reshape(-1), 0), shape=(num_grids, num_grids))
    Wb = scipy.sparse.dia_matrix((b_mask.reshape(-1), 0), shape=(num_grids, num_grids))
    plane = scipy.sparse.linalg.spsolve(
                Wz + Wb + lamb * L,
                (z_mask*z_min).reshape(-1) + (b_mask*b_min).reshape(-1)
            )

    plane = plane.reshape((size_x, size_y))

    return plane

class GroundPlaneEstimator(object):
    def __init__(self, cfg=None):
        """
        Args:
            pc_range (list, [6]): [(x, y, z)_min, (x, y, z)_max]
            voxel_size (list, [3]): [vx, vy, vz]

        """
        self.size_factor = np.array(cfg.get("size_factor", 1))
        self.voxel_size = np.array(cfg.get("voxel_size", [0.1, 0.1, 0.15]))
        split = cfg.get("split", "train")
        self.save_path = f'data/Waymo/{split}/ground_plane'
        os.makedirs(self.save_path, exist_ok=True)

    def __call__(self, seq):
        seq.toglobal()

        points = seq.points4d[:, :3]
        box_bottoms = seq.corners[:, [0, 3, 4, 7], :].reshape(-1, 3)
        pc_range = np.concatenate([points.min(0)[:3], points.max(0)[:3]], axis=0)
        grid_size = np.floor(
                        np.divide(
                            pc_range[-3:-1] - pc_range[:2],
                            self.voxel_size[:2] * self.size_factor,
                        )
                    ).astype(np.int32) + 1

        # box bottoms
        bx, by = np.floor(
                     np.divide(
                         box_bottoms[:, :2] - pc_range[:2],
                         self.voxel_size[:2] * self.size_factor
                     )
                 ).astype(np.int32).T
        bvoxels = torch.tensor(bx * grid_size[1] + by, dtype=torch.int64)
        box_bottoms = torch.tensor(box_bottoms, dtype=torch.float64)

        b_min = scatter(box_bottoms[:, 2], bvoxels, reduce='min', dim=0,
                        dim_size=np.prod(grid_size))
        b_mask = scatter(torch.ones(box_bottoms.shape[0], dtype=torch.long),
                         bvoxels, reduce='max', dim=0,
                         dim_size=np.prod(grid_size))

        b_min = b_min.numpy().reshape(grid_size)
        b_mask = b_mask.numpy().reshape(grid_size)

        # points
        vx, vy = np.floor(
                     np.divide(
                         points[:, :2] - pc_range[:2],
                         self.voxel_size[:2] * self.size_factor
                     )
                 ).astype(np.int32).T
        voxels = torch.tensor(vx * grid_size[1] + vy, dtype=torch.int64)
        points = torch.tensor(points, dtype=torch.float64)
        
        z_min = scatter(points[:, 2], voxels, reduce='min', dim=0,
                        dim_size=np.prod(grid_size))
        z_mask = scatter(torch.ones(points.shape[0], dtype=torch.long),
                         voxels, reduce='max', dim=0,
                         dim_size=np.prod(grid_size))

        z_min = z_min.numpy().reshape(grid_size)
        z_mask = z_mask.numpy().reshape(grid_size)

        # optimize for ground plane
        z = optimize_ground_plane(z_min, z_mask, b_min, b_mask)

        # decode 3D coordinates
        x, y = np.meshgrid(np.arange(grid_size[0]),
                           np.arange(grid_size[1]),
                           indexing="ij")
        x = x * self.size_factor * self.voxel_size[0] + pc_range[0]
        y = y * self.size_factor * self.voxel_size[1] + pc_range[1]
        
        # ground plane points in 3D coordinate
        ground_plane = np.stack([x, y, z], axis=-1)
        
        return ground_plane

def precompute_ground_plane(split, seq_id):
    save_path = f'data/Waymo/{split}/ground_plane'
    filename = f'{save_path}/seq_{seq_id}.pkl'
    if os.path.exists(filename):
        return
    estimator = GroundPlaneEstimator(
                    dict(
                        pc_range = [-75.2, -75.2, -2, 75.2, 75.2, 4],
                        size_factor=8,
                        voxel_size=[0.1, 0.1, 0.15],
                        split=split,
                    )
                )
    seq = Sequence.from_index(seq_id, split=split)
    ground_plane = estimator(seq)

    gp_dict = dict(
        ground_plane=ground_plane,
    )
    with open(f'{save_path}/seq_{seq_id}.pkl', 'wb') as fout:
        pickle.dump(gp_dict, fout)

def precompute_ground_plane_batch(split, start_seq_id, end_seq_id, num_processes=40):
    args = [(split, seq_id) for seq_id in range(start_seq_id, end_seq_id)]
    num_seqs = len(args)
    with Pool(num_processes) as p: # change according to your cpu
        r = list(tqdm(p.starmap(precompute_ground_plane, args), total=num_seqs))

if __name__ == '__main__':
    fire.Fire()
