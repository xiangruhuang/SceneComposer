import numpy as np
from det3d.core import Visualizer

vis = Visualizer(ground_plane=True)

vis.pointcloud('origin', np.zeros((1, 3)), radius=1e-1)

box = np.array([[10, 8, 0, 4, 8, 2, 0]], dtype=np.float32)
theta = np.arctan2(-8, 10) - np.arctan2(10, 8)
print(theta)
label = np.array([0])

vis.boxes_from_attr('box', box, label, radius=2e-2, color=(1,0,0))

rot_box = np.array([[-8, 10, 0, 4, 8, 2, theta]], dtype=np.float32)
vis.boxes_from_attr('rot-box', rot_box, label, radius=2e-2, color=(0,1,0))
vis.show()
