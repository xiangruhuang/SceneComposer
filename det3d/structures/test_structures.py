from .frame import Frame
from .sequence import Sequence
from det3d.core import Visualizer
from det3d.core.bbox import box_np_ops
import numpy as np

frame = Frame.from_index(0, 0)
frame = Frame.from_index(1, 0)
frame = Frame.from_index(2, 0)
print(frame.boxes)

seqs = []
for seq_id in range(10):
    seq = Sequence.from_index(seq_id, split='train_50', dtype=np.float64, no_points=True)
    seqs.append(seq)

seq = seqs[1]
seq.toframe(0)

vis = Visualizer()
vis.pointcloud('points', seq.points4d[:, :3])
corners = box_np_ops.center_to_corner_box3d(
              seq.boxes[:, :3],
              seq.boxes[:, 3:6],
              seq.boxes[:, -1],
          )
vis.boxes('boxes', corners, seq.classes)
vis.boxes('c-boxes', seq.corners, seq.classes)

vis.show()
