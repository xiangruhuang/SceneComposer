import argparse
from .sequence import Sequence
from .sequence_tools import *

estimator = GroundPlaneEstimator(
                dict(
                    pc_range = [-75.2, -75.2, -2, 75.2, 75.2, 4],
                    size_factor=8,
                    voxel_size=[0.1, 0.1, 0.15],
                )
            )

seq = Sequence.from_index(0, split='train_50')

ground_plane = estimator(seq)
