import numpy as np
import torch

from torchsom.core.som.dim.map_dimension import MapDimension


class MapIndicesBuilder:
    def build(self, map_dim: MapDimension):
        indices = np.asarray([[(i, j) for j in range(int(map_dim.cols))] for i in range(int(map_dim.rows))]) \
            .astype(np.float64)

        return torch.tensor(indices)
