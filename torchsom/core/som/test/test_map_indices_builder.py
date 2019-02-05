from unittest import TestCase

import numpy as np
import torch

from torchsom.core.som.dim.map_dimension import MapDimension
from torchsom.core.som.map_indices_builder import MapIndicesBuilder


class TestMapIndicesBuilder(TestCase):
    def test_build_indices(self):
        map_dim = MapDimension(5, 10, 2)
        map_indices = MapIndicesBuilder()

        indices = map_indices.build(map_dim)

        self.assertIsInstance(indices, torch.Tensor)
        self.assertEqual(indices.shape, (map_dim.rows, map_dim.cols, map_dim.features))
        for row_num in range(map_dim.rows):
            for col_num in range(map_dim.cols):
                expected_value = np.asarray([row_num, col_num])
                np.testing.assert_array_equal(indices[row_num, col_num].numpy(), expected_value)
