import numpy as np

from unittest import TestCase

import torch

from torchsom.core.som.bmu.bmu import Bmu
from torchsom.core.som.dim.map_dimension import MapDimension
from torchsom.core.som.distance.euclidean.euclidean_distance import EuclideanDistance


class TestBmu(TestCase):
    W = np.asarray([
        [
            [1, 2, 2], [5, 7, 8]
        ],
        [
            [0, 1, 0], [1, 2, 2]
        ],
        [
            [1, 2, 8], [3, 15, 5]
        ]
    ]).astype(np.float64)

    V = np.asarray([
        [1, 2, 1],
        [3, 12, 4]
    ]).astype(np.float64)

    @classmethod
    def setUpClass(cls):
        cls.__w_tensor = torch.tensor(cls.W)
        cls.__v_tensor = torch.tensor(cls.V)
        w_shape = cls.W.shape
        cls.__bmu = Bmu(EuclideanDistance(), MapDimension(w_shape[0], w_shape[1]))

    def test_find_bmu(self):
        bmus = self.__bmu.find(self.__w_tensor, self.__v_tensor)
        self.assertEqual(len(bmus), len(self.V))

        dist = bmus[0][0]
        expected_dist_shape = (self.W.shape[0], self.W.shape[1])
        self.assertEqual(dist.shape, expected_dist_shape)
        bmu_indices = bmus[0][1]
        self.assertTrue(np.array_equal(bmu_indices.numpy(), np.asarray([1, 1])))

        dist = bmus[1][0]
        self.assertEqual(dist.shape, expected_dist_shape)
        bmu_indices = bmus[1][1]
        self.assertTrue(np.array_equal(bmu_indices.numpy(), np.asarray([2, 1])))
