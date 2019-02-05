from typing import List

import numpy as np

from unittest import TestCase

import torch
from torch import Tensor

from torchsom.core.som.distance.euclidean.euclidean_distance import EuclideanDistance


class TestEuclideanDistance(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eucd_dist = EuclideanDistance()

    def test_claculate_euclidean_dist(self):
        w_array = np.random.normal(10, 5, (5, 5, 3)).astype(np.float64)
        v_array = np.random.normal(10, 5, (3, 3)).astype(np.float64)
        w = torch.tensor(w_array)
        v = torch.tensor(v_array)

        dist = self.eucd_dist.distance(v, w)

        self.assertEqual(len(dist), len(v))
        expected_dist = np.stack([self.__calculate_eucd_dist(w_array, v_vector) for v_vector in v_array])
        dist_array = self.__dist_to_numpy(dist)
        self.assertTrue(np.array_equal(dist_array, expected_dist))

    def __calculate_eucd_dist(self, w: np.ndarray, v: np.ndarray):
        return np.sqrt(np.sum(np.square(np.subtract(w, v)), axis=2))

    def __dist_to_numpy(self, dist: List[Tensor]):
        return np.stack(dist)
