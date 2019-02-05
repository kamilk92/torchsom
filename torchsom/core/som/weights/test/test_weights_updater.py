from unittest import TestCase

import numpy as np

import torch

# from torchsom.core.som.weights.weights_updater import WeightsUpdater
from torchsom.core.som.weights.weights_updater import WeightsUpdater


class TestWeightsUpdater(TestCase):
    MAP_SIZE = 5
    FEATURES_CNT = 3

    @classmethod
    def setUpClass(cls):
        cls.__map_indices = cls.__setup_map_indices()
        cls.__weights = cls.__setup_weights()
        cls.__weights_updater = WeightsUpdater(cls.__map_indices)

    def test_update_weights(self):
        bmu_indices = torch.tensor(np.asarray([2, 2]))
        neighbourhood_radius = torch.tensor(2)
        v_in = torch.tensor(np.ones(self.FEATURES_CNT, dtype=np.float64))
        learning_rate = torch.tensor(2, dtype=torch.float64)
        updated_weights = self.__weights_updater.update(v_in, bmu_indices, self.__weights, neighbourhood_radius,
                                                        learning_rate)
        self.assertEqual(updated_weights.shape, self.__weights.shape)

    def test_update_batch_weights(self):
        bmu_indices = torch.tensor(np.asarray([[2, 2], [1, 3]]))
        neighbourhood_radius = torch.tensor(2)
        v_in = torch.tensor(np.asarray([[2, 0, 1], [1, 1, 1]], dtype=np.float64))
        learning_rate = torch.tensor(2, dtype=torch.float64)
        updated_weights = self.__weights_updater.update_batch(v_in, bmu_indices, self.__weights, neighbourhood_radius,
                                                        learning_rate)
        self.assertEqual(updated_weights.shape, self.__weights.shape)

    @classmethod
    def __setup_map_indices(cls):
        return torch.tensor(np.asarray([[(i, j) for j in range(cls.MAP_SIZE)] for i in range(cls.MAP_SIZE)]))

    @classmethod
    def __setup_weights(cls):
        return torch.tensor(np.full((cls.MAP_SIZE, cls.MAP_SIZE, cls.FEATURES_CNT), 10).astype(np.float64))
