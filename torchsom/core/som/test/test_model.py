from unittest import TestCase

import numpy as np
import torch

from torchsom.core.som.bmu.bmu import Bmu
from torchsom.core.som.dim.map_dimension import MapDimension
from torchsom.core.som.distance.euclidean.euclidean_distance import EuclideanDistance
from torchsom.core.som.learning_rate import LearningRate
from torchsom.core.som.map_indices_builder import MapIndicesBuilder
from torchsom.core.som.model import SomModel
from torchsom.core.som.radius import Radius
from torchsom.core.som.weights.weights_updater import WeightsUpdater


class TestModel(TestCase):
    MAP_DIM = 10
    FEATS_CNT = 3

    @classmethod
    def setUpClass(cls):
        cls.__map_dim = MapDimension(cls.MAP_DIM, cls.MAP_DIM, cls.FEATS_CNT)
        cls.__map_indices = MapIndicesBuilder().build(cls.__map_dim)
        cls.__bmu = Bmu(EuclideanDistance(), cls.__map_indices)
        cls.__total_iter_cnt = torch.tensor(100, dtype=torch.float64)
        cls.__learning_rate = LearningRate(0.01, cls.__total_iter_cnt)
        cls.__radius = Radius(cls.__map_dim, cls.__total_iter_cnt)
        cls.__weights_updater = WeightsUpdater(cls.__map_indices)

    def test_update_batch(self):
        som = SomModel(self.__bmu, self.__learning_rate, self.__map_dim, self.__map_indices, self.__radius,
                       self.__weights_updater)
        batch = torch.tensor(np.asarray([[0.1, 0., 0.2], [0., 0., 0.5]]))
        current_iter = torch.tensor(float(1), dtype=torch.float64)

        weights = som.train(batch, current_iter)

        expected_w_shape = np.asarray([self.__map_dim.rows, self.__map_dim.cols, self.__map_dim.features])
        np.testing.assert_array_equal(weights.shape, expected_w_shape)
