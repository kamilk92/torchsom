import os
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch

from torchsom.core.dependency.som_container import SomContainer
from torchsom.core.som.bmu.bmu import Bmu
from torchsom.core.som.dim.map_dimension import MapDimension
from torchsom.core.som.distance.euclidean.euclidean_distance import EuclideanDistance
from torchsom.core.som.learning_rate import LearningRate
from torchsom.core.som.model import SomModel
from torchsom.core.som.radius import Radius
from torchsom.core.som.som import Som
from torchsom.core.som.train_params import TrainParams
from torchsom.core.som.weights.weights_updater import WeightsUpdater


class TestSomContainer(TestCase):
    def test_provide_bmu_with_default_dist(self):
        bmu = SomContainer.get_bmu(torch.tensor(np.zeros((5, 5))))
        self.assertIsInstance(bmu, Bmu)

    def test_provide_bmu_with_custim_dist(self):
        dist = MagicMock()
        bmu = SomContainer.get_bmu(torch.tensor(np.zeros((5, 5))), dist)
        self.assertIsInstance(bmu, Bmu)
        self.assertEqual(bmu._distance, dist)

    def test_provide_eucd_dist(self):
        eucd_dist = SomContainer.get_eucd_dist()
        self.assertIsInstance(eucd_dist, EuclideanDistance)

    def test_provide_learning_rate(self):
        learning_rate = SomContainer.get_learning_rate(.001, torch.tensor(100.0))
        self.assertIsInstance(learning_rate, LearningRate)

    def test_provide_map_dim(self):
        rows = np.random.randint(5, 10)
        cols = np.random.randint(5, 10)
        features = np.random.randint(3, 5)

        map_dim = SomContainer.get_map_dim(rows, cols, features)

        self.assertIsInstance(map_dim, MapDimension)
        self.assertEqual(map_dim.rows, rows)
        self.assertEqual(map_dim.cols, cols)
        self.assertEqual(map_dim.features, features)

    def test_provide_neighbourhood_radius(self):
        map_dim = MapDimension(10, 15, 3)
        total_iter_cnt = torch.tensor(100, dtype=torch.float64)

        radius = SomContainer.get_neighbourhood_radius(map_dim, total_iter_cnt)

        self.assertIsInstance(radius, Radius)

    def test_get_rgb_som(self):
        raise NotImplementedError

    def test_get_som(self):
        map_dim = MapDimension(10, 10, 3)
        board_dir = os.path.join(os.path.dirname(__file__), "board")
        train_params = TrainParams(50, 1, 10, 10, board_dir)
        som = SomContainer.get_som(map_dim, .001, 100.0, (-1, 1), board_dir, train_params)

        self.assertIsInstance(som, Som)

    def test_get_som_model(self):
        dist = EuclideanDistance()
        map_indices = torch.tensor(np.asarray((10, 10, 2)))
        bmu = Bmu(dist, map_indices)
        total_iter_cnt = torch.tensor(100, dtype=torch.float64)
        learning_rate = LearningRate(.001, total_iter_cnt)
        map_dim = MapDimension(10, 10, 3)
        radius = Radius(map_dim, total_iter_cnt)
        weights_updater = WeightsUpdater(map_indices)
        w_range = (-1.0, 1.0)

        som_model = SomContainer.get_som_model(bmu, learning_rate, map_dim, map_indices, radius, weights_updater,
                                               w_range)

        self.assertIsInstance(som_model, SomModel)

    def test_get_weights_update_provider(self):
        map_indices = torch.tensor(np.zeros((10, 10, 2)))

        weights_updater = SomContainer.get_weights_updater(map_indices)

        self.assertIsInstance(weights_updater, WeightsUpdater)
