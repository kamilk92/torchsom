from typing import Tuple

import numpy as np
import torch

from torchsom.core.som.bmu.bmu import Bmu
from torchsom.core.som.dim.map_dimension import MapDimension
from torchsom.core.som.learning_rate import LearningRate
from torchsom.core.som.radius import Radius
from torchsom.core.som.weights.weights_updater import WeightsUpdater


class SomModel:
    def __init__(self, bmu: Bmu, learning_rate: LearningRate, map_dim: MapDimension, map_indices: torch.Tensor,
                 neighbourhood_radius: Radius, weights_updater: WeightsUpdater, w_range: Tuple[float, float] = None):
        self.__bmu = bmu
        self.__learning_rate = learning_rate
        self.__map_dim = map_dim
        self.__map_indices = map_indices
        self.__neighbourhood_radius = neighbourhood_radius
        self.__weights_updater = weights_updater
        self.__weights: torch.Tensor = self.__init_weights(w_range or (-1, 1))

    @property
    def learning_rate(self) -> torch.Tensor:
        return self.__learning_rate.learning_rate

    @property
    def map_dim(self) -> MapDimension:
        return self.__map_dim

    @property
    def map_indices(self) -> torch.Tensor:
        return self.__map_indices

    @property
    def neighbourhood_radius(self) -> torch.Tensor:
        return self.__neighbourhood_radius.neighbourhood_radius

    @property
    def weights(self) -> np.ndarray:
        return self.__weights.numpy().copy()

    def train(self, batch: torch.Tensor, iter_num: torch.Tensor) -> torch.Tensor:
        bmu_indices_and_dist = self.__bmu.find(self.__weights, batch)
        bmu_indices, distances = self.__collect_bmu_indices_and_dist(bmu_indices_and_dist)
        radius = self.__neighbourhood_radius.radius(iter_num)
        learning_rate = self.__learning_rate.calculate(iter_num)
        self.__weights = self.__weights_updater.update_batch(batch, bmu_indices, self.__weights, radius, learning_rate)

        return self.__weights

    def __collect_bmu_indices_and_dist(self, bmu_dist_and_indices):
        bmu_indices = list()
        distances = list()
        for i in range(len(bmu_dist_and_indices)):
            bmu_indices.append(bmu_dist_and_indices[i][1])
            distances.append(bmu_dist_and_indices[i][0])

        return bmu_indices, distances

    def __init_weights(self, w_range: Tuple[float, float]) -> torch.Tensor:
        w_dim = (int(self.__map_dim.rows), int(self.__map_dim.cols), int(self.__map_dim.features))

        return torch.tensor(np.random.uniform(w_range[0], w_range[1], w_dim))
