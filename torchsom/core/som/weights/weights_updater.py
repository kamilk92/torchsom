import logging

import torch


class WeightsUpdater:
    def __init__(self, map_indices: torch.Tensor):
        self._logger = logging.getLogger(__name__)
        self.__map_indices = map_indices

    def update_batch(self, input_batch, batch_bmu_indices: torch.Tensor, weights,
                     neighbourhood_radius: torch.Tensor, learning_rate: torch.Tensor) -> torch.Tensor:
        for i in range(len(batch_bmu_indices)):
            weights = self.update(input_batch[i], batch_bmu_indices[i], weights, neighbourhood_radius, learning_rate)

        return weights

    def update(self, x: torch.Tensor, bmu_indices: torch.Tensor, weights,
               neighbourhood_r: torch.Tensor, learning_rate: torch.Tensor) -> torch.Tensor:
        dist_to_bmu = self.__dist_to_bmu(bmu_indices)
        influence = self.__influence(dist_to_bmu, neighbourhood_r)
        w_to_update_indices = self.__find_indices_weights_to_update(dist_to_bmu, neighbourhood_r, x.shape[0])
        new_weights = weights + learning_rate * influence * (-weights + x)
        new_weights = torch.where(w_to_update_indices, new_weights, weights)

        return new_weights

    def __dist_to_bmu(self, bmu_indices: torch.Tensor) -> torch.Tensor:
        return torch.sum((self.__map_indices - bmu_indices) ** 2, dim=2)

    def __find_indices_weights_to_update(self, dist_to_bmu: torch.Tensor, neighbourhood_r: torch.Tensor, features_cnt):
        w_to_update: torch.Tensor = dist_to_bmu < (neighbourhood_r ** 2)
        w_to_update = w_to_update.reshape((w_to_update.shape[0], w_to_update.shape[1], 1)).repeat(1, 1, features_cnt)

        return w_to_update

    def __influence(self, dist_to_bmu: torch.Tensor, neighbourhood_radius: torch.Tensor):
        mshape = self.__map_indices.shape
        influence = torch.exp(-dist_to_bmu / (neighbourhood_radius ** 2)).reshape((mshape[0], mshape[1], 1))

        return influence

    def __is_bmu_neighbourhood(self, dist_to_bmu: torch.Tensor, neighbourhood_radius: torch.Tensor) -> bool:
        return dist_to_bmu < (neighbourhood_radius ** 2)
