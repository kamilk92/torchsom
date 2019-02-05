import torch
from torch import Tensor
from typing import List, Tuple

from torchsom.core.som.distance.distance import Distance


class Bmu:
    def __init__(self, distance: Distance, map_indices: torch.Tensor):
        self._distance = distance
        self._map_indices = map_indices

    def find(self, w: Tensor, input_batch: Tensor) -> List[Tuple[Tensor, Tensor]]:
        distances = self._distance.distance(input_batch, w)
        dist_and_bmu_indices = self.__find_bmu_indices(distances)

        return dist_and_bmu_indices

    def __find_bmu_indices(self, distances: List[Tensor]) -> List[Tuple[Tensor, Tensor]]:
        return [(dist, self.__get_bmu_indices(dist)) for dist in distances]

    def __get_bmu_indices(self, dist: Tensor) -> Tensor:
        bmu_indices = self._map_indices[dist == dist.min()][-1]

        return bmu_indices
