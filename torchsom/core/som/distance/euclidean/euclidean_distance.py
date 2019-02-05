from typing import List

import torch
from torch import Tensor

from torchsom.core.som.distance.distance import Distance


class EuclideanDistance(Distance):
    def distance(self, input_batch: Tensor, w: Tensor) -> List[Tensor]:
        return [self.__input_vector_distance(w, v) for v in input_batch]

    def __input_vector_distance(self, w: torch.Tensor, v: torch.Tensor):
        return torch.sqrt(torch.sum(torch.pow(w - v, 2), dim=2))
