from abc import ABCMeta, abstractmethod
from typing import List

import torch
from torch import Tensor


class Distance(metaclass=ABCMeta):
    @abstractmethod
    def distance(self, input_batch: Tensor, w: Tensor) -> List[Tensor]:
        pass
