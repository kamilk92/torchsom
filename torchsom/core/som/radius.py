import torch
from torch import Tensor

from torchsom.core.som.dim.map_dimension import MapDimension


class Radius:
    def __init__(self, map_dim: MapDimension, total_iter_cnt: Tensor):
        self.__map_dim = map_dim
        self.__map_radius = self.__calc_map_radius()
        self.__neighbourhood_radius = None
        self.__time_const = self.__calc_time_const(total_iter_cnt)

    @property
    def neighbourhood_radius(self) -> torch.Tensor:
        return self.__neighbourhood_radius

    def radius(self, current_iter_num: Tensor) -> torch.Tensor:
        self.__neighbourhood_radius = self.__map_radius * torch.exp(-current_iter_num / self.__time_const)

        return self.__neighbourhood_radius

    def __calc_map_radius(self) -> torch.Tensor:
        return torch.tensor(max([self.__map_dim.rows, self.__map_dim.cols]) / 2, dtype=torch.float64)

    def __calc_time_const(self, total_iter_cnt: Tensor):
        log_rad = torch.log(self.__map_radius)
        time_const = total_iter_cnt / log_rad

        return time_const
