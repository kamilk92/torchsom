import torch


class LearningRate:
    def __init__(self, learning_rate: float, total_iter_cnt: torch.Tensor):
        self.__initial_learning_rate = torch.tensor(learning_rate, dtype=torch.float64)
        self.__learning_rate = self.__initial_learning_rate
        self.__total_iter_cnt = total_iter_cnt

    def calculate(self, iter_num: torch.Tensor) -> torch.Tensor:
        self.__learning_rate = self.__initial_learning_rate * torch.exp(-iter_num / self.__total_iter_cnt)

        return self.__learning_rate

    @property
    def learning_rate(self) -> torch.Tensor:
        return self.__learning_rate
