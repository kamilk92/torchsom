from abc import ABCMeta, abstractclassmethod, abstractmethod

from torch.utils.data import Dataset


class DatasetFactory(metaclass=ABCMeta):
    @abstractmethod
    def create_dataset(self) -> Dataset:
        pass
