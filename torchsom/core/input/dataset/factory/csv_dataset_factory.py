from typing import List

from torch.utils.data import Dataset

from torchsom.core.input.dataset.csv_dataset import CsvDataset
from torchsom.core.input.dataset.factory.datset_factory import DatasetFactory


class CsvDatasetFactory(DatasetFactory):
    def __init__(self, dataset_files: List[str], shuffle_files=False):
        self._dataset_files = dataset_files
        self._shuffle_files = shuffle_files

    def create_dataset(self) -> Dataset:
        return CsvDataset(self._dataset_files, self._shuffle_files)
