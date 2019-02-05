import random

import logging
import pandas as pd
from typing import List

from torch.utils.data import Dataset

from torchsom.core.exception import NoMoreDataException


class CsvDataset(Dataset):
    def __init__(self, dataset_files: List[str], shuffle_files=False):
        self._logger = logging.getLogger(__name__)
        self.__dataset_files = list(dataset_files) if not shuffle_files else self.__shuffle_files(dataset_files)
        self.__dataset_len = self.__calc_dataset_len()
        self.__processed_file_idx = -1
        self.__total_files_rows_processed = 0
        self.__files_processed = -1
        self.__current_df: pd.DataFrame = None

    def __len__(self):
        return self.__dataset_len

    def __getitem__(self, index):
        file_index = self.__file_index(index)
        if (self.__current_df is None) or (file_index > len(self.__current_df) - 1):
            self.__next_file()
            file_index = 0
        if self.__current_df is None:
            raise NoMoreDataException(
                "No more data in dataset. Processed files count : %d" % self.__files_processed)

        return self.__current_df.iloc[file_index].values

    def __file_index(self, index: int) -> int:
        if self.__files_processed == 0:
            return index

        return index - self.__total_files_rows_processed

    def __calc_dataset_len(self) -> int:
        self._logger.debug("Calculating dataset length.")
        dataset_len = 0
        for file in self.__dataset_files:
            dataset_len += len(pd.read_csv(file))
        self._logger.info("Dataset length: %d", dataset_len)

        return dataset_len

    def __next_file(self) -> pd.DataFrame:
        self.__files_processed += 1
        if self.__files_processed >= len(self.__dataset_files):
            return None
        self.__processed_file_idx += 1
        self.__total_files_rows_processed += len(self.__current_df) if (self.__current_df is not None) else 0
        file_to_process = self.__dataset_files[self.__processed_file_idx]
        self._logger.info("Processing file: '%s'" % file_to_process)
        self.__current_df = pd.read_csv(file_to_process)

        return self.__current_df

    def __shuffle_files(self, dataset_files: List[str]) -> List[str]:
        files = list(dataset_files)
        random.shuffle(files)

        return files
