import os
from typing import List

import numpy as np
import pandas as pd

from unittest import TestCase

import torch
from torch import Tensor
from torch.utils import data

from torchsom.core.input.dataset.csv_dataset import CsvDataset


class TestCsvDataset(TestCase):
    DATASET_FILES_CNT = 3
    DATASET_FILE_ROWS_NUM = 3
    DATASET_FILE_COLS_NUM = 3
    DATASET_ROOT_DIR = os.path.join(os.path.dirname(__file__), "dataset")

    @classmethod
    def setUpClass(cls):
        cls.__dataset_files = cls._generate_files(cls.DATASET_ROOT_DIR, cls.DATASET_FILES_CNT,
                                                  cls.DATASET_FILE_ROWS_NUM, cls.DATASET_FILE_COLS_NUM)

    def test_read_from_dataset(self):
        dataset = CsvDataset(self.__dataset_files)
        data_loader = data.DataLoader(dataset, batch_size=2)
        processed_rows_cnt = 0
        processed_data = list()
        for data_batch in enumerate(data_loader):
            processed_rows_cnt += len(data_batch)
            processed_data.append(data_batch[1])
        self.__assert_with_csv_data(self.__dataset_files, processed_data)

    def __assert_with_csv_data(self, files: List[str], data: List[Tensor]):
        files_data = [pd.read_csv(f).values for f in files]
        tensor_data = [t.numpy() for t in data]
        fiels_data_array = np.concatenate(files_data, axis=0)
        tensor_data_array = np.concatenate(tensor_data, axis=0)
        self.assertTrue(np.array_equal(fiels_data_array, tensor_data_array))

    @classmethod
    def _generate_files(cls, out_file_dir: str, files_cnt: int, rows_num: int, cols_num: int) -> List[str]:
        return [cls._generate_single_file(out_file_dir, i, rows_num, cols_num) for i in range(files_cnt)]

    @classmethod
    def _generate_single_file(cls, out_file_dir: str, file_idx: int, rows_num: int, cols_num: int):
        x = np.random.rand(rows_num, cols_num)
        cols = ["col%d" % i for i in range(cols_num)]
        df = pd.DataFrame(x, columns=cols)
        out_file = cls._write_df(out_file_dir, file_idx, df)

        return out_file

    @classmethod
    def _write_df(cls, dir_name: str, file_idx: int, df: pd.DataFrame):
        os.makedirs(dir_name, exist_ok=True)
        out_file = os.path.join(dir_name, "file-%d.csv" % file_idx)
        df.to_csv(out_file, index=False)

        return out_file
