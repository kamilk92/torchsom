import os

import logging
import pandas as pd
import numpy as np
from unittest import TestCase

import torch

from torchsom.core.input.dataset.csv_dataset import CsvDataset
from torchsom.core.input.dataset.factory.csv_dataset_factory import CsvDatasetFactory
from torchsom.core.som.bmu.bmu import Bmu
from torchsom.core.som.dim.map_dimension import MapDimension
from torchsom.core.som.distance.euclidean.euclidean_distance import EuclideanDistance
from torchsom.core.som.learning_rate import LearningRate
from torchsom.core.som.map_indices_builder import MapIndicesBuilder
from torchsom.core.som.model import SomModel
from torchsom.core.som.radius import Radius
from torchsom.core.som.som import Som
from torchsom.core.som.som_summary_writer import SomSummaryWriter
from torchsom.core.som.train_params import TrainParams
from torchsom.core.som.weights.weights_updater import WeightsUpdater
from torchsom.core.util import log_util


class TestSom(TestCase):
    BATCH_SIZE = 10
    EPOCHS = 5
    EVAL_BATCH_STEP = 5
    FEATS_CNT = 3
    LEARNING_RATE = 0.01
    MAP_SIZE = 20
    OUT_ROOT_DIR = os.path.join(os.path.dirname(__file__), "data")
    EVAL_ROOT_DIR = os.path.join(os.path.dirname(__file__), "eval")
    ROWS_CNT = 100
    SUMMARY_STEP = 1

    @classmethod
    def setUpClass(cls):
        cls.__logger = cls.__setup_logger()
        cls.__corpus_file = cls.__setup_file(cls.FEATS_CNT, 0, cls.ROWS_CNT)
        cls.__map_dim = MapDimension(cls.MAP_SIZE, cls.MAP_SIZE, cls.FEATS_CNT)
        cls.__map_indices = MapIndicesBuilder().build(cls.__map_dim)
        cls.__bmu = Bmu(EuclideanDistance(), cls.__map_indices)
        cls.__dataset = CsvDatasetFactory([cls.__corpus_file])
        cls.__total_iter_cnt = torch.tensor(cls.ROWS_CNT / cls.BATCH_SIZE, dtype=torch.float64)
        cls.__learning_rate = LearningRate(cls.LEARNING_RATE, cls.__total_iter_cnt)
        cls.__summary_writer = SomSummaryWriter(cls.EVAL_ROOT_DIR)
        cls.__radius = Radius(cls.__map_dim, cls.__total_iter_cnt)
        cls.__weights_updater = WeightsUpdater(cls.__map_indices)

    def test_train_som(self):
        som_model = SomModel(self.__bmu, self.__learning_rate, self.__map_dim, self.__map_indices, self.__radius,
                             self.__weights_updater)
        train_params = TrainParams(self.BATCH_SIZE, self.EPOCHS, self.SUMMARY_STEP, self.EVAL_BATCH_STEP,
                                   self.EVAL_ROOT_DIR)
        som = Som(som_model, self.__summary_writer, train_params)

        trained_weights = som.train(self.__dataset)

        self.assertEqual(trained_weights.shape, (self.MAP_SIZE, self.MAP_SIZE, self.FEATS_CNT))

    @classmethod
    def __setup_file(cls, feats_cnt: int, file_idx: int, rows_cnt: int):
        data = np.random.uniform(0, 1, (rows_cnt, feats_cnt))
        os.makedirs(cls.OUT_ROOT_DIR, exist_ok=True)
        out_file = os.path.join(cls.OUT_ROOT_DIR, "file-%d.csv" % file_idx)
        df = pd.DataFrame(data, columns=["feat%d" for i in range(data.shape[1])])
        df.to_csv(out_file, index=False)

        return out_file

    @classmethod
    def __setup_logger(cls):
        log_util.setup_logging("logging.yaml")

        return logging.getLogger(__name__)
