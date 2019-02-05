import logging
import os
import pickle
from unittest import TestCase

import datetime
import numpy as np
import torch
from typing import List

from somplot.scaled_weights_plot import ScaledWeightsPlot
from somplot.umatrix_plot import UmatrixPlot
from somplot.weights_plot import WeightsPlot
from torchsom.core.input.dataset.factory.csv_dataset_factory import CsvDatasetFactory
from torchsom.core.som.bmu.bmu import Bmu
from torchsom.core.som.dim.map_dimension import MapDimension
from torchsom.core.som.distance.euclidean.euclidean_distance import EuclideanDistance
from torchsom.core.som.learning_rate import LearningRate
from torchsom.core.som.map_indices_builder import MapIndicesBuilder
from torchsom.core.som.model import SomModel
from torchsom.core.som.radius import Radius
from torchsom.core.som.som_rgb import SomRgb
from torchsom.core.som.som_summary_writer import SomSummaryWriter
from torchsom.core.som.train_params import TrainParams
from torchsom.core.som.weights.weights_saver import WeightsSaver
from torchsom.core.som.weights.weights_updater import WeightsUpdater
from torchsom.core.util import log_util


class TestSomRgb(TestCase):
    BATCH_SIZE = 100
    CORPUS_ROOT_DIR = "torchsom/core/corpus/rgb/not-normalized"
    EPOCHS = 1
    EVAL_BATCH_STEP = 10
    EVAL_ROOT_DIR = os.path.join(os.path.dirname(__file__), "eval-out",
                                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    LEARNING_RATE = 0.1
    LOGGING_FILE = "logging.yaml"
    MAP_SIZE = 30
    SAMPLES_IN_FILE = 200000
    SCALER_FILE = "torchsom/core/corpus/rgb/small/scaler.pkl"
    SUMMARY_STEP = EVAL_BATCH_STEP

    @classmethod
    def setUpClass(cls):
        np.set_printoptions(threshold=np.inf)
        cls.__corpus_files = cls.__setup_corpus_files()
        cls.__logger = cls.__setup_logger()
        cls.__map_dim = MapDimension(cls.MAP_SIZE, cls.MAP_SIZE, 3)
        cls.__map_indices = MapIndicesBuilder().build(cls.__map_dim)
        cls.__bmu = Bmu(EuclideanDistance(), cls.__map_indices)
        cls.__dataset = CsvDatasetFactory(cls.__corpus_files, False)
        cls.__total_iter_cnt = torch.tensor(cls.SAMPLES_IN_FILE * len(cls.__corpus_files) / cls.BATCH_SIZE,
                                            dtype=torch.float64)
        cls.__learning_rate = LearningRate(cls.LEARNING_RATE, cls.__total_iter_cnt)
        cls.__radius = Radius(cls.__map_dim, cls.__total_iter_cnt)
        cls.__weights_updater = WeightsUpdater(cls.__map_indices)
        cls.__weights_plot = WeightsPlot()
        cls.__summary_writer = SomSummaryWriter(cls.EVAL_ROOT_DIR)
        cls.__weights_sver = WeightsSaver(3)

    def test_train_som(self):
        som_model = SomModel(self.__bmu, self.__learning_rate, self.__map_dim, self.__map_indices, self.__radius,
                             self.__weights_updater)
        train_params = TrainParams(self.BATCH_SIZE, self.EPOCHS, self.SUMMARY_STEP, self.EVAL_BATCH_STEP,
                                   self.EVAL_ROOT_DIR)
        som = SomRgb(som_model, self.__summary_writer, train_params, self.__weights_plot, self.__weights_sver)
        trained_weights = som.train(self.__dataset)

        self.assertEqual(trained_weights.shape, (self.MAP_SIZE, self.MAP_SIZE, 3))

    @classmethod
    def __setup_corpus_files(cls) -> List[str]:
        return [os.path.join(cls.CORPUS_ROOT_DIR, f) for f in os.listdir(cls.CORPUS_ROOT_DIR)]

    @classmethod
    def __setup_logger(cls) -> logging.Logger:
        log_util.setup_logging(cls.LOGGING_FILE)

        return logging.getLogger(__name__)

    @classmethod
    def __setup_weights_plot(cls) -> WeightsPlot:
        with open(cls.SCALER_FILE, "rb") as pkl_file:
            scaler = pickle.load(pkl_file)
        scaled_weigths_plot = ScaledWeightsPlot(scaler)

        return scaled_weigths_plot
