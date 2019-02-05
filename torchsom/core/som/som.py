import logging

import torch
from torch.utils import data

from torchsom.core.input.dataset.factory.datset_factory import DatasetFactory
from torchsom.core.som.model import SomModel
from torchsom.core.som.som_summary_writer import SomSummaryWriter
from torchsom.core.som.train_params import TrainParams


class Som:
    def __init__(self, map_model: SomModel, summary_writer: SomSummaryWriter, train_params: TrainParams):
        self._logger = logging.getLogger(__name__)
        self._batch_step = 0
        self._epoch_step = 0
        self._global_step = 0
        self._map_model = map_model
        self._summary_writer = summary_writer
        self._train_params = train_params

    def train(self, dataset_factory: DatasetFactory) -> torch.Tensor:
        for self._epoch_step in range(self._train_params.epochs):
            self._logger.info("Starting epoch %d.", self._epoch_step)
            data_loader = self._build_data_loader(dataset_factory)
            self.__train_epoch(data_loader)

        return self._map_model.weights

    def _build_data_loader(self, dataset_factory: DatasetFactory) -> data.DataLoader:
        dataset = dataset_factory.create_dataset()
        data_loader = data.DataLoader(dataset, batch_size=self._train_params.batch_size)

        return data_loader

    def _evaluate(self):
        self._logger.warning("Evaluating epoch %d, batch %d but evaluate not implemented.",
                             self._epoch_step, self._batch_step)

    def _write_summaries(self):
        if (self._batch_step % self._train_params.summary_step) != 0:
            return
        self._logger.debug("Writing summaries for epoch %d, batch %d, global step %d",
                           self._epoch_step, self._batch_step, self._global_step)
        self._summary_writer.write_learning_rate(self._map_model.learning_rate, self._global_step)
        self._summary_writer.write_radius(self._map_model.neighbourhood_radius, self._global_step)

    def __is_eval_batch(self) -> bool:
        eval_batch_step = self._train_params.eval_batch_step

        return (eval_batch_step > 0) and (self._batch_step % eval_batch_step == 0)

    def __train_epoch(self, data_loader: data.DataLoader):
        iter_num = torch.tensor(0, dtype=torch.float64)
        for self._batch_step, data_batch in enumerate(data_loader):
            self.__train_batch(data_batch, iter_num)
            iter_num += 1

    def __train_batch(self, data_batch: torch.Tensor, iter_num: torch.Tensor):
        self._map_model.train(data_batch, iter_num)
        self._write_summaries()
        self._global_step += 1
        if self.__is_eval_batch():
            self._evaluate()
