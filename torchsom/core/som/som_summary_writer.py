from datetime import datetime
from enum import Enum

import os

import logging

import torch
from tensorboardX import SummaryWriter


class SomSummaryWriter:
    class SummariesName(Enum):
        LEARNING_RATE = "learning_rate"
        RADIUS = "radius"

    def __init__(self, board_dir: str):
        self._logger = logging.getLogger(__name__)
        self._summary_writer = self._build_summary_writer(board_dir)

    def write_learning_rate(self, learning_rate: torch.Tensor, step: int):
        self._summary_writer.add_scalar(self.SummariesName.LEARNING_RATE.value, learning_rate, step)

        return self

    def write_radius(self, radius: torch.Tensor, step: int):
        self._summary_writer.add_scalar(self.SummariesName.RADIUS.value, radius, step)

        return self

    def _build_summary_writer(self, board_dir: str) -> SummaryWriter:
        board_dir = os.path.join(board_dir, datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        self._logger.info("Tensorboard root directory: %s", board_dir)

        return SummaryWriter(board_dir)
