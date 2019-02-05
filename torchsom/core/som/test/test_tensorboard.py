import os
from datetime import datetime
from unittest import TestCase

import torch
from tensorboardX.writer import SummaryWriter


class TestTensorboard(TestCase):
    LOG_DIR = os.path.join(os.path.dirname(__file__), "board")

    @classmethod
    def setUpClass(cls):
        cls._log_dir = cls.__setup_log_dir()

    @classmethod
    def __setup_log_dir(cls) -> str:
        log_dir = os.path.join(cls.LOG_DIR, "run-%s" % datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(log_dir)

        return log_dir

    def test_write_scalar(self):
        summary_writer = SummaryWriter(self._log_dir)
        tag_name = "learning_rate"
        learning_rate = torch.tensor(.01)
        for i in range(10):
            summary_writer.add_scalar(tag_name, learning_rate, i)
            learning_rate -= 0.005
