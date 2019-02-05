import os
import shutil

import pandas as pd
from datetime import datetime
from unittest import TestCase

from torchsom.core.input.dataset.generator.rgb_dataset import RgbDatasetGenerator


class TestRgbDataset(TestCase):
    OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

    @classmethod
    def setUpClass(cls):
        cls._rgb_dataset = RgbDatasetGenerator()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.OUT_DIR, ignore_errors=True)

    def test_generate_multiple_files(self):
        out_dir = os.path.join(self.OUT_DIR, datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        os.makedirs(out_dir)
        files_cnt = 5
        samples_in_file = 100
        self._rgb_dataset.generate(out_dir, files_cnt, samples_in_file)

        out_files = os.listdir(out_dir)
        self.assertEqual(len(out_files), files_cnt)
        for f in out_files:
            self._assert_file(os.path.join(out_dir, f), samples_in_file)

    def _assert_file(self, file: str, expected_len: int):
        df = pd.read_csv(file)
        self.assertEqual(df.shape, (expected_len, 3))
