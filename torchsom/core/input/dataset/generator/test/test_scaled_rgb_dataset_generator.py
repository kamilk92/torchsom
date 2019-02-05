import os
import shutil
from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

from torchsom.core.input.dataset.generator.scaled_rgb_dataset_generator import ScaledRgbDatasetGenerator


class TestScaledRgbDatasetGenerator(TestCase):
    OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.OUT_DIR, ignore_errors=True)

    def test_generate_scaled_rgb_dataset(self):
        files_cnt = 5
        samples_in_file = 100
        scaler = MinMaxScaler()
        dataset_generator = ScaledRgbDatasetGenerator(scaler)

        generated_files = dataset_generator.generate(self.OUT_DIR, files_cnt, samples_in_file)

        self.assertEqual(len(generated_files), files_cnt)
        for f in generated_files:
            self._assert_file(f, samples_in_file)

        self._assert_scaler_file()

    def _assert_file(self, file: str, expected_samples_cnt, min_range=0.0, max_range=1.0):
        self.assertTrue(os.path.isfile(file))
        df = pd.read_csv(file)
        self.assertEqual(df.shape, (expected_samples_cnt, 3))
        self._assert_values_in_range(df.values, min_range, max_range)

    def _assert_values_in_range(self, values: np.ndarray, min_range: float, max_range: float):
        for value in np.nditer(values):
            self.assertGreaterEqual(np.asscalar(value), min_range)
            self.assertLessEqual(np.asscalar(value), max_range)

    def _assert_scaler_file(self):
        self.assertTrue(any(f == "scaler.pkl" for f in os.listdir(self.OUT_DIR)))
        file = os.path.join(self.OUT_DIR, "scaler.pkl")
        scaler = joblib.load(file)
        self.assertIsInstance(scaler, MinMaxScaler)
