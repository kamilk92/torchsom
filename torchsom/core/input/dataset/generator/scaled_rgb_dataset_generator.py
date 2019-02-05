import os

import pandas as pd
import numpy as np
import pickle

from typing import List, Union

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.externals import joblib

from torchsom.core.input.dataset.generator.rgb_dataset import RgbDatasetGenerator


class ScaledRgbDatasetGenerator(RgbDatasetGenerator):
    def __init__(self, scaler: Union[TransformerMixin, BaseEstimator]):
        self._scaler = scaler

    def generate(self, out_dir: str, files_cnt: int, samples_in_file: int):
        generated_files = super().generate(out_dir, files_cnt, samples_in_file)
        self._fit_scaler(generated_files)
        self._normalize_files(generated_files)
        self._pickle_scaler(os.path.dirname(generated_files[0]))

        return generated_files

    def _fit_scaler(self, generated_files: List[str]) -> np.ndarray:
        values = [pd.read_csv(rgb_file).values for rgb_file in generated_files]
        values = np.concatenate(values)
        self._scaler.fit(values)

    def _normalize_files(self, generated_files: List[str]):
        for f in generated_files:
            self._normalize_single_file(f)

    def _normalize_single_file(self, file: str):
        df = pd.read_csv(file)
        values = df.values
        values = self._scaler.transform(values)
        df = pd.DataFrame(values, columns=list(df.columns.values))
        df.to_csv(file, index=False)

    def _pickle_scaler(self, dirname: str):
        out_file = os.path.join(dirname, "scaler.pkl")
        with open(out_file, "wb") as pkl_file:
            pickle.dump(self._scaler, pkl_file)
