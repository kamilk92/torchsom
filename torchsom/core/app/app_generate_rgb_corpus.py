import datetime
import os

from sklearn.preprocessing.data import StandardScaler
from typing import List

from sklearn.preprocessing import MinMaxScaler

from torchsom.core.input.dataset.generator.rgb_dataset import RgbDatasetGenerator
from torchsom.core.input.dataset.generator.scaled_rgb_dataset_generator import ScaledRgbDatasetGenerator

FILES_CNT_TO_GENERATE = 1
OUT_ROOT_DIR = os.path.join(os.path.dirname(__file__), "out")


class AppGenerateRgbCorpus:
    def __init__(self, rgb_dataset_generator: RgbDatasetGenerator):
        self.__rgb_dataset_generator = rgb_dataset_generator

    def generate(self, files_cnt: int, samples_in_file: int, out_root_dir: str) -> List[str]:
        out_dir = self._build_out_dir(out_root_dir)

        return self.__rgb_dataset_generator.generate(out_dir, files_cnt, samples_in_file)

    def _build_out_dir(self, out_root_dir: str) -> str:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_dir = os.path.join(out_root_dir, current_date)
        os.makedirs(out_dir, exist_ok=True)

        return out_dir


if __name__ == '__main__':
    scaler = MinMaxScaler((-1, 1))
    dataset_generator = ScaledRgbDatasetGenerator(scaler)
    app_generate_rgb_corpus = AppGenerateRgbCorpus(dataset_generator)
    app_generate_rgb_corpus.generate(FILES_CNT_TO_GENERATE, 100000, OUT_ROOT_DIR)
