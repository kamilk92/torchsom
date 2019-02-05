import numpy as np
import os
import pandas as pd


class RgbDatasetGenerator:
    def generate(self, out_dir: str, files_cnt: int, samples_in_file: int):
        os.makedirs(out_dir, exist_ok=True)

        return [self._generate_file(out_dir, i, samples_in_file) for i in range(files_cnt)]

    def _generate_file(self, out_dir: str, file_idx: int, samples_in_file: int):
        data = self._generate_data(samples_in_file)
        filename = os.path.join(out_dir, "rgb-%d.csv" % file_idx)
        df = pd.DataFrame(data, columns=["r", "g", "b"])
        df.to_csv(filename, index=False)

        return filename

    def _generate_data(self, samples_cnt: int) -> np.ndarray:
        x = np.random.normal(100, 100, (samples_cnt, 3)).astype(np.float64)
        x[x < 0] = 0
        x[x > 255] = 255

        return x
