import logging
import numpy as np
import os
import pandas as pd


class WeightsSaver:
    def __init__(self, feats_cnt):
        self._logger = logging.getLogger(__name__)
        self.__feats_cnt = feats_cnt

    def save(self, weights: np.ndarray, out_dir: str, epoch: int, batch_step: int):
        weights = weights.reshape(-1, self.__feats_cnt)
        df = pd.DataFrame(weights, columns=["feat%d" % i for i in range(self.__feats_cnt)])
        out_file = self.__build_out_file(out_dir, epoch, batch_step)
        df.to_csv(out_file, index=False)
        self._logger.info("Weights saved into: %s", out_file)

        return out_file

    def __build_out_file(self, out_dir: str, epoch: int, batch_step: int) -> str:
        return os.path.join(out_dir, "w-epoch-%d-batch-%d.csv" % (epoch, batch_step))
