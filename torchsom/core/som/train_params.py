import os


class TrainParams:
    def __init__(self, batch_size: int, epochs: int, summary_step: int, eval_batch_step: int = 0,
                 eval_root_dir: str = None):
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__eval_batch_step = eval_batch_step
        self.__eval_root_dir = eval_root_dir
        self.__summary_step = summary_step

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @property
    def epochs(self) -> int:
        return self.__epochs

    @property
    def eval_batch_step(self) -> int:
        return self.__eval_batch_step

    @property
    def eval_root_dir(self) -> str:
        return self.__eval_root_dir

    @property
    def summary_step(self) -> int:
        return self.__summary_step
