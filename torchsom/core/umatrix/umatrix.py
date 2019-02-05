import torch
import numpy as np

from torchsom.core.som.distance.distance import Distance


class Umatrix:
    def __init__(self, dist: Distance):
        self.__dist = dist
        self.__umatrix = None

    @property
    def umatrix(self) -> torch.Tensor:
        return self.__umatrix

    def build_umatrix(self, w: torch.Tensor) -> torch.Tensor:
        self.__umatrix = torch.tensor(np.zeros((2 * w.shape[0] - 1, 2 * w.shape[1] - 1)))
        for row_num in range(w.shape[0]):
            for col_num in range(w.shape[1]):
                self.__set_umatrix_value(w, row_num, col_num)

        return self.__umatrix

    def __set_umatrix_value(self, w: torch.Tensor, row_num: int, col_num: int):
        if row_num + 1 < w.shape[0]:
            dist = self.__calc_dist(w, row_num, col_num, row_num + 1, col_num)
            self.__umatrix[2 * row_num + 1][2 * col_num] = dist
        if (row_num + 1 < w.shape[0]) and (col_num + 1 < w.shape[1]):
            dist = self.__calc_dist(w, row_num, col_num, row_num + 1, col_num + 1)
            self.__umatrix[2 * row_num + 1][2 * col_num + 1] = dist
        if col_num + 1 < w.shape[1]:
            dist = self.__calc_dist(w, row_num, col_num, row_num, col_num + 1)
            self.__umatrix[2 * row_num][2 * col_num + 1] = dist
        self.__calc_umatrix_mean_value(row_num, col_num)

    def __calc_dist(self, w: torch.Tensor, val_row: int, val_col: int, sub_val_row: int, sub_val_col: int):
        dist_x_param = self.__get_dist_x(w, val_row, val_col)
        dist_y_param = self.__get_dist_y(w, sub_val_row, sub_val_col)

        return self.__dist.distance(dist_x_param, dist_y_param)[0][0][0]

    def __calc_umatrix_mean_value(self, row_num: int, col_num: int) -> torch.Tensor:
        neighbourhood_values = list()
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                if (i == 0) and (j == 0):
                    continue
                umatrix_row = 2 * row_num + i
                if (umatrix_row < 0) or (umatrix_row >= self.__umatrix.shape[0]):
                    continue
                umatrix_col = 2 * col_num + j
                if (umatrix_col < 0) or (umatrix_col >= self.__umatrix.shape[1]):
                    continue
                neighbourhood_values.append(self.__umatrix[umatrix_row][umatrix_col])

        self.__umatrix[2 * row_num][2 * col_num] = torch.mean(torch.tensor(neighbourhood_values))

    def __get_dist_x(self, w: torch.Tensor, row_num: int, col_num: int):
        x = w[row_num][col_num]
        x = x.reshape(1, x.shape[0]) if len(x.shape) < 2 else x

        return x

    def __get_dist_y(self, w: torch.Tensor, row_num: int, col_num: int):
        y = w[row_num][col_num]
        y_shape_len = len(y.shape)
        y = y.reshape(1, 1, y.shape[y_shape_len - 1]) if y_shape_len < 3 else y

        return y
