import numpy as np
import torch

from unittest import TestCase

from torchsom.core.som.distance.euclidean.euclidean_distance import EuclideanDistance
from torchsom.core.umatrix.umatrix import Umatrix


class TestUmatrix(TestCase):
    @classmethod
    def setUpClass(cls):
        dist = EuclideanDistance()
        cls.__umatrix = Umatrix(dist)

    def test_generate_umatrix(self):
        w = torch.tensor(np.asarray([i for i in range(1, 10, 1)]), dtype=torch.float64).reshape((3, 3))
        umatrix = np.round(self.__umatrix.build_umatrix(w).numpy(), 2)
        expected = np.round(np.asarray([
            [2.67, 1, 2.6, 1, 2.67],
            [3, 4, 3, 4, 3],
            [3, 1, 3, 1, 3],
            [3, 4, 3, 4, 3],
            [2.67, 1, 2.6, 1, 2.67]
        ]), 2)
        np.testing.assert_array_equal(umatrix, expected)
        indices = np.asarray([(i, j) for i in range(umatrix.shape[0]) for j in range(umatrix.shape[1]) if
                   ((i % 2 != 0) or (j % 2 != 0))])
        print(indices)
        x_indices = indices[:, 0].flatten()
        y_indices = indices[:, 1].flatten()
        print(x_indices)
        print(y_indices)
        print(umatrix[x_indices, y_indices])
        # x_indices = indices[:, :, 0].flatten()
        # y_indices = indices[:, :, 1].flatten()
        # print(umatrix[x_indices, y_indices])
        # print(indices)
        # eq = np.where(indices == True)
        # indices = np.concatenate(eq).reshape(2, -1).T.reshape((3, 3, 2))
        # print(umatrix)
        # indices = np.asarray([(0, 0), (3, 3)])
        # print(umatrix.take(indices))
        # print(umatrix)
        # print(umatrix[eq].reshape(w.shape))
        # eq = np.where(indices == False)
        # print(umatrix)
        # print(umatrix[eq].shape)
