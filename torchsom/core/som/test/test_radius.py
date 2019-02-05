from unittest import TestCase

import torch

from torchsom.core.som.dim.map_dimension import MapDimension
from torchsom.core.som.radius import Radius


class TestRadius(TestCase):
    MAP_ROWS = 10
    MAP_COLS = 10

    def test_calculate_radius(self):
        map_dim = MapDimension(self.MAP_ROWS, self.MAP_COLS)
        total_iter_cnt = torch.tensor(float(1000))
        radius = Radius(map_dim, total_iter_cnt)
        iter_cnt = 100
        r = [radius.radius(i) for i in range(iter_cnt)]
        self.assertEqual(len(r), iter_cnt)
