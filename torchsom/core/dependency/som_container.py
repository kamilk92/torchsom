import dependency_injector.containers as containers
import dependency_injector.providers as providers
import torch
from typing import Tuple

from somplot.scaler import Scaler
from torchsom.core.dependency.som_plot_container import SomPlotContainer
from torchsom.core.som.bmu.bmu import Bmu
from torchsom.core.som.dim.map_dimension import MapDimension
from torchsom.core.som.distance.distance import Distance
from torchsom.core.som.distance.euclidean.euclidean_distance import EuclideanDistance
from torchsom.core.som.learning_rate import LearningRate
from torchsom.core.som.map_indices_builder import MapIndicesBuilder
from torchsom.core.som.model import SomModel
from torchsom.core.som.radius import Radius
from torchsom.core.som.som import Som
from torchsom.core.som.som_rgb import SomRgb
from torchsom.core.som.som_summary_writer import SomSummaryWriter
from torchsom.core.som.train_params import TrainParams
from torchsom.core.som.weights.weights_saver import WeightsSaver
from torchsom.core.som.weights.weights_updater import WeightsUpdater


class SomContainer(containers.DeclarativeContainer):
    eucd_dist_provider = providers.Factory(EuclideanDistance)
    bmu_provider = providers.Factory(Bmu)
    learning_rate_provider = providers.Factory(LearningRate)
    map_dim_provider = providers.Factory(MapDimension)
    map_indices_builder_provider = providers.Factory(MapIndicesBuilder)
    neighbourhood_radius_provider = providers.Factory(Radius)
    rgb_som_provider = providers.Factory(SomRgb)
    som_model_provider = providers.Factory(SomModel)
    som_provider = providers.Factory(Som)
    som_summary_writer_provider = providers.Factory(SomSummaryWriter)
    weights_saver_provider = providers.Factory(WeightsSaver)
    weights_updater_provider = providers.Factory(WeightsUpdater)

    @classmethod
    def get_bmu(cls, map_indices: torch.Tensor, dist: Distance = None) -> Bmu:
        dist = dist or cls.eucd_dist_provider()

        return cls.bmu_provider(dist, map_indices)

    @classmethod
    def get_eucd_dist(cls) -> EuclideanDistance:
        return cls.eucd_dist_provider()

    @classmethod
    def get_learning_rate(cls, learning_rate: float, total_iter_cnt: torch.Tensor) -> LearningRate:
        return cls.learning_rate_provider(learning_rate, total_iter_cnt)

    @classmethod
    def get_map_dim(cls, rows: int, cols: int, features: int) -> MapDimension:
        return cls.map_dim_provider(rows, cols, features)

    @classmethod
    def get_map_indices_builder(cls) -> MapIndicesBuilder:
        return cls.map_indices_builder_provider()

    @classmethod
    def get_neighbourhood_radius(cls, map_dim: MapDimension, total_iter_cnt: torch.Tensor) -> Radius:
        return cls.neighbourhood_radius_provider(map_dim, total_iter_cnt)

    @classmethod
    def get_rgb_som(cls, map_dim: MapDimension, learning_rate_val: float, total_iter_cnt: float,
                    w_range: Tuple[float, float], board_dir: str, train_params: TrainParams, dist: Distance = None,
                    w_plot_scaler: Scaler = None, save_w=False):
        som_model = cls.get_som_model_from_params(map_dim, learning_rate_val, total_iter_cnt, w_range, dist)
        summary_writer = cls.get_som_summary_writer(board_dir)
        w_plot = SomPlotContainer.get_scaled_weights_plot(w_plot_scaler) if w_plot_scaler \
            else SomPlotContainer.get_weights_plot()
        w_saver = cls.get_weights_saver(map_dim.features) if save_w else None

        return cls.rgb_som_provider(som_model, summary_writer, train_params, w_plot, w_saver)

    @classmethod
    def get_som(cls, map_dim: MapDimension, learning_rate_val: float, total_iter_cnt: float,
                w_range: Tuple[float, float], board_dir: str, train_params: TrainParams, dist: Distance = None):
        som_model = cls.get_som_model_from_params(map_dim, learning_rate_val, total_iter_cnt, w_range, dist)
        summary_writer = cls.get_som_summary_writer(board_dir)

        return cls.som_provider(som_model, summary_writer, train_params)

    @classmethod
    def get_som_model(cls, bmu: Bmu, learning_rate: LearningRate, map_dim: MapDimension, map_indices: torch.Tensor,
                      neighbourhood_radius: Radius, weights_updater: WeightsUpdater,
                      w_range: Tuple[float, float] = None) -> SomModel:
        return cls.som_model_provider(bmu, learning_rate, map_dim, map_indices, neighbourhood_radius, weights_updater,
                                      w_range)

    @classmethod
    def get_som_model_from_params(cls, map_dim: MapDimension, learning_rate_val: float, total_iter_cnt: float,
                                  w_range: Tuple[float, float], dist: Distance = None):
        map_indices_builder = cls.get_map_indices_builder()
        map_indices = map_indices_builder.build(map_dim)
        dist = dist or cls.get_eucd_dist()
        bmu = cls.get_bmu(map_indices, dist)
        learning_rate_obj = cls.get_learning_rate(learning_rate_val, torch.tensor(total_iter_cnt, dtype=torch.float64))
        radius = cls.get_neighbourhood_radius(map_dim, total_iter_cnt)
        w_updater = cls.get_weights_updater(map_indices)
        som_model = cls.get_som_model(bmu, learning_rate_obj, map_dim, map_indices, radius, w_updater, w_range)

        return som_model

    @classmethod
    def get_som_summary_writer(cls, board_dir: str) -> SomSummaryWriter:
        cls.som_summary_writer_provider(board_dir)

    @classmethod
    def get_weights_saver(cls, feats_cnt: int) -> WeightsSaver:
        cls.weights_saver_provider(feats_cnt)

    @classmethod
    def get_weights_updater(cls, map_indicecs: torch.Tensor) -> WeightsUpdater:
        return cls.weights_updater_provider(map_indicecs)
