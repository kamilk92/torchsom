import dependency_injector.containers as containers
import dependency_injector.providers as providers

from somplot.scaled_weights_plot import ScaledWeightsPlot
from somplot.scaler import Scaler
from somplot.weights_plot import WeightsPlot


class SomPlotContainer(containers.DeclarativeContainer):
    scaled_weights_plot_provider = providers.Factory(ScaledWeightsPlot)
    weights_plot_provider = providers.Factory(WeightsPlot)

    @classmethod
    def get_scaled_weights_plot(cls, scaler: Scaler) -> ScaledWeightsPlot:
        return cls.scaled_weights_plot_provider(scaler)

    @classmethod
    def get_weights_plot(cls) -> WeightsPlot:
        return cls.weights_plot_provider()
