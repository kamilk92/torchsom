import os

from somplot.weights_plot import WeightsPlot
from torchsom.core.som.model import SomModel
from torchsom.core.som.som import Som
from torchsom.core.som.som_summary_writer import SomSummaryWriter
from torchsom.core.som.train_params import TrainParams
from torchsom.core.som.weights.weights_saver import WeightsSaver


class SomRgb(Som):
    def __init__(self, map_model: SomModel, summary_writer: SomSummaryWriter, train_params: TrainParams,
                 w_plot: WeightsPlot, weights_saver: WeightsSaver):
        super().__init__(map_model, summary_writer, train_params)
        self._w_plot = w_plot
        self._weights_saver = weights_saver

    def _evaluate(self):
        self._logger.info("Evaluating epoch %d, batch %d.", self._epoch_step, self._batch_step)
        w = self._map_model.weights
        self._weights_saver.save(w, self._train_params.eval_root_dir, self._epoch_step, self._batch_step)
        plot_filename = self._build_eval_plot_filename()
        self._w_plot.plot_weights(w, plot_filename)
        self._logger.debug("Evaluate plot saved into: '%s'", plot_filename)

    def _build_eval_plot_filename(self):
        root_dir = self._train_params.eval_root_dir or os.getcwd()
        eval_plot_filename = os.path.join(root_dir, "rgb-som-epoch-%d-batch-%d.html" %
                                          (self._epoch_step, self._batch_step))

        return eval_plot_filename
