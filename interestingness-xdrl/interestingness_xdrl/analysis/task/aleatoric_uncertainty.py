import os
import numpy as np
import logging
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.analysis import AnalysisBase, AnalysisConfiguration
from interestingness_xdrl.util.math import get_outliers_dist_mean, save_list_csv, gaussian_entropy

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

MODEL_VARIANCE_IDX = 2


def mean_entropy(next_obs_var):
    # next_obs_var shape is (num_nets, obs_dim)
    return np.mean([gaussian_entropy(next_obs_var[i]) for i in range(next_obs_var.shape[0])])


class AleatoricUncertaintyAnalysis(AnalysisBase):
    """
    Represents an analysis of the environment's aleatoric uncertainty. This is the statistical uncertainty
    representative of the inherent system stochasticity, i.e., the unknowns that differ each time we run the same
    experiment (here execute the same action in the same state). Given an ensemble of probabilistic predictive models,
    this analysis computes the mean entropy across the models' next-observation distributions.
    """

    def __init__(self, data, analysis_config, img_fmt):
        """
        Creates a new analysis.
        :param list[InteractionDataPoint] data: the interaction data collected to be analyzed.
        :param AnalysisConfiguration analysis_config: the analysis configuration containing the necessary parameters.
        :param str img_fmt: the format of the images to be saved.
        """
        super().__init__(data, analysis_config, img_fmt)
        # derived data
        self.all_pred_vars = np.zeros(len(data))  # timestep-indexed mean predictive models' variance
        self.low_pred_vars = []  # timesteps where prediction variance is much higher than average
        self.high_pred_vars = []  # timesteps where prediction variance is much lower than average
        self.mean_pred_var = 0.  # mean overall prediction variance across all timesteps

    def analyze(self, output_dir):
        if len(self.data) == 0 or self.data[0].next_obs is None:
            logging.info('Aleatoric uncertainty: nothing to analyze, skipping')
            return

        logging.info('Analyzing aleatoric uncertainty...')

        # gets mean prediction variance and outliers, where next_obs shape is (sample+mean+var (3), num_nets, obs_dim)
        self.all_pred_vars = np.array([mean_entropy(datapoint.next_obs[MODEL_VARIANCE_IDX])
                                       for datapoint in self.data])
        self.mean_pred_var = self.all_pred_vars.mean(0)
        all_outliers = set(get_outliers_dist_mean(self.all_pred_vars, self.config.aleatoric_outlier_stds, True, True))

        # registers outliers
        self.low_pred_vars = []
        self.high_pred_vars = []
        for t in range(1, len(self.data) - 1):
            # tests for above outlier
            if t in all_outliers and self.all_pred_vars[t] > self.mean_pred_var and \
                    self.all_pred_vars[t - 1] <= self.all_pred_vars[t] > self.all_pred_vars[t + 1]:
                self.high_pred_vars.append(t)
            # tests for below outlier
            elif t in all_outliers and self.all_pred_vars[t] < self.mean_pred_var and \
                    self.all_pred_vars[t - 1] >= self.all_pred_vars[t] < self.all_pred_vars[t + 1]:
                self.low_pred_vars.append(t)

        # sorts outliers
        self.high_pred_vars.sort(key=lambda i: self.all_pred_vars[i], reverse=True)
        self.low_pred_vars.sort(key=lambda i: self.all_pred_vars[i])

        # summary of elements
        logging.info('Finished')
        logging.info('\tMean aleatoric uncertainty over {} timesteps: {:.3f}'.format(
            len(self.data), self.mean_pred_var))
        logging.info('\tFound {} situations with high aleatoric uncertainty (stds={}): {}'.format(
            len(self.high_pred_vars), self.config.aleatoric_outlier_stds, self.high_pred_vars))
        logging.info('\tFound {} situations with low aleatoric uncertainty (stds={}): {}'.format(
            len(self.low_pred_vars), self.config.aleatoric_outlier_stds, self.low_pred_vars))

        logging.info('Saving report in {}...'.format(output_dir))

        # saves analysis report
        self.save(os.path.join(output_dir, 'aleatoric-uncertainty.pkl.gz'))

        rwd_std = self.all_pred_vars.std(0)
        save_list_csv(list(self.all_pred_vars), os.path.join(output_dir, 'all-aleatoric-uncert.csv'))
        self._plot_elements(self.all_pred_vars, self.high_pred_vars, self.low_pred_vars,
                            self.mean_pred_var + self.config.aleatoric_outlier_stds * rwd_std,
                            self.mean_pred_var - self.config.aleatoric_outlier_stds * rwd_std,
                            os.path.join(output_dir, 'aleatoric-uncert-time.{}'.format(self.img_fmt)),
                            'High aleatoric uncert. threshold', 'Low aleatoric uncert. threshold',
                            'Aleatoric Uncertainty', 'Mean Prediction Variance')

        save_list_csv(self.low_pred_vars, os.path.join(output_dir, 'low-aleatoric-uncert.csv'))
        save_list_csv(self.high_pred_vars, os.path.join(output_dir, 'high-aleatoric-uncert.csv'))

    def get_element_datapoint(self, datapoint):
        if datapoint.next_obs is None:
            return 'invalid', 0

        pred_var_std = self.all_pred_vars.std(0)
        pred_var = datapoint.next_obs[MODEL_VARIANCE_IDX].mean(axis=(1, 2))
        above_thresh = self.mean_pred_var + self.config.aleatoric_outlier_stds * pred_var_std
        below_thresh = self.mean_pred_var - self.config.aleatoric_outlier_stds * pred_var_std
        return 'high-aleatoric-uncert' if pred_var >= above_thresh else \
                   'low-aleatoric-uncert' if pred_var <= below_thresh else '', \
               pred_var

    def get_element_time(self, t):
        return 'high-aleatoric-uncert' if t in self.high_pred_vars else \
                   'low-aleatoric-uncert' if t in self.low_pred_vars else '', \
               self.all_pred_vars[t]
