import os
import numpy as np
import logging
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.analysis import AnalysisBase, AnalysisConfiguration
from interestingness_xdrl.util.math import get_outliers_dist_mean, save_list_csv

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

MODEL_MEAN_IDX = 1


class EpistemicUncertaintyVarianceAnalysis(AnalysisBase):
    """
    Represents an analysis of the agent's epistemic uncertainty with regards to the environment's dynamics.
    This is the systematic uncertainty, due to things one could in principle know but do not in practice.
    This is the subjective uncertainty, i.e., due to limited data. Given an ensemble of probabilistic predictive models,
    this analysis computes the "disagreement" among the predictive models by measuring the prediction variance.
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
        self.all_pred_vars = np.zeros(len(data))  # timestep-indexed predictive models' variance
        self.low_pred_vars = []  # timesteps where variance is much higher than average
        self.high_pred_vars = []  # timesteps where variance is much lower than average
        self.mean_pred_var = 0.  # mean overall prediction variance across all timesteps

    def analyze(self, output_dir):
        if len(self.data) == 0 or self.data[0].next_obs is None:
            logging.info('Epistemic uncertainty: nothing to analyze, skipping')
            return

        logging.info('Analyzing epistemic uncertainty via variance in models\' predictions...')

        # gets mean prediction variance and outliers, where shape is (sample+mean+var (3), num_nets, obs_dim)
        next_obs = np.array([datapoint.next_obs[MODEL_MEAN_IDX] for datapoint in self.data])
        self.all_pred_vars = np.mean(np.square(np.linalg.norm(np.swapaxes(next_obs, 0, 1) - next_obs.mean(axis=1),
                                                              axis=-1)), axis=0)
        self.mean_pred_var = self.all_pred_vars.mean(0)
        all_outliers = set(get_outliers_dist_mean(self.all_pred_vars, self.config.epistemic_outlier_stds, True, True))

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
        logging.info('\tMean epistemic uncertainty over {} timesteps: {:.3f}'.format(
            len(self.data), self.mean_pred_var))
        logging.info('\tFound {} situations with high epistemic uncertainty (stds={}): {}'.format(
            len(self.high_pred_vars), self.config.epistemic_outlier_stds, self.high_pred_vars))
        logging.info('\tFound {} situations with low epistemic uncertainty (stds={}): {}'.format(
            len(self.low_pred_vars), self.config.epistemic_outlier_stds, self.low_pred_vars))

        logging.info('Saving report in {}...'.format(output_dir))

        # saves analysis report
        self.save(os.path.join(output_dir, 'epistemic-uncert-var.pkl.gz'))

        rwd_std = self.all_pred_vars.std(0)
        save_list_csv(list(self.all_pred_vars), os.path.join(output_dir, 'all-epistemic-uncert-var.csv'))
        self._plot_elements(self.all_pred_vars, self.high_pred_vars, self.low_pred_vars,
                            self.mean_pred_var + self.config.epistemic_outlier_stds * rwd_std,
                            self.mean_pred_var - self.config.epistemic_outlier_stds * rwd_std,
                            os.path.join(output_dir, 'epistemic-uncert-var-time.{}'.format(self.img_fmt)),
                            'High epistemic uncert. threshold', 'Low epistemic uncert. threshold',
                            'Epistemic Uncertainty', 'Predictive Models\' Observation Variance')

        save_list_csv(self.low_pred_vars, os.path.join(output_dir, 'low-epistemic-uncert-var.csv'))
        save_list_csv(self.high_pred_vars, os.path.join(output_dir, 'high-epistemic-uncert-var.csv'))

    def get_element_datapoint(self, datapoint):
        if datapoint.next_obs is None:
            return 'invalid', 0

        # next_obs shape is (sample+mean+var (3), num_nets, obs_dim)
        next_obs = datapoint.next_obs[MODEL_MEAN_IDX]
        pred_var = np.mean(np.square(np.linalg.norm(next_obs - next_obs.mean(axis=0), axis=-1)), axis=0)
        pred_var_std = self.all_pred_vars.std(0)
        above_thresh = self.mean_pred_var + self.config.aleatoric_outlier_stds * pred_var_std
        below_thresh = self.mean_pred_var - self.config.aleatoric_outlier_stds * pred_var_std
        return 'high-epistemic-uncert-var' if pred_var >= above_thresh else \
                   'low-epistemic-uncert-var' if pred_var <= below_thresh else '', \
               pred_var

    def get_element_time(self, t):
        return 'high-epistemic-uncert-var' if t in self.high_pred_vars else \
                   'low-epistemic-uncert-var' if t in self.low_pred_vars else '', \
               self.all_pred_vars[t]
