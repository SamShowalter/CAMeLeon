import os
import numpy as np
import logging
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.analysis import AnalysisBase, AnalysisConfiguration
from interestingness_xdrl.util.math import get_outliers_dist_mean, save_list_csv

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

MODEL_MEAN_IDX = 1
MODEL_VARIANCE_IDX = 2


class EpistemicUncertaintyJRDAnalysis(AnalysisBase):
    """
    Represents an analysis of the agent's epistemic uncertainty with regards to the environment's dynamics.
    This is the systematic uncertainty, due to things one could in principle know but do not in practice.
    This is the subjective uncertainty, i.e., due to limited data. Given an ensemble of probabilistic predictive models,
    this analysis computes the "disagreement" among the predictive models via the Jensen-Rényi Divergence (JRD).
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
        self.all_pred_jrds = np.zeros(len(data))  # timestep-indexed predictive models' variance
        self.low_pred_jrds = []  # timesteps where variance is much higher than average
        self.high_pred_jrds = []  # timesteps where variance is much lower than average
        self.mean_pred_jrd = 0.  # mean overall prediction variance across all timesteps

    def analyze(self, output_dir):
        if len(self.data) == 0 or self.data[0].next_obs is None:
            logging.info('Epistemic uncertainty: nothing to analyze, skipping')
            return

        logging.info('Analyzing epistemic uncertainty via JRD in models\' prob. predictions...')

        # gets mean prediction JRD and outliers, where next_obs shape is (sample+mean+var (3), num_nets, obs_dim)

        # based on https://github.com/nnaisense/MAX/blob/master/utilities.py
        # shape: (steps, ensemble_size, d_state)
        mu = np.array([datapoint.next_obs[MODEL_MEAN_IDX] for datapoint in self.data], dtype=np.double)
        var = np.array([datapoint.next_obs[MODEL_VARIANCE_IDX] for datapoint in self.data], dtype=np.double)
        steps, es, d_s = mu.shape

        # entropy of the mean
        entropy_mean = np.zeros(steps, dtype=np.double)
        for i in range(es):
            for j in range(es):
                mu_i, mu_j = mu[:, i], mu[:, j]  # shape: both (steps, d_state)
                var_i, var_j = var[:, i], var[:, j]  # shape: both (steps, d_state)

                mu_diff = mu_j - mu_i  # shape: (steps, d_state)
                var_sum = var_i + var_j  # shape: (steps, d_state)

                pre_exp = (mu_diff * 1 / var_sum * mu_diff)  # shape: (steps, d_state)
                pre_exp = np.sum(pre_exp, axis=-1)  # shape: (steps)
                exp = np.exp(-1 / 2 * pre_exp)  # shape: (steps)

                den = np.sqrt(np.prod(var_sum, axis=-1))  # shape: (steps)

                entropy_mean += exp / den  # shape: (steps)

        entropy_mean = -np.log(entropy_mean / (((2 * np.pi) ** (d_s / 2)) * (es * es)))  # shape: (steps)

        # mean of entropies
        total_entropy = np.prod(var, axis=-1)  # shape: (steps, ensemble_size)
        total_entropy = np.log(((2 * np.pi) ** d_s) * total_entropy)  # shape: (steps, ensemble_size)
        total_entropy = 1 / 2 * total_entropy + (d_s / 2) * np.log(2)  # shape: (steps, ensemble_size)
        mean_entropy = total_entropy.mean(axis=1)

        # Jensen-Renyi divergence
        self.all_pred_jrds = entropy_mean - mean_entropy  # shape: (steps)

        self.mean_pred_jrd = self.all_pred_jrds.mean(0)
        all_outliers = set(get_outliers_dist_mean(self.all_pred_jrds, self.config.epistemic_outlier_stds, True, True))

        # registers outliers
        self.low_pred_jrds = []
        self.high_pred_jrds = []
        for t in range(1, len(self.data) - 1):
            # tests for above outlier
            if t in all_outliers and self.all_pred_jrds[t] > self.mean_pred_jrd and \
                    self.all_pred_jrds[t - 1] <= self.all_pred_jrds[t] > self.all_pred_jrds[t + 1]:
                self.high_pred_jrds.append(t)
            # tests for below outlier
            elif t in all_outliers and self.all_pred_jrds[t] < self.mean_pred_jrd and \
                    self.all_pred_jrds[t - 1] >= self.all_pred_jrds[t] < self.all_pred_jrds[t + 1]:
                self.low_pred_jrds.append(t)

        # sorts outliers
        self.high_pred_jrds.sort(key=lambda i: self.all_pred_jrds[i], reverse=True)
        self.low_pred_jrds.sort(key=lambda i: self.all_pred_jrds[i])

        # summary of elements
        logging.info('Finished')
        logging.info('\tMean epistemic uncertainty over {} timesteps: {:.3f}'.format(
            len(self.data), self.mean_pred_jrd))
        logging.info('\tFound {} situations with high epistemic uncertainty (stds={}): {}'.format(
            len(self.high_pred_jrds), self.config.epistemic_outlier_stds, self.high_pred_jrds))
        logging.info('\tFound {} situations with low epistemic uncertainty (stds={}): {}'.format(
            len(self.low_pred_jrds), self.config.epistemic_outlier_stds, self.low_pred_jrds))

        logging.info('Saving report in {}...'.format(output_dir))

        # saves analysis report
        self.save(os.path.join(output_dir, 'epistemic-uncert-jrd.pkl.gz'))

        rwd_std = self.all_pred_jrds.std(0)
        save_list_csv(list(self.all_pred_jrds), os.path.join(output_dir, 'all-epistemic-uncert-jrd.csv'))
        self._plot_elements(self.all_pred_jrds, self.high_pred_jrds, self.low_pred_jrds,
                            self.mean_pred_jrd + self.config.epistemic_outlier_stds * rwd_std,
                            self.mean_pred_jrd - self.config.epistemic_outlier_stds * rwd_std,
                            os.path.join(output_dir, 'epistemic-uncert-jrd-time.{}'.format(self.img_fmt)),
                            'High epistemic uncert. threshold', 'Low epistemic uncert. threshold',
                            'Epistemic Uncertainty', 'Predictive Models\' Observation JRD')

        save_list_csv(self.low_pred_jrds, os.path.join(output_dir, 'low-epistemic-uncert-jrd.csv'))
        save_list_csv(self.high_pred_jrds, os.path.join(output_dir, 'high-epistemic-uncert-jrd.csv'))

    def get_element_datapoint(self, datapoint):
        if datapoint.next_obs is None:
            return 'invalid', 0

        # next_obs shape is (sample+mean+var (3), num_nets, obs_dim)
        next_obs = datapoint.next_obs[MODEL_MEAN_IDX]
        pred_var = np.mean(np.square(np.linalg.norm(next_obs - next_obs.mean(axis=0), axis=-1)), axis=0)
        pred_var_std = self.all_pred_jrds.std(0)
        above_thresh = self.mean_pred_jrd + self.config.aleatoric_outlier_stds * pred_var_std
        below_thresh = self.mean_pred_jrd - self.config.aleatoric_outlier_stds * pred_var_std
        return 'high-epistemic-uncert-jrd' if pred_var >= above_thresh else \
                   'low-epistemic-uncert-jrd' if pred_var <= below_thresh else '', \
               pred_var

    def get_element_time(self, t):
        return 'high-epistemic-uncert-jrd' if t in self.high_pred_jrds else \
                   'low-epistemic-uncert-jrd' if t in self.low_pred_jrds else '', \
               self.all_pred_jrds[t]
