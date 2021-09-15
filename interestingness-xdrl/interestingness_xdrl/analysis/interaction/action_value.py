import os
import numpy as np
import logging
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.analysis import AnalysisBase, AnalysisConfiguration
from interestingness_xdrl.util.math import get_outliers_dist_mean, save_list_csv

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class ActionValueAnalysis(AnalysisBase):
    """
    Represents an analysis of an agent's state action-value function. It extracts information on the states where the
    maximum absolute difference between the values of any two actions (i.e., action value peak-to-peak) is
    significantly more or less valued than others (outliers).
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
        self.all_action_diffs = np.zeros(len(data))  # timestep-indexed mean action-value differences (max-min)
        self.low_action_diffs = []  # timesteps where value diffs are much higher than average
        self.high_action_diffs = []  # timesteps where value diffs are much lower than average
        self.mean_diff = 0.  # mean overall action-value difference across all timesteps

    def analyze(self, output_dir):
        logging.info('Analyzing maximum value difference...')

        # gets mean ptps and outliers
        self.all_action_diffs = np.array([np.mean([np.ptp(factor_values) for factor_values in datapoint.action_values])
                                          for datapoint in self.data])
        self.mean_diff = self.all_action_diffs.mean(0)
        all_outliers = set(get_outliers_dist_mean(self.all_action_diffs, self.config.value_outlier_stds, True, True))

        # registers outliers
        self.low_action_diffs = []
        self.high_action_diffs = []
        for t in range(1, len(self.data) - 1):
            # tests for above outlier
            if t in all_outliers and self.all_action_diffs[t] > self.mean_diff and \
                    self.all_action_diffs[t - 1] <= self.all_action_diffs[t] > self.all_action_diffs[t + 1]:
                self.high_action_diffs.append(t)
            # tests for below outlier
            elif t in all_outliers and self.all_action_diffs[t] < self.mean_diff and \
                    self.all_action_diffs[t - 1] >= self.all_action_diffs[t] < self.all_action_diffs[t + 1]:
                self.low_action_diffs.append(t)

        # sorts outliers
        self.high_action_diffs.sort(key=lambda i: self.all_action_diffs[i], reverse=True)
        self.low_action_diffs.sort(key=lambda i: self.all_action_diffs[i])

        # summary of elements
        logging.info('Finished')
        logging.info('\tMean max action-value difference over {} timesteps: {:.3f}'.format(
            len(self.data), self.mean_diff))
        logging.info('\tFound {} high action-value differences (stds={}): {}'.format(
            len(self.high_action_diffs), self.config.value_outlier_stds, self.high_action_diffs))
        logging.info('\tFound {} low action-value differences (stds={}): {}'.format(
            len(self.low_action_diffs), self.config.value_outlier_stds, self.low_action_diffs))

        logging.info('Saving report in {}...'.format(output_dir))

        # saves analysis report
        self.save(os.path.join(output_dir, 'action-value.pkl.gz'))

        value_std = self.all_action_diffs.std(0)
        save_list_csv(list(self.all_action_diffs), os.path.join(output_dir, 'all-action-diffs.csv'))
        self._plot_elements(self.all_action_diffs, self.high_action_diffs, self.low_action_diffs,
                            self.mean_diff + self.config.value_outlier_stds * value_std,
                            self.mean_diff - self.config.value_outlier_stds * value_std,
                            os.path.join(output_dir, 'action-diff-time.{}'.format(self.img_fmt)),
                            'High action-diff. threshold', 'Low action-diff. threshold',
                            'Maximum Action-Value Difference', 'Action-Value Diff.')

        save_list_csv(self.low_action_diffs, os.path.join(output_dir, 'low-action-diffs.csv'))
        save_list_csv(self.high_action_diffs, os.path.join(output_dir, 'high-action-diffs.csv'))

    def get_element_datapoint(self, datapoint):
        value_std = self.all_action_diffs.std(0)
        action_diff = np.mean([np.ptp(factor_values) for factor_values in datapoint.action_values])
        return 'high-action-diff' if action_diff >= self.mean_diff + self.config.value_outlier_stds * value_std else \
                   'low-action-diff' if action_diff <= self.mean_diff - self.config.value_outlier_stds * value_std \
                       else '', \
               action_diff

    def get_element_time(self, t):
        return 'high-action-diff' if t in self.high_action_diffs else \
                   'low-action-diff' if t in self.low_action_diffs else '', \
               self.all_action_diffs[t]
