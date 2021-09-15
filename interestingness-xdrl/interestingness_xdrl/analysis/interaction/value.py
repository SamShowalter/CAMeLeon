import os
import numpy as np
import logging
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.analysis import AnalysisBase, AnalysisConfiguration
from interestingness_xdrl.util.math import get_outliers_dist_mean, save_list_csv

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class ValueAnalysis(AnalysisBase):
    """
    Represents an analysis of an agent's state value function. It extracts information on the states that are
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
        self.values = np.zeros(len(data))  # timestep-indexed state value
        self.low_values = []  # timesteps where values are much higher than average
        self.high_values = []  # timesteps where values are much lower than average
        self.value_mean = 0.  # mean overall value across all timesteps
        self.value_std = 0.  # std dev of mean overall value across all timesteps

    def analyze(self, output_dir):
        logging.info('Analyzing value...')

        # gets mean values and outliers
        self.values = np.array([datapoint.value for datapoint in self.data])
        self.value_mean = np.mean(self.values, axis=0)
        self.value_std = np.std(self.values, axis=0)
        all_outliers = set(get_outliers_dist_mean(self.values, self.config.value_outlier_stds, True, True))

        # registers outliers
        self.low_values = []
        self.high_values = []
        for t in range(1, len(self.data) - 1):
            # tests for above outlier
            if t in all_outliers and self.values[t] > self.value_mean and \
                    self.values[t - 1] <= self.values[t] > self.values[t + 1]:
                self.high_values.append(t)
            # tests for below outlier
            elif t in all_outliers and self.values[t] < self.value_mean and \
                    self.values[t - 1] >= self.values[t] < self.values[t + 1]:
                self.low_values.append(t)

        # sorts outliers
        self.high_values.sort(key=lambda i: self.values[i], reverse=True)
        self.low_values.sort(key=lambda i: self.values[i])

        # summary of elements
        logging.info('Finished')
        logging.info(f'\tMean value received over {len(self.data)} timesteps: '
                     f'{self.value_mean:.3f} ± {self.value_std:.3f}')
        logging.info('\tFound {} high values (stds={}): {}'.format(
            len(self.high_values), self.config.value_outlier_stds, self.high_values))
        logging.info('\tFound {} low values (stds={}): {}'.format(
            len(self.low_values), self.config.value_outlier_stds, self.low_values))

        logging.info('Saving report in {}...'.format(output_dir))

        # saves analysis report
        self.save(os.path.join(output_dir, 'value.pkl.gz'))

        value_std = self.values.std(0)
        self._save_time_dataset_csv(self.values, 'Value', os.path.join(output_dir, 'values-time.csv'))
        self._plot_elements_sp(self.values,
                               self.value_mean + self.config.value_outlier_stds * value_std,
                               self.value_mean - self.config.value_outlier_stds * value_std,
                               os.path.join(output_dir, 'value-time.{}'.format(self.img_fmt)),
                               'High value threshold', 'Low value threshold',
                               'Value', 'Value')

        save_list_csv(self.low_values, os.path.join(output_dir, 'low-values.csv'))
        save_list_csv(self.high_values, os.path.join(output_dir, 'high-values.csv'))

    def get_element_datapoint(self, datapoint):
        value_std = self.values.std(0)
        value = datapoint.value
        return 'high-value' if value >= self.value_mean + self.config.value_outlier_stds * value_std else \
                   'low-value' if value <= self.value_mean - self.config.value_outlier_stds * value_std else '', \
               value

    def get_element_time(self, t):
        return 'high-value' if t in self.high_values else \
                   'low-value' if t in self.low_values else '', \
               self.values[t]
