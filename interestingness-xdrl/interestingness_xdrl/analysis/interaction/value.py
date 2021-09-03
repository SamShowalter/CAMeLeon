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

    def __init__(self, data, analysis_config, img_fmt, tag = "value"):
        """
        Creates a new analysis.
        :param list[InteractionDataPoint] data: the interaction data collected to be analyzed.
        :param AnalysisConfiguration analysis_config: the analysis configuration containing the necessary parameters.
        :param str img_fmt: the format of the images to be saved.
        """
        super().__init__(data, analysis_config, img_fmt, tag =tag)
        # derived data
        self.values = np.zeros(len(data))  # timestep-indexed state value
        self.low_values = []  # timesteps where values are much higher than average
        self.high_values = []  # timesteps where values are much lower than average
        self.value_mean = 0.  # mean overall value across all timesteps
        self.value_std = 0.  # std dev of mean overall value across all timesteps

    def analyze(self, output_dir):
        logging.info('Analyzing value...')

        # gets mean values and outliers
        self.values = np.array([self._get_datapoint_value(datapoint) for datapoint in self.data])
        self.value_mean = np.mean(self.values, axis=0)
        self.value_std = np.std(self.values, axis=0)
        self._get_actual_value_to_go()
        all_outliers = set(get_outliers_dist_mean(self.values, self.config.value_outlier_stds, True, True))

        # registers outliers
        self.low_values = []
        self.high_values = []
        for t in range(1, len(self.data) - 1):
            # tests for above outlier
            if t in all_outliers and self.values[t] > self.value_mean and \
                    self.values[t - 1] <= self.values[t] > self.values[t + 1]:
                self.data[t].interestingness.add_metric(self.tag,
                                               'high_value',
                                               1)
                self.high_values.append((t,self.data[t].rollout_name,self.data[t].rollout_timestep))
            # tests for below outlier
            elif t in all_outliers and self.values[t] < self.value_mean and \
                    self.values[t - 1] >= self.values[t] < self.values[t + 1]:
                self.data[t].interestingness.add_metric(self.tag,
                                               'low_value',
                                               1)
                self.low_values.append((t,self.data[t].rollout_name,self.data[t].rollout_timestep))

        # sorts outliers
        self.high_values.sort(key=lambda i: self.values[i[0]], reverse=True)
        self.low_values.sort(key=lambda i: self.values[i[0]])
        self.high_values_lkp = [d[0] for d in self.high_values]
        self.low_values_lkp = [d[0] for d in self.low_values]

        # Rebuild self.data dictionary
        data_dict = self._group_data_by_episode(self.data, outdir = output_dir, make_dirs = True)

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
        self.save(os.path.join(output_dir, f'{self.tag}.pkl.gz'))

        value_std = self.values.std(0)

        # Save time dataset
        self._save_time_dataset_csv(self.data, ["value","actual_value_to_go","high_value","low_value"],
                                    os.path.join(output_dir, 'value-time'),
                                    default = 0)

        subtitles = [d.rollout_name for d in self.data]
        self._plot_elements_sp(self.values,
                               self.value_mean + self.config.value_outlier_stds * value_std,
                               self.value_mean - self.config.value_outlier_stds * value_std,
                               output_dir, 'value-time',
                               'High value threshold', 'Low value threshold',
                               'Value', 'Value',subtitles = subtitles)


        self._plot_elements_separate("value",
                               self.value_mean + self.config.value_outlier_stds * value_std,
                               self.value_mean - self.config.value_outlier_stds * value_std,
                               output_dir, "value-time",
                               'High value threshold', 'Low value threshold',
                               'Value', 'Value')

        return self.data

    def get_element_datapoint(self, datapoint):
        value_std = self.values.std(0)
        value = datapoint.value
        return 'high-value' if value >= self.value_mean + self.config.value_outlier_stds * value_std else \
                   'low-value' if value <= self.value_mean - self.config.value_outlier_stds * value_std else '', \
               value

    def get_element_time(self, t):
        return 'high-value' if t in self.high_values_lkp else \
                   'low-value' if t in self.low_values_lkp else '', \
               self.values[t]

    def _get_datapoint_value(self, datapoint):
        """TODO: Docstring for get_datapoint_reward.

        :datapoint: TODO
        :returns: TODO

        """
        datapoint.interestingness.add_metric(self.tag,
                                             'value',
                                             datapoint.value)
        return datapoint.value

    def _get_actual_value_to_go(self):
        """Get actual value to go

        """

        curr_episode = self.data[0].rollout_name
        curr_value_to_go = 0

        for i in reversed(list(range(len(self.data)))):
            datapoint =self.data[i]
            episode = datapoint.rollout_name

            if curr_episode != episode:
                curr_episode = episode
                curr_value_to_go = 0

            curr_value_to_go += datapoint.reward
            datapoint.interestingness.add_metric(self.tag,
                                                'actual_value_to_go',
                                                curr_value_to_go)

