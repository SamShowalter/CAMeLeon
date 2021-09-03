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

    def __init__(self, data, analysis_config, img_fmt, tag = "action_value"):
        """
        Creates a new analysis.
        :param list[InteractionDataPoint] data: the interaction data collected to be analyzed.
        :param AnalysisConfiguration analysis_config: the analysis configuration containing the necessary parameters.
        :param str img_fmt: the format of the images to be saved.
        """
        super().__init__(data, analysis_config, img_fmt,tag=tag)
        # derived data
        self.all_action_diffs = np.zeros(len(data))  # timestep-indexed mean action-value differences (max-min)
        self.low_action_diffs = []  # timesteps where value diffs are much higher than average
        self.high_action_diffs = []  # timesteps where value diffs are much lower than average
        self.mean_diff = 0.  # mean overall action-value difference across all timesteps

    def analyze(self, output_dir):
        logging.info('Analyzing maximum value difference...')

        # gets mean ptps and outliers

        # Can convert for multiprocessing if needed
        self.all_action_diffs = np.array([self._get_action_diffs(datapoint) for datapoint in self.data])
        self.mean_diff = self.all_action_diffs.mean(0)
        all_outliers = set(get_outliers_dist_mean(self.all_action_diffs, self.config.value_outlier_stds, True, True))

        # registers outliers
        self.low_action_diffs = []
        self.high_action_diffs = []
        for t in range(1, len(self.data) - 1):
            # tests for above outlier
            if t in all_outliers and self.all_action_diffs[t] > self.mean_diff and \
                    self.all_action_diffs[t - 1] <= self.all_action_diffs[t] > self.all_action_diffs[t + 1]:
                self.data[t].interestingness.add_metric(self.tag,
                                               'high_action_diff',
                                               1)
                self.high_action_diffs.append((t,self.data[t].rollout_name,self.data[t].rollout_timestep))
            # tests for below outlier
            elif t in all_outliers and self.all_action_diffs[t] < self.mean_diff and \
                    self.all_action_diffs[t - 1] >= self.all_action_diffs[t] < self.all_action_diffs[t + 1]:
                self.data[t].interestingness.add_metric(self.tag,
                                               'low_action_diff',
                                               1)
                self.low_action_diffs.append((t,self.data[t].rollout_name,self.data[t].rollout_timestep))

        # sorts outliers
        self.high_action_diffs.sort(key=lambda i: self.all_action_diffs[i[0]], reverse=True)
        self.low_action_diffs.sort(key=lambda i: self.all_action_diffs[i[0]])
        self.high_action_diffs_lkp = [d[0] for d in self.high_action_diffs]
        self.low_action_diffs_lkp = [d[0] for d in self.low_action_diffs]

        # Rebuild self.data dictionary
        data_dict = self._group_data_by_episode(self.data, outdir = output_dir, make_dirs = True)

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
        self.save(os.path.join(output_dir, f'{self.tag}.pkl.gz'))

        value_std = self.all_action_diffs.std(0)

        # Save time dataset
        self._save_time_dataset_csv(self.data, ["action_value_max_min_diffs","mean_action_value_max_min_diff","high_action_diff","low_action_diff"],
                                    os.path.join(output_dir, 'mean-exec-value-diff-time'),
                                    default = 0)


        self._write_tuple_list_csv(self.high_action_diffs,
                                   ['list_index','rollout_name','rollout_timestep'],
                                   os.path.join(output_dir, 'high-action-diffs'))
        self._write_tuple_list_csv(self.low_action_diffs,
                                   ['list_index','rollout_name','rollout_timestep'],
                                   os.path.join(output_dir, 'low-action-diffs'))


        self._plot_elements(self.all_action_diffs, self.high_action_diffs_lkp, self.low_action_diffs_lkp,
                            self.mean_diff + self.config.value_outlier_stds * value_std,
                            self.mean_diff - self.config.value_outlier_stds * value_std,
                            os.path.join(output_dir, 'action-diff-time.{}'.format(self.img_fmt)),
                            'High action-diff. threshold', 'Low action-diff. threshold',
                            'Maximum Action-Value Difference', 'Action-Value Diff.')


        self._plot_elements_separate("mean_action_value_max_min_diff",
                            self.mean_diff + self.config.value_outlier_stds * value_std,
                            self.mean_diff - self.config.value_outlier_stds * value_std,
                            output_dir, "action-diff-time",
                            'High action-diff. threshold', 'Low action-diff. threshold',
                            'Maximum Action-Value Difference', 'Action-Value Diff.')

        self._write_tuple_list_csv(self.low_action_diffs,
                                   ['list_index','rollout_name','rollout_timestep'],
                                   os.path.join(output_dir, 'low-action-diffs'))
        self._write_tuple_list_csv(self.low_action_diffs,
                                   ['list_index','rollout_name','rollout_timestep'],
                                   os.path.join(output_dir, 'high-action-diffs'))

        return self.data

    def get_element_datapoint(self, datapoint):
        value_std = self.all_action_diffs.std(0)
        action_diff = np.mean([np.ptp(factor_values) for factor_values in datapoint.action_values])
        return 'high-action-diff' if action_diff >= self.mean_diff + self.config.value_outlier_stds * value_std else \
                   'low-action-diff' if action_diff <= self.mean_diff - self.config.value_outlier_stds * value_std \
                       else '', \
               action_diff

    def get_element_time(self, t):
        return 'high-action-diff' if t in self.high_action_diffs_lkp else \
                   'low-action-diff' if t in self.low_action_diffs_lkp else '', \
               self.all_action_diffs[t]

    def _get_action_diffs(self, datapoint):
        """TODO: Docstring for _get_action_diffs.

        :datapoint: TODO
        :returns: TODO

        """

        action_value_max_min_diffs = [np.ptp(dist) for dist in datapoint.action_values]
        datapoint.interestingness.add_metric(self.tag,
                                                          'action_value_max_min_diffs',
                                                          action_value_max_min_diffs)

        mean_action_value_max_min_diff = np.mean(action_value_max_min_diffs)
        datapoint.interestingness.add_metric(self.tag,
                                                          'mean_action_value_max_min_diff',
                                                          mean_action_value_max_min_diff)

        return mean_action_value_max_min_diff




