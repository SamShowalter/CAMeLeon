import os
import logging
import sys
import numpy as np
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.analysis import AnalysisBase, AnalysisConfiguration
from interestingness_xdrl.util.math import save_list_csv

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'




class ExecutionValueAnalysis(AnalysisBase):
    """
    Represents an analysis of the agent's learned policy, calculating the mean peak-to-peak action probability. It
    extracts information on the states where this quantity is significantly higher or lower than others (outliers).
    """

    def __init__(self, data, analysis_config, img_fmt,tag = "execution_value"):
        """
        Creates a new analysis.

        :param list[InteractionDataPoint] data: the interaction data collected to be analyzed.
        :param AnalysisConfiguration analysis_config: the analysis configuration containing the necessary parameters.
        :param str img_fmt: the format of the images to be saved.
        """
        super().__init__(data, analysis_config, img_fmt, tag = tag)

        # derived data
        self.all_execution_diffs = np.zeros(len(data))  # timestep-indexed max execution difference for all actions
        self.mean_execution_diffs = []  # timestep-indexed mean max. execution difference
        self.mean_action_factor_diffs = []  # mean max. execution difference for each action factor
        self.mean_execution_diff = 0.  # mean overall max. execution difference across all timesteps
        self.high_exec_differences = []  # timesteps where mean max. execution difference is above a threshold

    def analyze(self, output_dir):
        logging.info('Analyzing action execution value (maximum minimum differences)...')

        # prepares multiprocessing
        pool = self._get_mp_pool()

        # gets action-factor exec max differences for each timestep
        self.data = pool.map(self._get_action_max_differences, self.data)
        self.all_execution_diffs = np.array([d.interestingness.get_metric(self.tag,
                                            'action_max_min_diffs') for d in self.data])

        # registers mean max diff outliers
        self.mean_execution_diffs = self.all_execution_diffs.mean(axis=1)
        self.mean_execution_diff = self.mean_execution_diffs.mean(0)
        self.high_exec_differences = []
        for t in range(1, len(self.data) - 1):

            # tests for high difference element (local maximum)
            if self.mean_execution_diffs[t] >= self.config.uncertain_exec_min_div and \
                    self.mean_execution_diffs[t - 1] <= self.mean_execution_diffs[t] > self.mean_execution_diffs[t + 1]:
                self.data[t].interestingness.add_metric(self.tag,
                                               'high_exec_difference',
                                               1)
                self.high_exec_differences.append((t,self.data[t].rollout_name,self.data[t].rollout_timestep))

        # sorts outliers
        self.high_exec_differences.sort(key=lambda i: self.mean_execution_diffs[i[0]], reverse=True)
        self.high_exec_differences_lkp = [d[0] for d in self.high_exec_differences]

        # gets mean action factor max execution difference
        self.mean_action_factor_diffs = self.all_execution_diffs.mean(axis=0)

        # Rebuild self.data dictionary
        data_dict = self._group_data_by_episode(self.data, outdir = output_dir, make_dirs = True)

        # summary of elements
        logging.info('Finished')
        logging.info('\tMean action-execution differences (peak-to-peak) over {} timesteps: {:.3f}'.format(
            len(self.data), self.mean_execution_diff))
        logging.info('\tFound {} high execution differences (min diff={}): {}'.format(
            len(self.high_exec_differences), self.config.uncertain_exec_min_div, self.high_exec_differences))

        logging.info('Saving report in {}...'.format(output_dir))

        # saves analysis report
        self.save(os.path.join(output_dir, f'{self.tag}.pkl.gz'))

        # Save time dataset
        self._save_time_dataset_csv(self.data, ["action_max_min_diffs","mean_action_max_min_diff","action_mean_max_min_diff","high_exec_difference"],
                                    os.path.join(output_dir, 'mean-exec-value-diff-time'),
                                    default = 0)

        self._write_tuple_list_csv(self.high_exec_differences,
                                   ['list_index','rollout_name','rollout_timestep'],
                                   os.path.join(output_dir, 'high-exec-differences'))

        self._plot_elements(self.mean_execution_diffs, self.high_exec_differences_lkp, [],
                            self.config.uncertain_exec_min_div, np.nan,
                            os.path.join(output_dir, 'mean-exec-value-diff-time.{}'.format(self.img_fmt)),
                            'High exec. diff. threshold', '',
                            'Action Execution Max. Difference', 'Max. Prob. Difference')

        self._plot_elements_separate("mean_action_max_min_diff",
                            self.config.uncertain_exec_min_div, np.nan, output_dir,'exec-value-diff-time',
                            'High exec. diff. threshold', '',
                            'Action Execution Max. Difference', 'Max. Prob. Difference')

        self._plot_action_factor_divs(
            self.mean_action_factor_diffs, os.path.join(output_dir, 'mean-exec-value-diffs.{}'.format(self.img_fmt)),
            'Mean Action-Factor Execution Diff', 'Max. Prob. Difference')

        if self.all_execution_diffs.shape[1] > 1:
            for d in self.data:

                if d.interestingness.get_metric(self.tag,'high_exec_difference', default = 0):
                    self._plot_timestep_action_factor_divs(d,"action_max_min_diffs",
                        output_dir, 'high-exec-value-action-diffs',
                        'Mean Action-Factor Execution Diff', 'Max. Prob. Difference')
        # Return data
        return self.data

    def get_element_datapoint(self, datapoint):
        exec_diff = datapoint.interestingness.get_metric(self.tag, "mean_action_max_min_diff")
        return 'high-exec-diff' if exec_diff >= self.config.uncertain_exec_min_div else '', \
               exec_diff

    def get_element_time(self, t):
        return 'high-exec-diff' if t in self.high_exec_differences_lkp else '', \
               self.mean_execution_diffs[t]

    def _get_action_max_differences(self, datapoint):
        """
        Gets the peak-to-peak (maximum minimum difference) associated with each given distribution over actions for each
        factor.
        :param list[np.ndarray] action_probs: the probability distribution over actions for each action factor.
        :rtype: list[float]
        :return: a list with the max. difference associated with each action-factor distribution.
        """
        # gets mean evenness (diversity) of distribution over all action factors
        action_max_min_diffs = [np.ptp(dist) for dist in datapoint.action_probs]
        datapoint.interestingness.add_metric(self.tag,
                                                          'action_max_min_diffs',
                                                          action_max_min_diffs)

        mean_action_max_min_diff = np.mean(action_max_min_diffs)
        datapoint.interestingness.add_metric(self.tag,
                                                          'mean_action_max_min_diff',
                                                          mean_action_max_min_diff)

        return datapoint


