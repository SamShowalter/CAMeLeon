import os
import logging
import numpy as np
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.analysis import AnalysisBase, AnalysisConfiguration
from interestingness_xdrl.util.math import save_list_csv

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def _get_action_max_differences(action_probs):
    """
    Gets the peak-to-peak (maximum minimum difference) associated with each given distribution over actions for each
    factor.
    :param list[np.ndarray] action_probs: the probability distribution over actions for each action factor.
    :rtype: list[float]
    :return: a list with the max. difference associated with each action-factor distribution.
    """
    # gets peak-to-peak of distribution over all action factors
    return [np.ptp(dist) for dist in action_probs]


class ExecutionValueAnalysis(AnalysisBase):
    """
    Represents an analysis of the agent's learned policy, calculating the mean peak-to-peak action probability. It
    extracts information on the states where this quantity is significantly higher or lower than others (outliers).
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
        data = [datapoint.action_probs for datapoint in self.data]
        self.all_execution_diffs = np.array(pool.map(_get_action_max_differences, data))

        # registers mean max diff outliers
        self.mean_execution_diffs = self.all_execution_diffs.mean(axis=1)
        self.mean_execution_diff = self.mean_execution_diffs.mean(0)
        self.high_exec_differences = []
        for t in range(1, len(self.data) - 1):
            # tests for high difference element (local maximum)
            if self.mean_execution_diffs[t] >= self.config.uncertain_exec_min_div and \
                    self.mean_execution_diffs[t - 1] <= self.mean_execution_diffs[t] > self.mean_execution_diffs[t + 1]:
                self.high_exec_differences.append(t)

        # sorts outliers
        self.high_exec_differences.sort(key=lambda i: self.mean_execution_diffs[i], reverse=True)

        # gets mean action factor max execution difference
        self.mean_action_factor_diffs = self.all_execution_diffs.mean(axis=0)

        # summary of elements
        logging.info('Finished')
        logging.info('\tMean action-execution differences (peak-to-peak) over {} timesteps: {:.3f}'.format(
            len(self.data), self.mean_execution_diff))
        logging.info('\tFound {} high execution differences (min diff={}): {}'.format(
            len(self.high_exec_differences), self.config.uncertain_exec_min_div, self.high_exec_differences))

        logging.info('Saving report in {}...'.format(output_dir))

        # saves analysis report
        self.save(os.path.join(output_dir, 'execution-value.pkl.gz'))
        np.savetxt(
            os.path.join(output_dir, 'all-execution-diffs.csv'), self.all_execution_diffs, '%s', ',', comments='')

        save_list_csv(self.mean_execution_diffs, os.path.join(output_dir, 'mean-exec-diff-time.csv'))
        self._plot_elements(self.mean_execution_diffs, self.high_exec_differences, [],
                            self.config.certain_exec_max_div, np.nan,
                            os.path.join(output_dir, 'mean-exec-diff-time.{}'.format(self.img_fmt)),
                            'High exec. diff. threshold', '',
                            'Action Execution Max. Difference', 'Max. Prob. Difference')

        save_list_csv(self.mean_action_factor_diffs, os.path.join(output_dir, 'mean-action-diffs.csv'))
        self._plot_action_factor_divs(
            self.mean_action_factor_diffs, os.path.join(output_dir, 'mean-action-diffs.{}'.format(self.img_fmt)),
            'Mean Action-Factor Execution Diff', 'Max. Prob. Difference')

        save_list_csv(self.high_exec_differences, os.path.join(output_dir, 'high-exec-differences.csv'))

        for t in self.high_exec_differences:
            self._plot_action_factor_divs(
                self.all_execution_diffs[t],
                os.path.join(output_dir, 'high-exec-{}-action-diffs.{}'.format(t, self.img_fmt)),
                'Mean Action-Factor Execution Diff', 'Max. Prob. Difference')

    def get_element_datapoint(self, datapoint):
        exec_diff = np.mean(_get_action_max_differences(datapoint.action_probs))
        return 'high-exec-diff' if exec_diff >= self.config.uncertain_exec_min_div else '', \
               exec_diff

    def get_element_time(self, t):
        return 'high-exec-diff' if t in self.high_exec_differences else '', \
               self.mean_execution_diffs[t]
