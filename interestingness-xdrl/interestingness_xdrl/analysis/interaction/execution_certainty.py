import os
import logging
import numpy as np
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.analysis import AnalysisBase, AnalysisConfiguration
from interestingness_xdrl.util.math import get_distribution_evenness, save_list_csv

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def _get_action_dist_evenness(action_probs):
    """
    Gets the evenness/true diversity associated with each given distribution over actions for each factor.
    :param list[np.ndarray] action_probs: the probability distribution over actions for each action factor.
    :rtype: list[float]
    :return: a list with the evenness associated with each action-factor distribution.
    """
    # gets mean evenness (diversity) of distribution over all action factors
    return [get_distribution_evenness(dist) for dist in action_probs]


class ExecutionCertaintyAnalysis(AnalysisBase):
    """
    Represents an analysis of the agent's history of action selection in the environment. Namely, it calculates the mean
    evenness of action executions, and the (un)certain execution situations, measured by how (un)even the action
    selection at that timestep was.
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
        self.all_execution_divs = np.zeros(len(data))  # timestep-indexed execution diversity for all actions
        self.mean_execution_divs = []  # timestep-indexed mean execution diversity
        self.mean_action_factor_divs = []  # mean execution diversity for each action factor
        self.mean_execution_div = 0.  # mean overall execution diversity across all timesteps
        self.uncertain_executions = []  # timesteps where execution diversity is above a threshold
        self.certain_executions = []  # timesteps where execution diversity is below a threshold

    def analyze(self, output_dir):
        logging.info('Analyzing action execution uncertainty...')

        # prepares multiprocessing
        pool = self._get_mp_pool()

        # gets action-factor exec diversities for each timestep
        data = [datapoint.action_probs for datapoint in self.data]
        self.all_execution_divs = np.array(pool.map(_get_action_dist_evenness, data))

        # registers mean exec outliers
        self.mean_execution_divs = self.all_execution_divs.mean(axis=1)
        self.mean_execution_div = self.mean_execution_divs.mean(0)
        self.uncertain_executions = []
        self.certain_executions = []
        for t in range(1, len(self.data) - 1):
            # tests for uncertain element (local maximum)
            if self.mean_execution_divs[t] >= self.config.uncertain_exec_min_div and \
                    self.mean_execution_divs[t - 1] <= self.mean_execution_divs[t] > self.mean_execution_divs[t + 1]:
                self.uncertain_executions.append(t)
            # tests for certain element (local minimum)
            elif self.mean_execution_divs[t] <= self.config.certain_exec_max_div and \
                    self.mean_execution_divs[t - 1] >= self.mean_execution_divs[t] < self.mean_execution_divs[t + 1]:
                self.certain_executions.append(t)

        # sorts outliers
        self.uncertain_executions.sort(key=lambda i: self.mean_execution_divs[i], reverse=True)
        self.certain_executions.sort(key=lambda i: self.mean_execution_divs[i])

        # gets mean action factor execution diversity
        self.mean_action_factor_divs = self.all_execution_divs.mean(axis=0)

        # summary of elements
        logging.info('Finished')
        logging.info('\tMean action-execution certainty over {} timesteps: {:.3f}'.format(
            len(self.data), self.mean_execution_div))
        logging.info('\tFound {} uncertain action executions (min div={}): {}'.format(
            len(self.uncertain_executions), self.config.uncertain_exec_min_div, self.uncertain_executions))
        logging.info('\tFound {} certain action executions (max div={}): {}'.format(
            len(self.certain_executions), self.config.certain_exec_max_div, self.certain_executions))

        logging.info('Saving report in {}...'.format(output_dir))

        # saves analysis report
        self.save(os.path.join(output_dir, 'execution-certainty.pkl.gz'))
        np.savetxt(os.path.join(output_dir, 'all-execution-divs.csv'), self.all_execution_divs, '%s', ',', comments='')

        self._save_time_dataset_csv(self.mean_execution_divs, 'Execution Uncertainty',
                                    os.path.join(output_dir, 'mean-exec-div-time.csv'))
        self._plot_elements_sp(self.mean_execution_divs,
                               self.config.uncertain_exec_min_div, self.config.certain_exec_max_div,
                               os.path.join(output_dir, 'mean-exec-div-time.{}'.format(self.img_fmt)),
                               'Uncert. exec. threshold', 'Cert. exec. threshold',
                               'Action Execution Uncertainty', 'Norm. True Diversity')

        save_list_csv(self.mean_action_factor_divs, os.path.join(output_dir, 'mean-action-divs.csv'))
        self._plot_action_factor_divs(
            self.mean_action_factor_divs, os.path.join(output_dir, 'mean-action-divs.{}'.format(self.img_fmt)),
            'Mean Action-Factor Execution Uncertainty', 'Norm. True Diversity')

        save_list_csv(self.certain_executions, os.path.join(output_dir, 'certain-executions.csv'))
        save_list_csv(self.uncertain_executions, os.path.join(output_dir, 'uncertain-executions.csv'))

        for t in self.certain_executions:
            self._plot_action_factor_divs(
                self.all_execution_divs[t],
                os.path.join(output_dir, 'cert-exec-{}-action-divs.{}'.format(t, self.img_fmt)),
                'Mean Action-Factor Execution Uncertainty', 'Norm. True Diversity')
        for t in self.uncertain_executions:
            self._plot_action_factor_divs(
                self.all_execution_divs[t],
                os.path.join(output_dir, 'uncert-exec-{}-action-divs.{}'.format(t, self.img_fmt)),
                'Mean Action-Factor Execution Uncertainty', 'Norm. True Diversity')

    def get_element_datapoint(self, datapoint):
        mean_execution_div = np.mean(_get_action_dist_evenness(datapoint.action_probs))
        return 'cert-exec' if mean_execution_div <= self.config.certain_exec_max_div else \
                   'uncert-exec' if mean_execution_div >= self.config.uncertain_exec_min_div else '', \
               mean_execution_div

    def get_element_time(self, t):
        return 'cert-exec' if t in self.certain_executions else \
                   'uncert-exec' if t in self.uncertain_executions else '', \
               self.mean_execution_divs[t]
