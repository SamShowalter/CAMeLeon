import os
import sys
import logging
import numpy as np
from functools import partial
from collections import OrderedDict
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.analysis import AnalysisBase, AnalysisConfiguration
from interestingness_xdrl.util.math import get_distribution_evenness, save_list_csv

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'





class ExecutionCertaintyAnalysis(AnalysisBase):
    """
    Represents an analysis of the agent's history of action selection in the environment. Namely, it calculates the mean
    evenness of action executions, and the (un)certain execution situations, measured by how (un)even the action
    selection at that timestep was.
    """

    def __init__(self, data, analysis_config, img_fmt, tag = "execution_uncertainty"):
        """
        Creates a new analysis.
        :param Dict[Episode_id][List[InteractionDataPoint]] data: the interaction data collected to be analyzed.
        :param AnalysisConfiguration analysis_config: the analysis configuration containing the necessary parameters.
        :param str img_fmt: the format of the images to be saved.
        """
        super().__init__(data, analysis_config, img_fmt, tag = tag)

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
        # data = self._flatten_episodes()
        self.data = pool.map(self._get_action_dist_evenness,self.data)

        self.all_execution_divs = np.array([d.interestingness.get_metric(self.tag,
                                            'action_dist_evenness') for d in self.data])

        # registers mean exec outliers
        self.mean_execution_divs = self.all_execution_divs.mean(axis=1)
        self.mean_execution_div = self.mean_execution_divs.mean(0)
        self.uncertain_executions = []
        self.certain_executions = []

        for t in range(1, len(self.data) - 1):

            # tests for uncertain element (local maximum) - Provide full self.data point
            if self.mean_execution_divs[t] >= self.config.uncertain_exec_min_div and \
                    self.mean_execution_divs[t - 1] <= self.mean_execution_divs[t] > self.mean_execution_divs[t + 1]:
                self.data[t].interestingness.add_metric(self.tag,
                                               'uncertain_execution',
                                               1)
                self.uncertain_executions.append((t,self.data[t].rollout_name,self.data[t].rollout_timestep))

            # tests for certain element (local minimum) - Provide full self.data point
            elif self.mean_execution_divs[t] <= self.config.certain_exec_max_div and \
                    self.mean_execution_divs[t - 1] >= self.mean_execution_divs[t] < self.mean_execution_divs[t + 1]:
                self.data[t].interestingness.add_metric(self.tag,
                                               'certain_execution',
                                               1)
                self.certain_executions.append((t,self.data[t].rollout_name,self.data[t].rollout_timestep))


        # sorts outliers
        self.uncertain_executions.sort(key=lambda i: self.mean_execution_divs[i[0]], reverse=True)
        self.certain_executions.sort(key=lambda i: self.mean_execution_divs[i[0]])
        self.certain_execs_lkp = [d[0] for d in self.certain_executions]
        self.uncertain_execs_lkp = [d[0] for d in self.uncertain_executions]

        # gets mean action factor execution diversity
        self.mean_action_factor_divs = self.all_execution_divs.mean(axis=0)

        # Rebuild self.data dictionary
        data_dict = self._group_data_by_episode(self.data, outdir = output_dir, make_dirs = True)

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
        self.save(os.path.join(output_dir, f'{self.tag}.pkl.gz'))

        # Save time dataset
        self._save_time_dataset_csv(self.data, ["action_dist_evenness","mean_action_execution_div","uncertain_execution","certain_execution"],
                                    os.path.join(output_dir, 'mean-exec-div-time'),
                                    default = 0)

        subtitles = [d.rollout_name for d in self.data]
        self._plot_elements_sp(self.mean_execution_divs,
                               self.config.uncertain_exec_min_div, self.config.certain_exec_max_div,
                               output_dir,'mean-exec-div-time',
                               'Uncert. exec. threshold', 'Cert. exec. threshold',
                               'Action Execution Uncertainty', 'Norm. True Diversity',
                               subtitles = subtitles)

        # Plot mean execution_div
        self._plot_elements_separate("mean_action_execution_div",
                               self.config.uncertain_exec_min_div, self.config.certain_exec_max_div,output_dir,
                               'exec-div-time','Uncert. exec.', 'Cert. exec.',
                               'Action Execution Uncertainty', 'Norm. True Diversity')

        self._plot_action_factor_divs(
            self.mean_action_factor_divs, os.path.join(output_dir, 'mean-action-divs.{}'.format(self.img_fmt)),
            'Mean Action-Factor Execution Uncertainty', 'Norm. True Diversity')

        self._write_tuple_list_csv(self.certain_executions,
                                   ['list_index','rollout_name','rollout_timestep'],
                                   os.path.join(output_dir, 'certain-executions'))
        self._write_tuple_list_csv(self.uncertain_executions,
                                   ['list_index','rollout_name','rollout_timestep'],
                                   os.path.join(output_dir, 'uncertain-executions'))

        if self.all_execution_divs.shape[1] > 1:
            for d in self.data:
                if d.interestingness.get_metric(self.tag,'certain_execution', default = 0):
                    self._plot_timestep_action_factor_divs(d,'action_dist_evenness',
                        output_dir,'cert-exec-action-divs',
                        'Mean Action-Factor Execution Uncertainty', 'Norm. True Diversity')
                elif d.interestingness.get_metric(self.tag,'uncertain_execution', default = 0):
                    self._plot_timestep_action_factor_divs(d,'action_dist_evenness',
                        output_dir,'uncert-exec-action-divs',
                        'Mean Action-Factor Execution Uncertainty', 'Norm. True Diversity')

        return self.data


    def get_element_datapoint(self, datapoint):
        mean_execution_div = datapoint.interestingness.get_metric(self.tag, "mean_action_execution_div")
        return 'cert-exec' if mean_execution_div <= self.config.certain_exec_max_div else \
                   'uncert-exec' if mean_execution_div >= self.config.uncertain_exec_min_div else '', \
               mean_execution_div

    def get_element_time(self, t):
        return 'cert-exec' if t in self.certain_execs_lkp else \
                   'uncert-exec' if t in self.uncertain_execs_lkp else '', \
               self.mean_execution_divs[t]

    def _get_action_dist_evenness(self, datapoint):
        """
        Gets the evenness/true diversity associated with each given distribution over actions for each factor.
        :param list[np.ndarray] action_probs: the probability distribution over actions for each action factor.
        :rtype: list[float]
        :return: a list with the evenness associated with each action-factor distribution.
        """
        # gets mean evenness (diversity) of distribution over all action factors
        action_dist_evenness = [get_distribution_evenness(dist) for dist in datapoint.action_probs]
        datapoint.interestingness.add_metric(self.tag,
                                                          'action_dist_evenness',
                                                          action_dist_evenness)

        mean_action_execution_div = np.mean(action_dist_evenness)
        datapoint.interestingness.add_metric(self.tag,
                                                          'mean_action_execution_div',
                                                          mean_action_execution_div)

        return datapoint

