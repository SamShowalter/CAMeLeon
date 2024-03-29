import os
import numpy as np
import logging
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.analysis import AnalysisBase, AnalysisConfiguration
from interestingness_xdrl.util.math import get_outliers_dist_mean, save_list_csv

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class RewardAnalysis(AnalysisBase):
    """
    Represents an analysis of the agent's history of rewards in the environment. Namely, it calculates the mean
    reward received, and the reward outlier situations, measured by how distant a reward is from its mean.
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
        self.all_rewards = np.zeros(len(data))  # timestep-indexed received reward
        self.low_rewards = []  # timesteps where rewards are much higher than average
        self.high_rewards = []  # timesteps where rewards are much lower than average
        self.mean_reward = 0.  # mean overall reward across all timesteps

    def analyze(self, output_dir):
        logging.info('Analyzing reward...')

        # gets mean rewards and outliers
        self.all_rewards = np.array([datapoint.reward for datapoint in self.data])
        self.mean_reward = self.all_rewards.mean(0)
        all_outliers = set(get_outliers_dist_mean(self.all_rewards, self.config.rwd_outlier_stds, True, True))

        # registers outliers
        self.low_rewards = []
        self.high_rewards = []
        for t in range(1, len(self.data) - 1):
            # tests for above outlier
            if t in all_outliers and self.all_rewards[t] > self.mean_reward and \
                    self.all_rewards[t - 1] <= self.all_rewards[t] > self.all_rewards[t + 1]:
                self.high_rewards.append(t)
            # tests for below outlier
            elif t in all_outliers and self.all_rewards[t] < self.mean_reward and \
                    self.all_rewards[t - 1] >= self.all_rewards[t] < self.all_rewards[t + 1]:
                self.low_rewards.append(t)

        # sorts outliers
        self.high_rewards.sort(key=lambda i: self.all_rewards[i], reverse=True)
        self.low_rewards.sort(key=lambda i: self.all_rewards[i])

        # summary of elements
        logging.info('Finished')
        logging.info('\tMean reward received over {} timesteps: {:.3f}'.format(
            len(self.data), self.mean_reward))
        logging.info('\tFound {} high rewards (stds={}): {}'.format(
            len(self.high_rewards), self.config.rwd_outlier_stds, self.high_rewards))
        logging.info('\tFound {} low rewards (stds={}): {}'.format(
            len(self.low_rewards), self.config.rwd_outlier_stds, self.low_rewards))

        logging.info('Saving report in {}...'.format(output_dir))

        # saves analysis report
        self.save(os.path.join(output_dir, 'reward.pkl.gz'))

        rwd_std = self.all_rewards.std(0)
        save_list_csv(list(self.all_rewards), os.path.join(output_dir, 'all_rewards.csv'))
        self._plot_elements(self.all_rewards, self.high_rewards, self.low_rewards,
                            self.mean_reward + self.config.rwd_outlier_stds * rwd_std,
                            self.mean_reward - self.config.rwd_outlier_stds * rwd_std,
                            os.path.join(output_dir, 'reward-time.{}'.format(self.img_fmt)),
                            'High reward threshold', 'Low reward threshold',
                            'Reward', 'Reward')

        save_list_csv(self.low_rewards, os.path.join(output_dir, 'low_rewards.csv'))
        save_list_csv(self.high_rewards, os.path.join(output_dir, 'high_rewards.csv'))

    def get_element_datapoint(self, datapoint):
        rwd_std = self.all_rewards.std(0)
        reward = datapoint.reward
        return 'high-reward' if reward >= self.mean_reward + self.config.rwd_outlier_stds * rwd_std else \
                   'low-reward' if reward <= self.mean_reward - self.config.rwd_outlier_stds * rwd_std else '', \
               reward

    def get_element_time(self, t):
        return 'high-reward' if t in self.high_rewards else \
                   'low-reward' if t in self.low_rewards else '', \
               self.all_rewards[t]
