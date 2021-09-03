__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class InteractionDataPoint(object):
    """
    Represents an interaction data-point, i.e., a moment of the agent's interaction with an environment.
    """

    def __init__(self, obs, action, reward, value, action_values,
                 action_probs, action_factors, new_episode,
                 rollout_name, rollout_timestep, rollout_tag,
                 next_obs=None, next_rwds=None):
        """
        Creates a new interaction data-point.
        :param obs: the agent's observation containing different observation features.
        :param action: the action to execute given the current observation (for the different action factors).
        :param float reward: the reward associated with the observation.
        :param float value: the value associated with the observation.
        :param action_values: the value(s) associated with each action factor given the observation.
        :param action_probs: the probability distributions associated with each action factor given the observation.
        :param list[str] action_factors: the action factors' labels.
        :param bool new_episode: whether this datapoint marks the beginning of a new episode.
        :param str rollout_name: Name of the rollout datapoint came from
        :param list next_obs: the next-step observations as predicted by the agent, if available.
        :param list next_rwds: the next-step rewards as predicted by the agent, if available.
        """
        self.observation = obs
        self.action = action
        self.reward = reward
        self.value = value
        self.action_values = action_values
        self.action_probs = action_probs
        self.action_factors = action_factors
        self.new_episode = new_episode
        self.rollout_name = rollout_name
        self.rollout_timestep = rollout_timestep
        self.next_obs = next_obs
        self.next_rwds = next_rwds
        self.rollout_tag = rollout_tag
        self.interestingness = None

    def __str__(self):
        """String representation

        """
        return f"IDP(rollout: {self.rollout_name} | step: {self.rollout_timestep})"

    def __repr__(self):
        """String representation

        """
        return f"IDP(rollout: {self.rollout_name} | step: {self.rollout_timestep})"

