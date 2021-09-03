import numpy as np
from pysc2.env.environment import TimeStep
from pysc2.lib.actions import FunctionCall
from interestingness_xdrl.agents import Agent

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class _Actor(object):
    def __init__(self, agent, converter, rwd_dim):
        self._agent = agent
        self._converter = converter
        self._rwd_dim = rwd_dim
        self._cur_ag_obs = None

    def set_cur_agent_obs(self, ag_obs):
        self._cur_ag_obs = ag_obs

    def act(self, observations):
        # check batch size, should be [num_nets, batch_size, obs_dim]
        if len(observations.shape) == 3:
            num_nets = observations.shape[0]
        else:
            num_nets = 1
            observations = [observations]

        # gets actions by converting observations and getting corresponding actions from agent (policy)
        dummy_rwds = np.zeros((len(self._cur_ag_obs), self._rwd_dim))
        actions = []
        for i in range(num_nets):
            obs = self._converter.to_agent_observations(observations[i], dummy_rwds, self._cur_ag_obs)
            actions.append([idp.action for idp in self._agent.get_interaction_datapoints(obs)])
        return np.array(actions)


def get_agent_rollouts(pe_model, agent, converter, agent_obs, agent_actions, rwd_dim,
                       batch_size=32, rollout_length=1, deterministic=True):
    """
    Gets prediction rollouts for a given set of observations according to an agent's policy. Returns both the
    sampled observation/reward, and their mean and variance values.
    :param BNN pe_model: the probabilistic models ensemble.
    :param Agent agent: the agent used to select actions during the rollout.
    :param SampleConverter converter: the converter from agent observations to latent representations.
    :param list[TimeStep] agent_obs: the list of agent's observations.
    :param list[list[FunctionCall]] agent_actions: the list of executed actions.
    :param int rwd_dim: the dimensionality of the rewards.
    :param int batch_size: the batch size to get the predictive rollouts.
    :param int rollout_length: the length of the rollout
    :param bool deterministic: whether to make deterministic (`True`) or stochastic (`False`) samples.
    :rtype: (np.ndarray, np.ndarray)
    :return: a tuple containing the reward and observation rollouts, of shape
            (rollout_length, sample+mean+var (3), num_nets, num_steps, rwd/obs_dim)
    """
    # actor is a wrapper that converts latent obs to agent obs and then samples the agent's actions
    actor = _Actor(agent, converter, rwd_dim)

    # gets rollouts
    rwd_rollouts = []
    obs_rollouts = []
    for i in range(0, len(agent_obs), batch_size):
        # get batch of latent observations
        end = min(len(agent_obs), i + batch_size)
        observations, actions, _ = converter.from_agent_observations(agent_obs[i:end], agent_actions[i:end])

        actor.set_cur_agent_obs(agent_obs[i:end])

        # get predictive rollout
        rwd_rollout, obs_rollout = pe_model.get_rollouts(
            observations, actions, actor.act, rollout_length, deterministic)
        rwd_rollouts.append(rwd_rollout)
        obs_rollouts.append(obs_rollout)

    # gets rollouts in the shape (rollout_length, sample+mean+var (3), num_nets, num_steps, rwd/obs_dim)
    rwd_rollouts = np.concatenate(rwd_rollouts, axis=3)
    obs_rollouts = np.concatenate(obs_rollouts, axis=3)
    return obs_rollouts, rwd_rollouts
