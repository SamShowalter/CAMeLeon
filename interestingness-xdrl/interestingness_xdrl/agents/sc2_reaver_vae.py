import numpy as np
from pysc2.env.environment import TimeStep
from pysc2.lib.actions import FunctionCall
from imago.models.sequential.pets.converters.reaver_vae_converter import ReaverVAESampleConverter
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.agents import Agent
from interestingness_xdrl.agents.sc2_reaver import SC2ReaverAgent

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class SC2ReaverVAEAgent(Agent):
    """
    Represents a reaver-based SC2 agent that "sees" the world through the lens of a VAE. This agent is just intended
    for purposes of testing the VAE reconstructions.
    """

    def __init__(self, agent, converter):
        """
        Creates a new agent.
        :param SC2ReaverAgent agent: the "real"/base reaver agent.
        :param ReaverVAESampleConverter converter: the converter to convert samples to/from VAE latent obs.
        """
        super().__init__(0)
        self._agent = agent
        self._converter = converter

        # different types of observations
        self._prev_agent_obs = None
        self._prev_z = None
        self._prev_z_mu = None
        self._prev_z_logvar = None

    def get_interaction_datapoints(self, observations):
        """
        Gets interaction datapoints given input observations summarizing the state of the agent's interaction with the
        environment.
        :param (list[TimeStep], list[list[FunctionCall]]) observations: a tuple containing the corresponding agent
        observations and the agent executed actions.
        :rtype: list[InteractionDataPoint]
        :return: a list with the agent's interaction datapoints given the observations.
        """
        # encodes observations and samples the VAE model's output
        agent_obs, agent_actions = observations
        self._prev_z, actions, rwds = self._converter.from_agent_observations(agent_obs, agent_actions)
        self._prev_agent_obs = self._converter.to_agent_observations(self._prev_z, rwds, agent_obs)
        self._prev_z_mu = self._converter.z_mu
        self._prev_z_logvar = self._converter.z_logvar

        # return decoded observations
        return self._agent.get_interaction_datapoints(self._prev_agent_obs)

    @property
    def agent_observations(self):
        """
        Gets the latest agent observations perceived by the agent after encoding/decoding from the VAE model.
        :rtype: list[TimeStep]
        :return: the latest agent observations.
        """
        return self._prev_agent_obs

    @property
    def latent_observations(self):
        """
        Gets the latest observation as sampled/encoded by the VAE.
        :rtype: np.ndarray
        :return: the latest latent observation.
        """
        return self._prev_z

    @property
    def latent_means(self):
        """
        Gets the latest observation mean as encoded by the VAE.
        :rtype: np.ndarray
        :return: the latest latent observation mean.
        """
        return self._prev_z_mu

    @property
    def latent_log_vars(self):
        """
        Gets the latest observation log-variance as encoded by the VAE.
        :rtype: np.ndarray
        :return: the latest latent observation log-variance.
        """
        return self._prev_z_logvar

    def get_counterfactuals(self, datapoint):
        return self._agent.get_counterfactuals(datapoint)
