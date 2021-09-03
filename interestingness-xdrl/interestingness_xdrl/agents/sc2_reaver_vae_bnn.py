import tensorflow as tf
import numpy as np
from pysc2.env.environment import TimeStep
from pysc2.lib.actions import FunctionCall
from imago.models.sequential.pets.bnn import BNN
from imago.models.sequential.pets.converters.reaver_vae_converter import ReaverVAESampleConverter
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.agents.sc2_reaver import SC2ReaverAgent

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class SC2ReaverVAEBNNAgent(SC2ReaverAgent):
    """
    Represents a reaver SC2 agent that generates next-state and reward predictions using a probabilistic ensemble (PE)
    model trained with a ConvVAE's latent representation.
    """

    def __init__(self, converter, pe_model, aif, seed, results_dir, experiment, env_name, obs_features, action_set,
                 action_spatial_dim, safety, safe_distance, safe_flee_angle_range, safe_margin, safe_unsafe_steps,
                 gin_files, gin_bindings, agent_name='vtrace', gpu=False, horizon=16, deterministic=True):
        """
        Creates a new reaver agent / actor by loading the TF policy model from a results directory.
        :param ReaverVAESampleConverter converter: the converter to convert samples to/from VAE latent obs.
        :param BNN pe_model: the probabilistic ensemble model trained using the VAE's latent representation.
        :param AgentInterfaceFormat aif: the agent interface format needed to define the state-action space.
        :param int seed: the seed used to initialize *both* tensorflow and the SC2 environment.
        :param str results_dir: the path to the root directory containing the reaver policy to be loaded.
        :param str experiment: the name of the experiment, corresponding to the sub-directory inside results_dir from
        which to load the reaver policy.
        :param str env_name: either Gym env id or PySC2 map name to run the agent in.
        :param str obs_features: named observation feature space.
        :param str action_set: named action set.
        :param int action_spatial_dim: dimension of spatial actions.
        :param bool safety: whether to use a 'safe policy' for behavior.
        :param int safe_distance: safe policy will try to maintain this distance from enemies.
        :param float safe_flee_angle_range: range of perturbations to angle of flee direction (units of `pi` radians).
        :param int safe_margin: magnitude of evasive movement is  `safe_distance - current_distance + safe_margin`.
        :param list[int,int] safe_unsafe_steps: two-element list containing number of steps to execute, respectively,
        the safe and unsafe policies.
        :param list[str] gin_files: the list of path(s) to gin config(s).
        :param list[str] gin_bindings: the list og gin bindings to override config values.
        :param str agent_name: the name of the RL agent/class.
        :param bool gpu: whether to use the GPU to run the policy.
        :param int horizon: the planning horizon from which to get the predicted next states and rewards.
        :param bool deterministic: whether to make deterministic (`True`) or stochastic (`False`) predictions.
        """
        with tf.Graph().as_default():  # due to having initialized the PE model, we need a new graph
            super().__init__(aif, seed, results_dir, experiment, env_name, obs_features, action_set, action_spatial_dim,
                             safety, safe_distance, safe_flee_angle_range, safe_margin, safe_unsafe_steps, gin_files,
                             gin_bindings, agent_name, gpu)

        self._pe_model = pe_model
        self._converter = converter
        self._horizon = horizon
        self._deterministic = deterministic

        self._rollout_ag_obs = []  # used for rollout prediction

    def get_interaction_datapoints(self, observations):
        """
        Gets interaction datapoints given input observations summarizing the state of the agent's interaction with the
        environment.
        :param (list[TimeStep], list[list[FunctionCall]]) observations: a tuple containing the corresponding agent
        observations and the agent executed actions.
        :rtype: list[InteractionDataPoint]
        :return: a list with the agent's interaction datapoints given the observations.
        """
        agent_obs, agent_actions = observations

        # gets datapoints' information from policy
        datapoints = super().get_interaction_datapoints(agent_obs)

        # encodes observations and samples the VAE model's output
        obs, actions, _ = self._converter.from_agent_observations(agent_obs, agent_actions)

        # sample the predictive model and fills in prediction info
        self._rollout_ag_obs = agent_obs
        next_rwds, next_obs = self._pe_model.get_rollouts(obs, actions, self._act, self._horizon, self._deterministic)
        for i, datapoint in enumerate(datapoints):
            # shape: (length, sample+mean+var (3), num_nets, batch_size, rwd/obs_dim)
            datapoint.next_rwds = next_rwds[-1, ..., i, :]
            datapoint.next_obs = next_obs[-1, ..., i, :]

        return datapoints

    def _act(self, observations):
        # check batch size, should be [num_nets, batch_size, obs_dim]
        if len(observations.shape) == 3:
            num_nets = observations.shape[0]
        else:
            num_nets = 1
            observations = [observations]

        # gets actions by converting observations and getting corresponding actions from agent (policy)
        dummy_rwds = np.zeros((len(self._rollout_ag_obs), self._pe_model.rwd_dim))
        actions = []
        for i in range(num_nets):
            obs = self._converter.to_agent_observations(observations[i], dummy_rwds, self._rollout_ag_obs)
            actions.append([idp.action for idp in super().get_interaction_datapoints(obs)])
        return np.array(actions)
