import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from pysc2.env.environment import TimeStep, StepType
from imago.models.sequential.pets.bnn import BNN
from imago.models.behav.reaver_behav import REAVER_ACT_SPECS
from imago.models.sequential.pets.converters.rb_vae_converter import RBVAESampleConverter
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.agents import Agent

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

# TODO currently hardcoded, but perturbations should occur along feature categories
PERTURB_ALPHAS = [5, 10, 20]


class SC2RBVAEBNNAgent(Agent):
    """
    Represents an SC2 agent that collects interaction data from a "reaver behavior model" and generates next-state and
    reward predictions using a probabilistic ensemble (PE) model trained with a ConvVAE's latent representation.
    """

    def __init__(self, converter, pe_model, seed, horizon=16, deterministic=True):
        """
        Creates a new reaver agent / actor by loading the TF policy model from a results directory.
        :param RBVAESampleConverter converter: the converter to convert samples to/from VAE latent obs.
        :param BNN pe_model: the probabilistic ensemble model trained using the VAE's latent representation.
        :param int seed: the seed used to initialize *both* tensorflow and the SC2 environment.
        which to load the reaver policy.
        :param int horizon: the planning horizon from which to get the predicted next states and rewards.
        :param bool deterministic: whether to make deterministic (`True`) or stochastic (`False`) predictions.
        """
        super().__init__(seed)

        self._pe_model = pe_model
        self._converter = converter
        self._horizon = horizon
        self._deterministic = deterministic

        self._action_factors = [act.replace('_', ' ') for act, _ in REAVER_ACT_SPECS]
        # mimic reaver policy at reaver.models.base.policy.MultiPolicy.make_dist
        self._action_probs = [tf.placeholder(tf.float32, [None, np.prod(act_shape)], name)
                              for name, act_shape in REAVER_ACT_SPECS]
        self._action_samples = [tfp.distributions.Categorical(probs=act_probs).sample()
                                for act_probs in self._action_probs]
        self._rollout_ag_obs = []  # used for rollout prediction
        self._prev_z = None
        self._prev_z_mu = None
        self._prev_z_logvar = None

        self.sess = tf.Session()

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

    def get_interaction_datapoints(self, observations, predictive=True):
        """
        Gets interaction datapoints given input observations summarizing the state of the agent's interaction with the
        environment.
        :param list[TimeStep] observations: the agent's observations containing different observation features.
        :param bool predictive: whether to include predictive / dynamics information in the datapoint.
        :rtype: list[InteractionDataPoint]
        :return: a list with the agent's interaction datapoints given the observations.
        """

        # encodes observations and samples the VAE model's output
        z, *_ = self._converter.from_agent_observations(observations, [[] for _ in range(len(observations))])
        self._prev_z = z
        self._prev_z_mu = self._converter.z_mu
        self._prev_z_logvar = self._converter.z_logvar

        # gets actions from VAE behavior model
        actions = [self.sess.run(self._action_samples[i], {self._action_probs[i]: self._converter.action_probs[i]})
                   for i in range(len(self._action_samples))]
        actions = np.array(actions).T

        # sample the predictive model
        if predictive and self._pe_model is not None:
            self._rollout_ag_obs = observations
            next_rwds, next_obs = self._pe_model.get_rollouts(z, actions, self._act, self._horizon, self._deterministic)
        else:
            predictive = False
            next_obs = next_rwds = np.zeros(len(observations))

        # gets interaction data, converting back from batch format
        datapoints = []
        for i in tqdm(range(len(observations)), 'Creating datapoints'):
            # gets policy data from reaver behavior model
            value = self._converter.value[i]
            action_probs = [prob[i] for prob in self._converter.action_probs]
            action_values = [np.zeros(np.prod(act_spec_shape))
                             for _, act_spec_shape in REAVER_ACT_SPECS]  # not available
            action = actions[i]  # this is the action the agent would do based on the policy
            new_episode = observations[i].step_type == StepType.FIRST

            # get predictive info # TODO create object that contains also datapoint info for future states
            # length, sample+mean+var (3), num_nets, batch_size, dim
            next_obs = next_obs[-1, ..., i, :] if predictive else None
            next_rwds = next_rwds[-1, ..., i, :] if predictive else None

            # creates datapoint
            datapoints.append(InteractionDataPoint(
                observations[i], action, observations[i].reward, value,
                action_values, action_probs, self._action_factors, new_episode,
                next_obs, next_rwds))

        return datapoints

    def get_counterfactuals(self, datapoint, perturb_dirs=None):
        """
        Gets interaction data for counterfactuals of the given datapoint. Counterfactuals are alternatives
        (similar/close) to the given datapoint's state/observation that potentially can lead to different behavior.
        :param InteractionDataPoint datapoint: the datapoint for which to compute counterfactuals.
        :param list[str] perturb_dirs: the directions along which to perturb the model. `None` perturbs along all
        available directions in the model.
        :rtype: list[tuple[InteractionDataPoint, str]]
        :return: a list of datapoints counterfactual to the given datapoint and corresponding description.
        """
        if perturb_dirs is None or len(perturb_dirs) == 0:
            perturb_dirs = list(self._converter.rb_vae_model.planes_by_label.keys())

        counter_obs = []
        counter_desc = []

        # for each perturbation direction
        for direction in perturb_dirs:
            for alpha in PERTURB_ALPHAS:
                # encodes observation and samples the VAE model's output
                z, *_ = self._converter.from_agent_observations([datapoint.observation], [[]])

                # perturb latent encoding along the direction / plane according to alpha step
                plane = self._converter.rb_vae_model.planes_by_label[direction]
                z_p = z + alpha * plane

                # convert back to agent observation
                orig_rwd = np.array([datapoint.reward / self._converter._max_reward])
                agent_obs = self._converter.to_agent_observations(z_p, orig_rwd, [datapoint.observation])[0]
                counter_obs.append(agent_obs)
                counter_desc.append('{}*{}'.format(alpha, direction.title()))

        # gets interaction datapoints (policy info) from all counterfactual observations
        counterfactuals = self.get_interaction_datapoints(counter_obs, False)

        return list(zip(counterfactuals, counter_desc))

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
            actions.append([idp.action for idp in self.get_interaction_datapoints(obs, False)])
        return np.array(actions)
