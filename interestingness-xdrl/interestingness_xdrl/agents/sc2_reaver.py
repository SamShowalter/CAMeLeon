import copy
import itertools
import os
import gin
import logging
import reaver
import numpy as np
import tensorflow as tf
from pysc2.env import mock_sc2_env
from pysc2.env.environment import StepType, TimeStep
from pysc2.lib.features import PlayerRelative, AgentInterfaceFormat
from pysc2.lib.units import get_unit_type
from pysc2.lib.actions import FUNCTIONS, numpy_to_python
from reaver.agents import vtrace, composite
from reaver.agents.nop import NopAgent
from reaver.agents.sc2 import safe_policy
from reaver.envs.sc2 import ActionWrapper, ObservationWrapper, ReaverStateActionSpace
from reaver.utils import Experiment
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.agents import Agent

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

ACTION_SET = {
    'minimal': reaver.envs.sc2.MINIMAL_ACTION_SET,
    'screen': reaver.envs.sc2.MINIMAL_SCREEN_ACTION_SET}

OBS_FEATURES = {'minimal': reaver.envs.sc2.MINIMAL_FEATURES,
                'vae': reaver.envs.sc2.VAE_FEATURES,
                'vae2': reaver.envs.sc2.VAE_FEATURES_2,
                'screen': reaver.envs.sc2.MINIMAL_SCREEN_FEATURES}

AGENT_CLS = {
    'a2c': reaver.agents.AdvantageActorCriticAgent,
    "nop": reaver.agents.nop.NopAgent,
    'ppo': reaver.agents.ProximalPolicyOptimizationAgent,
    'vtrace': reaver.agents.vtrace.VTraceAgent
}


class SC2ReaverAgent(Agent):
    """
    Represents a reaver agent that behaves according to a deep RL policy model.
    """

    def __init__(self, aif, seed, results_dir, experiment, env_name, obs_features, action_set, action_spatial_dim,
                 safety, safe_distance, safe_flee_angle_range, safe_margin, safe_unsafe_steps,
                 gin_files, gin_bindings, agent_name='vtrace', gpu=False):
        """
        Creates a new reaver agent / actor by loading the TF policy model from a results directory.
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
        """
        super().__init__(seed)

        # configures gin
        _configure_gin(gin_files, gin_bindings, agent_name, gpu)

        # Check if results_dir/experiment exists.  If not, flag a warning
        policy_dir = os.path.join(results_dir, experiment)
        if not(os.path.exists(policy_dir)):
            logging.warning("SC2ReaverAgent WARNING Policy directory={} does not exist, STARTING WITH RANDOM AGENT".format(policy_dir))
        # create reaver experiment
        expt = Experiment(results_dir, env_name, agent_name, experiment)

        # build state-action space
        self._sa_space = _create_sa_space(obs_features, action_set, aif, action_spatial_dim)

        # prepare tensorflow
        tf.set_random_seed(seed)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess_mgr = reaver.utils.tensorflow.SessionManager(sess, expt.path, training_enabled=False)

        # creates agent / actor
        self._actor = _create_agent(agent_name, self._sa_space, sess_mgr,
                                    safety, safe_distance, safe_flee_angle_range, safe_margin, safe_unsafe_steps)

        # gets policy softmax distribution according to logits (based on reaver/models/sc2/policy.py)
        self._action_probs_vars = [self._actor.dists[i].probs for i in range(len(self._actor.dists))]

        # shortcuts
        self._session_mgr = self._actor.sess_mgr
        self._sample = self._actor.sample
        self._value_tensor = self._actor.value
        self._logits = self._actor.logits

        self._obs_inputs = self._actor.obs_inputs
        self._obs_wrapper = self._sa_space.obs_wrapper
        self._act_wrapper = self._sa_space.act_wrapper
        self._action_factors = [act.name.replace('_', ' ') for act in self._act_wrapper.spec.spaces]

    def get_interaction_datapoints(self, observations):
        """
        Gets interaction datapoints given input observations summarizing the state of the agent's interaction with the
        environment.
        :param list[TimeStep] observations: the agent's observations containing different observation features.
        :rtype: list[InteractionDataPoint]
        :return: a list with the agent's interaction datapoints given the observations.
        """

        # wraps observations
        w_obs = []
        rewards = []
        for obs in observations:
            # keep copy as reaver overwrites this when wrapping...
            available_actions = obs.observation['available_actions'].copy()
            o, r, _ = self._obs_wrapper([obs])
            w_obs.append(o)
            rewards.append(r)
            obs.observation['available_actions'] = available_actions  # restore

        # need to convert to batch format
        obs_batch = []
        for i in range(len(w_obs[0])):
            obs_batch.append(np.array([o[i] for o in w_obs]))

        # collects all policy information given the observation
        actor_data = [self._sample, self._value_tensor, self._logits, self._action_probs_vars]
        actions, values, logits, probs = self._session_mgr.run(actor_data, self._obs_inputs, obs_batch)

        # gets interaction data, converting back from batch format
        datapoints = []
        for i in range(len(values)):
            value = values[i]
            action_probs = [prob[i] for prob in probs]
            action_values = [val[i] for val in logits]
            action = [act[i] for act in actions]  # this is the action the agent would do based on the policy
            new_episode = observations[i].step_type == StepType.FIRST

            datapoints.append(InteractionDataPoint(
                observations[i], action, rewards[i], value,
                action_values, action_probs, self._action_factors, new_episode))
        return datapoints

    def get_counterfactuals(self, datapoint):
        # gets units by side and type
        all_units = [{} for _ in range(len(PlayerRelative))]
        for unit in datapoint.observation.observation.feature_units:
            if unit.unit_type not in all_units[unit.alliance]:
                all_units[unit.alliance][unit.unit_type] = []
            all_units[unit.alliance][unit.unit_type].append(unit)

        # gets counterfactual observations by removing different number of units for each side and type
        rng = np.random.RandomState(0)
        counter_obs = []
        counter_desc = []
        for rel in range(len(all_units)):
            for units in all_units[rel].values():
                for i in range(0, len(units)):
                    rem_units = [units[j] for j in rng.choice(np.arange(len(units)), i + 1, False)]
                    obs, desc = _get_remove_units_counterfactual(rem_units, datapoint.observation)
                    counter_obs.append(obs)
                    counter_desc.append(desc)

        # gets interaction datapoints (policy info) from all counterfactual observations
        counterfactuals = self.get_interaction_datapoints(counter_obs)

        return list(zip(counterfactuals, counter_desc))

    def _reverse_action(self, agent_action):

        # reverse of ActionWrapper.__call__() at reaver/envs/sc2/env.py
        action = [0] * len(self._act_wrapper.spec.spaces)

        # get action idx
        fn_id = np.where([f.id == agent_action.function for f in FUNCTIONS])[0]
        fn_id = fn_id[0] if len(fn_id) > 0 else 0
        fn_id_idx = self._act_wrapper.func_ids.index(fn_id)
        action[0] = fn_id_idx

        # get args
        for i, arg_type in enumerate(FUNCTIONS[fn_id].args):
            arg_name = arg_type.name
            if arg_name in self._act_wrapper.args:
                arg = agent_action.arguments[i]
                arg = numpy_to_python(arg)

                # adapted from FunctionCall.init_with_validation at pysc2/lib/actions.py
                if arg_type.values:  # enum
                    arg = list(arg_type.values).index(arg[0])
                elif len(arg) == 1 and isinstance(arg[0], int):  # Allow bare ints.
                    arg = arg[0]
                elif len(arg) > 1:
                    # pysc2 expects spatial coords, but we have flattened => attempt to fix
                    arg = [int(a / self._act_wrapper.spatial_scale - 0.5) for a in arg]
                    arg = arg[1] * self._act_wrapper.action_spatial_dim + arg[0]

                arg_idx = self._act_wrapper.args.index(arg_name) + 1
                action[arg_idx] = arg

        return action


def _get_remove_units_counterfactual(units, obs):
    # removes each unit and updates relevant layers
    obs = copy.deepcopy(obs)
    feature_screen = obs.observation.feature_screen
    desc = 'Removed: '
    for unit in units:
        desc += '{} {} at ({},{}); '.format(
            PlayerRelative(unit.alliance).name, get_unit_type(unit.unit_type).name, unit.x, unit.y)
        x, y = unit.x - unit.radius, unit.y - unit.radius
        u_size = unit.radius * 2 + 1
        for i, j in itertools.product(range(u_size), range(u_size)):
            x_i, y_j = x + i, y + j
            was_zero = feature_screen.unit_density[y_j, x_i] == 0
            feature_screen.unit_density[y_j, x_i] = max(0, feature_screen.unit_density[y_j, x_i] - 1)

            # if no more units here, zeros all other layers at this location
            if not was_zero and feature_screen.unit_density[y_j, x_i] == 0:
                for layer in feature_screen:
                    layer[y_j, x_i] = 0

    return obs, desc


def _configure_gin(gin_files, gin_bindings, agent_name, gpu):
    # (copied from 'reaver/run2.py')
    base_path = os.path.dirname(os.path.abspath(__file__))
    extra_gin_files = ['base.gin']
    extra_gin_files = [os.path.join(base_path, 'configs', agent_name, fl)
                       for fl in extra_gin_files]
    extra_gin_files = [f for f in extra_gin_files if os.path.isfile(f)]
    extra_gin_files += gin_files

    if (not gpu or 'CUDA_VISIBLE_DEVICES' not in os.environ
            or os.environ['CUDA_VISIBLE_DEVICES'].strip() == ""):
        gin_bindings.append('build_cnn_nature.data_format = \'channels_last\'')
        gin_bindings.append('build_fully_conv.data_format = \'channels_last\'')

    gin.parse_config_files_and_bindings(extra_gin_files, gin_bindings)


def _create_sa_space(obs_features, action_set, agent_interface_format, action_spatial_dim):
    # (adapted from 'reaver/envs/sc2/env.py')
    obs_features = OBS_FEATURES[obs_features]
    action_ids = ACTION_SET[action_set]
    obs_features = obs_features or reaver.envs.sc2.MINIMAL_FEATURES.copy()
    action_ids = action_ids or reaver.envs.sc2.MINIMAL_ACTION_SET[:]

    obs_spatial_dim = agent_interface_format.feature_dimensions.screen[0]
    action_spatial_dim = action_spatial_dim or obs_spatial_dim
    act_wrapper = ActionWrapper(action_spatial_dim, action_ids, obs_spatial_dim=obs_spatial_dim)
    obs_wrapper = ObservationWrapper(obs_features, action_ids)

    mock_env = mock_sc2_env.SC2TestEnv(agent_interface_format=[agent_interface_format])

    act_wrapper.make_spec(mock_env.action_spec())
    obs_wrapper.make_spec(mock_env.observation_spec())
    mock_env.close()

    return ReaverStateActionSpace(
        agent_interface_format, obs_spatial_dim, action_spatial_dim, obs_wrapper, act_wrapper)


def _create_agent(agent_name, state_action_space, sess_mgr,
                  safety, safe_distance, safe_flee_angle_range, safe_margin, safe_unsafe_steps):
    # (copied from 'reaver/run2.py')
    obs_spec = state_action_space.obs_wrapper.spec
    act_spec = state_action_space.act_wrapper.spec

    if agent_name == 'vtrace':
        actor = vtrace.Actor.create(
            reaver.models.sc2.fully_conv.build_fully_conv,
            reaver.models.sc2.policy.SC2MultiPolicy,
            state_action_space, sess_mgr)

        # behavior_actor = actor
        if safety:
            base_policy = safe_policy.AvoidEnemiesPolicy(
                obs_spatial_dim=state_action_space.obs_spatial_dim,
                safe_distance=safe_distance, safe_margin=safe_margin, flee_angle_range=safe_flee_angle_range)

            policies = [
                safe_policy.AvoidEnemiesPolicyActorCriticAdapter(
                    base_policy, state_action_space=state_action_space,
                    logit_wrapper=lambda x: vtrace.VTraceAgent.ExtraTrainingData(x)),
                actor
            ]

            if safe_unsafe_steps[0] == 'override':
                behavior_actor = composite.DecisionListPolicy(policies)
            else:
                steps = [int(x) for x in safe_unsafe_steps]
                behavior_actor = composite.RoundRobinPolicy(policies, steps)
        else:
            behavior_actor = actor

        vtrace.VTraceAgent(obs_spec, act_spec, actor, sess_mgr=sess_mgr)

    elif agent_name == 'nop':
        agent = NopAgent()
        behavior_actor = agent
    else:
        agent = AGENT_CLS[agent_name](obs_spec, act_spec, sess_mgr=sess_mgr, n_envs=1)
        behavior_actor = agent
    return behavior_actor
