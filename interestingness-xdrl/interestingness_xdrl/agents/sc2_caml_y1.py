import copy
import importlib
import itertools
import numpy as np
import tensorflow as tf
import sc2scenarios
from pysc2.env.environment import TimeStep, StepType
from pysc2.lib.features import PlayerRelative, AgentInterfaceFormat
from pysc2.lib.units import get_unit_type
from sc2scenarios.scenarios.assault.common import normalize_unit_type
from tqdm import tqdm
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.agents import Agent

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class SC2CAMLY1Agent(Agent):
    """
    Represents a CAML Y1 agent that behaves according to a reaver deep RL policy model.
    """

    def __init__(self, aif, seed, sc2data_root, task_module, environment, policy):
        """
        Creates a new reaver agent / actor by loading the TF policy model from a results directory.
        :param AgentInterfaceFormat aif: the agent interface format needed to define the state-action space.
        :param int seed: the seed used to initialize *both* tensorflow and the SC2 environment.
        :param str sc2data_root: the root directory for input data (such as Reaver checkpoints).
        :param str task_module: the fully-qualified name of importable module defining task-creation API.
        :param str environment: the environment spec; must be a symbol defined in the `task_module`.
        :param str policy: the policy spec; must be a symbol defined in the `task_module`.
        """
        super().__init__(seed)

        # sets sc2 data root, following "sc2scenarios/sc2scenarios/bin/play_scenarios.py:148"
        if sc2data_root is not None:
            sc2scenarios.set_sc2data_root(sc2data_root)

        tf.set_random_seed(seed)

        # loads agent policy according to env spec, following "sc2scenarios.bin.play_scenarios.run_thread"
        task_module = importlib.import_module(task_module)
        env_spec = getattr(task_module, environment)
        env = _FakeEnv(env_spec, aif)
        policy_spec = getattr(task_module, policy)
        self.policy = task_module.create_policy(policy_spec, env, 'blue')
        self._actor = self.policy._actor

        # gets policy softmax distribution according to logits (based on reaver/models/sc2/policy.py)
        self._action_probs_vars = [self._actor.dists[i].probs for i in range(len(self._actor.dists))]

        # shortcuts to tensors
        self._obs_inputs = self._actor.obs_inputs
        self._sample = self._actor.sample
        self._value_tensor = self._actor.value
        self._logits = self._actor.logits

        # get labels for action factors
        self._action_factors = []
        for unit_type in env.blue_unit_types:
            self._action_factors.append(f'Action {unit_type.name}')
        for unit_type in env.blue_unit_types:
            self._action_factors.append(f'Location {unit_type.name}')
        for unit_type in env.blue_unit_types:
            self._action_factors.append(f'Target {unit_type.name}')

    def get_interaction_datapoints(self, observations, batch_size=-1):
        """
        Gets interaction datapoints given input observations summarizing the state of the agent's interaction with the
        environment.
        :param list[TimeStep] observations: the agent's observations containing different observation features.
        :param int batch_size: the batch size used to sample the policy. `-1` uses all observations at once.
        :rtype: list[InteractionDataPoint]
        :return: a list with the agent's interaction datapoints given the observations.
        """

        # converts observations and create batch, following "sc2scenarios.reaver.policy.ReaverPolicy.step"
        rewards = []
        all_obs = []
        for obs in observations:
            o, r, _ = self.policy.obs_filter([obs])
            all_obs.append(o[0])
            rewards.append(r)
        all_obs = np.array(all_obs)
        obs_batches = np.split(all_obs, np.arange(batch_size, len(all_obs), batch_size))

        # gets policy data, following "sc2scenarios.reaver.policy.ReaverPolicy.step"
        actor_data = [self._sample, self._value_tensor, self._logits, self._action_probs_vars]
        datapoints = []

        i = 0
        for obs_batch in tqdm(obs_batches, 'Getting interaction data from batches'):
            actions, values, logits, probs = self._actor.sess_mgr.run(actor_data, self._obs_inputs, [obs_batch])
            for b in range(len(values)):
                # gets interaction data, converting back from batch format
                observation = obs_batch[b]
                reward = rewards[i + b]
                new_episode = observations[i + b].step_type == StepType.FIRST
                value = values[b]
                action_probs = [p for f_prob in probs for p in f_prob[b]]
                action_values = [l for f_log in logits for l in f_log[b]]
                # this is the action the agent would do based on the policy
                action = np.array([factor[b] for factor in actions]).flatten()

                datapoints.append(InteractionDataPoint(
                    observation, action, reward, value,
                    action_values, action_probs, self._action_factors, new_episode))

            i += len(values)

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


class _FakeEnv(object):
    """
    Fake methods and properties of 'sc2scenarios.scenarios.assault.common.AssaultTaskEnvironment', to be used by
    the reaver policy.
    """

    def __init__(self, env_spec, aif):
        # following "sc2scenarios/sc2scenarios/scenarios/assault/common.py:448"
        self.red_unit_types = [normalize_unit_type(t) for t in env_spec['arguments']['red_unit_types']]
        self.blue_unit_types = [normalize_unit_type(t) for t in env_spec['arguments']['blue_unit_types']]
        self._unit_types = {
            "red": self.red_unit_types,
            "blue": self.blue_unit_types
        }
        self.interface_formats = [aif]

    def player_unit_types(self, player_name):
        # copy of "sc2scenarios.scenarios.assault.common.AssaultTaskEnvironment.player_unit_types"
        return self._unit_types[player_name]

    def enemy_of(self, player_name):
        # copy of "sc2scenarios.scenarios.assault.common.AssaultTaskEnvironment.enemy_of"
        if player_name == "blue":
            return "red"
        elif player_name == "red":
            return "blue"
        else:
            raise ValueError("player_name = '{}'".format(player_name))

    def grid_locations(self):
        # copy of "sc2scenarios.scenarios.assault.common.AssaultTaskEnvironment.grid_locations"
        yield from (f + r for (f, r) in itertools.product(
            map(chr, range(ord('a'), ord('k') + 1)), map(str, range(1, 8 + 1))))


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
