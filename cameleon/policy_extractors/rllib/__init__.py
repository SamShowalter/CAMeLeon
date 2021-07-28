#################################################################################
#
#             Project Title:  Policy Extraction General Class for Cameleon, Abstracts
#                             policies away just like RLlib (wrapper class)
#             Author:         Sam Showalter
#             Date:           2021-07-22
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import numpy as np
import tensorflow as tf
import torch
from abc import ABC

from ray.rllib.env.base_env import _DUMMY_AGENT_ID

#################################################################################
# Build Rllib extractor
#################################################################################

def build_rllib_policy_extractor(model,
                                episode,
                                worker,
                                framework = "tf2",
                                env = None):
    """ Build a policy extractor for rllib just like how
    RLlib builds trainers (or my best attempt at that anyway)

    :model: str:            Type of policy used
    :episode: Episode:      current episode object from Rllib
    :worker: RolloutWorker: Current worker running episode
    :framework: str:        Either tf, tf2 or torch
    :env: GymVectorizedEnv: Environment for game, vectorized

    """
    from cameleon.policy_extractors.rllib.registry import EXTRACTORS

    return EXTRACTORS[model](model,
                             episode,
                             worker,
                             framework,
                             env)

#################################################################################
#   Build RLlib Policy Extractor
#################################################################################

class BaseRLlibPolicyExtractor(ABC):

    """Extracts Policy Information from RLlib"""

    def __init__(self,
                 model,
                 episode,
                 worker,
                 framework = "tf2",
                 env = None):
        """
        :model: str:            Type of policy used
        :episode: Episode:      current episode object from Rllib
        :worker: RolloutWorker: Current worker running episode
        :framework: str:        Either tf, tf2 or torch
        :env: GymVectorizedEnv: Environment for game, vectorized

        """
        self.episode = episode
        self.model = model
        self.env = env
        self.framework = framework

        self.policy = self._get_policy()
        self.observation = self._get_last_observation()
        self.model_out = self._get_model_out()
        self.logits = None

        assert self.framework in ['tf','tf2', 'torch'],\
            "ERROR: framework needs to be tf, tf2, or torch"

    def _expand_dims(self, tensor,
                     axis = 0):
        """Expand dims for tensor

        :tensor: Numpy, torch, or tf object

        """
        if isinstance(tensor,np.ndarray):
            return np.expand_dims(tensor, axis = axis)
        elif isinstance(tensor, torch.Tensor):
            return torch.unsqueeze(tensor, dim = axis)



    def _get_model_out(self):
        """Get the model output

        :returns: Model output logits

        """

        model_out = None
        observation = self.observation
        if self.framework == "torch":
            observation = torch.from_numpy(observation)

        model_out, _ = self.policy.model\
            .from_batch({"obs": self._expand_dims(observation,
                                            axis=0)
                        })

        return model_out

    def _get_policy(self):
        """Get the policy from the rollout
        worker

        :returns: agent policy

        """

        return self.episode.policy_map[
            self.episode.policy_for()]

    def _get_last_observation(self):
        """Get last observation

        :returns: Last observation seen
                  by agent

        """

        return self.episode.last_observation_for()

    def get_last_pi_info(self):
        """Get last pi info stats,
        which may vary by policy

        :returns: pi_info

        """
        return self.episode.last_pi_info_for()

    def get_last_observation(self):
        """
        Returns last observation, just keeps API
        consistent

        :returns: observation

        """
        return self.observation

    def get_last_action(self):
        """Get last action

        :returns: Action

        """

        return self.episode.last_action_for()


    def get_last_info(self):
        """Get last info

        :returns: info

        """
        return self.episode.last_info_for()


    def get_last_done(self):
        """Get last done
        TODO: This does not work!
        DO NOT USE THIS, it never shows
        done = True

        :returns: done

        """

        return self.episode.last_done_for()

    def get_last_reward(self, agent_id = _DUMMY_AGENT_ID):
        """Get last reward from observation

        """
        reward =self.episode._agent_reward_history[agent_id][-1]
        if isinstance(reward, tuple):
            reward = reward[0]

        return reward

    def get_advantage_estimate(self):
        """Get Advantage Estimate
        A(s) = Q(a,s) - V(s)
        if it is available.

        :returns: Advantage estimate if available

        """
        return None


    def get_value_function_estimate(self):
        """Get value function estimate for
        current state V(s)

        :returns: Value function estimate V(s)

        """

        return None

    def get_action_logits(self):
        """Get logits from execution
        given observation

        """
        return None

    def get_action_dist(self):
        """Get action distribution for
        the current state pi(a|s). Should
        work with either torch or tf2

        (NOT with tf1 and lazy eval, just
        save yourself some grief and use
        eager execution)

        :returns: Current action dist. pi(a|s)

        """

        return None

    def get_q_function_dist(self):
        """TODO: Docstring for get_q_function_dist.

        :arg1: TODO
        :returns: TODO

        """

        return None


#################################################################################
#   Main Method
#################################################################################




