#################################################################################
#
#             Project Title:  Callbacks to Collect information from Rllib
#             Author:         Sam Showalter
#             Date:           07-21-2021
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################


from typing import Dict
import tensorflow as tf
import argparse
import sys
import numpy as np
import os

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy

# Custom imports
from cameleon.policy_extractors.rllib import build_rllib_policy_extractor
from cameleon.utils.general import _write_pkl, _write_hkl,_read_pkl, _read_hkl

#################################################################################
# RLlib Callback Class for Cameleon
#################################################################################


class RLlibCallbacks(DefaultCallbacks):
    """
    Callbacks to store information from Cameleon

    """

    def __init__(self, outdir,
                 model,
                 framework,
                 no_frame = True,
                 use_hickle = True):

        DefaultCallbacks.__init__(self)

        self.outdir = outdir
        self.model = model
        self.framework = framework
        self.no_frame = no_frame
        self.use_hicke = use_hickle
        self.write_compressed = _write_hkl if use_hickle else _write_pkl
        self.read_compressed = _read_hkl if use_hickle else _write_pkl
        self.ext = "hkl" if use_hickle else "pkl"
        # This needs to be -1 because of weird issues with
        # the parallel processing of rollouts
        self.episode_num = -1
        self.step_num = 0
        self.reward_total = 0
        self.last_done = False
        self.episode = {}

    def write_episode(self):
        """Write the episode

        """
        # and len(self.rollout) > 0
        self.episode_id = "{}_ep{}_s{}_r{}.{}".format(os.getpid(),
                                         self.episode_num,
                                         self.step_num,
                                         str(round(self.reward_total)).replace("-","n"),
                                         self.ext)

        if self.outdir and len(self.episode) > 0:
            self.write_compressed(self.episode,
                  self.outdir + self.episode_id)

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, \
            "ERROR: `on_episode_start()` callback should be called right " \
            "after env reset!"
        # print("episode {} (env-idx={}) started.".format(
        #     episode.episode_id, env_index))
        # print(episode.episode_id)
        self.episode = {}
        self.step_num = 0
        self.reward_total = 0
        self.episode_num += 1


    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       episode: MultiAgentEpisode, env_index: int, **kwargs):

        # Make sure this episode is ongoing.
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"

        pe = build_rllib_policy_extractor(
                                  self.model,
                                  episode,
                                  worker,
                                  framework = self.framework,
                                  env = base_env)

        # Get reward for total reward
        reward = pe.get_last_reward()
        self.reward_total += reward
        self.last_done = pe.get_last_done()
        info =pe.get_last_info()
        if self.no_frame:
            del info['frame']

        # Add information for the episode
        self.episode[self.step_num] = {
            "observation": pe.get_last_observation(),
            "action":pe.get_last_action(),
            "reward":reward,
            "done": self.last_done,
            "info":info,
            "pi_info":pe.get_last_pi_info(),
            "value_function":pe.get_value_function_estimate(),
            "action_dist" : pe.get_action_dist(),
            "q_function_dist": pe.get_q_function_dist(),
            "action_logits":pe.get_action_logits()
        }


        self.step_num += 1

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        # Write episode
        self.write_episode()


#################################################################################
#   Main Method
#################################################################################
