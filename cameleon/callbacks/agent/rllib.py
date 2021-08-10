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
import argparse
import sys
import numpy as np
import os
import hashlib


import tensorflow as tf
import torch

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


class RLlibIxdrlCallbacks(DefaultCallbacks):
    """
    Callbacks to store information from Cameleon

    """

    def __init__(self,
                 args,
                 config):

        DefaultCallbacks.__init__(self)

        self.args = args
        self.config = config

        self.outdir = args.writer_dir
        self.model = args.model_name
        self.framework = config['framework']
        self.no_frame = args.no_frame
        self.train_epochs = self._extract_train_epochs()
        self.use_hickle = args.use_hickle
        self.write_compressed = _write_hkl if self.use_hickle else _write_pkl
        self.read_compressed = _read_hkl if self.use_hickle else _write_pkl
        self.ext = "hkl" if self.use_hickle else "pkl"

        # This needs to be -1 because of weird issues with
        # the parallel processing of rollouts
        self.episode_num = -1
        self.step_num = 0
        self.reward_total = 0
        self.last_done = False
        self.episode = {}

    def _extract_train_epochs(self):
        """Extract training epochs from checkpoint
        file, if possible.

        """
        if not self.args.checkpoint_path:
            return 0
        else:
            try:
                return int(self.args.checkpoint_path.split("-")[-1])
            except:
                print("ERROR extracting training epochs from checkpoint file."
                      "Default filename has been altered. Defaulting to 0")
                return 0


    def write_episode(self):
        """Write the episode

        """
        obs_hash = hashlib.shake_256(str(self.first_obs).encode()).hexdigest(6)
        self.first_obs = None

        # and len(self.rollout) > 0
        self.episode_id = "{}_ep{}_s{}_r{}_pid{}-{}.{}".format(
                                         obs_hash,
                                         self.train_epochs,
                                         self.step_num,
                                         str(round(self.reward_total)).replace("-","n"),
                                         self.episode_num,
                                         os.getpid(),
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
        self.first_obs = None
        self.reward_total = 0
        self.episode_num += 1

        # Environment observation at start
        env = base_env.get_unwrapped()[0]
        obs = env.gen_obs()

        # THIS CAUSES ERRORS
        # obs = env.reset()

        # Get the last observation
        self.first_obs = obs

        # Add information for the episode
        self.episode[self.step_num] = {
            "observation": obs,
        }



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
                                  env = base_env,
                                  episode_start = False)

        # Get reward for total reward and obs
        reward = pe.get_last_reward()
        self.reward_total += reward
        self.last_done = pe.get_last_done()
        info =pe.get_last_info()
        obs =self.episode[self.step_num]['observation']


        if self.no_frame:
            info['frame'] = None

        # Add information for the episode
        self.episode[self.step_num] = {
            "observation": obs,
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

        # Add information for the next episode observation
        self.episode[self.step_num] = {
            "observation": episode.last_observation_for(),
        }

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        # Write episode and delete last observation
        # since there was no action taken, roll back steps
        del self.episode[self.step_num]
        # self.step_num -=1
        self.write_episode()

#######################################################################
# Main method
#######################################################################

