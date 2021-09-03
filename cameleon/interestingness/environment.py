#################################################################################
#
#             Project Title:  Port saved rollout data into interestingness
#                             environment object
#             Author:         Sam Showalter
#             Date:           2021-07-26
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import sys
import os
import glob
import re
from collections import OrderedDict
from tqdm import tqdm

sys.path.append("../interestingness-xdrl")
from interestingness_xdrl.environments import EnvironmentData, Environment

from cameleon.utils.general import _write_pkl, _read_pkl, _write_hkl, _read_hkl

#######################################################################
# Cameleon Interestingness Agent
#######################################################################

class CameleonInterestingnessEnvironment(Environment):

    """Cameleon Interestingness Environment"""

    def __init__(self,
                 rollout_dir,
                 env_name,
                 agent_name,
                 framework,
                 outdir = "data/interestingness/",
                 action_factors = ['left','right','up','down'],
                 rollout_regex =r'[a-z\d]*_cp(\d+)_s(\d+)_r*(.+).[ph]kl',
                 use_hickle = True,
                 seed = None):

        Environment.__init__(self, seed)
        self.outdir = outdir

        # Make if it does not exist
        if not os.path.exists(outdir):
            os.makedirs(self.outdir)

        self.write_compressed = _write_hkl if use_hickle else _write_pkl
        self.read_compressed = _read_hkl if use_hickle else _read_pkl
        self.ext = "hkl" if use_hickle else "pkl"
        self.rollout_dir = rollout_dir
        self.rollout_path_regex = rollout_regex
        self.env_name = env_name
        self.agent_name = agent_name
        self.framework = framework
        self.action_factors = action_factors
        self.last_frame = None
        self.episodes = OrderedDict()


    def _init_episode(self,
                      episode_id,
                      episode_filepath,
                      steps,
                      reward,
                      epochs):
        """Docstring for _init_episode.

        :episode_id: Int:       ID of episode
        :episode_filepath: Str: Episode path
        :steps: int:            Number of timesteps in episode
        :reward: float:         Total reward for episode
        :epochs: int:           Number of epochs model trained for before rollout

        :returns: Dict:         Subdictionary for specific episode

        """
        assert episode_id not in self.episodes,\
            "ERROR: Episode id collision - ID {}"\
            .format(episode_id)

        self.episodes[episode_id] = {"data":{},
                                     "steps":steps,
                                     "name":episode_id.replace(".{}".format(self.ext),""),
                                     "epochs_trained":epochs,
                                     "total_reward":reward,
                                     "filepath":episode_filepath}

        return self.episodes[episode_id]

    def _add_timestep(self, timestep,
                      timestep_data, episode):
        """Add timestep to episode
        artifact

        :timestep: Int:       Timestep
        :timestep_data: Dict: Timestep-specific data
        :episode: Dict:       subdictionary of self.episodes

        """
        t = timestep_data
        episode[timestep] = {
                                "frame":       t['info']['env'],
                                "observation": t['observation'],
                                "action":      t["action"],
                                "new_episode": (timestep == 0)
                            }


    def add_episode(self,
                      episode_data,
                      episode_id,
                      episode_filepath,
                      steps,
                      reward,
                      epochs):
        """Add episode data to environment store

        :episode_data:Dict:     Rollout data
        :episode_id: Int:       ID of episode
        :episode_filepath: Str: Episode path
        :steps: int:            Number of timesteps in episode
        :reward: float:         Total reward for episode
        :epochs: int:           Number of epochs model trained for before rollout

        """
        episode = self._init_episode(episode_id,
                                           episode_filepath,
                                           steps,
                                           reward,
                                           epochs)

        assert len(episode_data) == episode["steps"],\
            """ERROR: Episode metadata step and length mismatch:
            Step: {} - Length: {}""".format(episode["steps"],
                                            len(episode_data))

        episode_store = episode['data']
        for i in range(len(episode_data)):
            self._add_timestep(i,
                               episode_data[i],
                               episode_store)


    def _get_rollout_paths(self):
        """Get path to all rollout pickle
        files in directory

        :returns: list[str]: List of rollout files (recursive)

        """
        self.rollout_paths = glob.glob(self.rollout_dir + "/**/*.{}"\
                                       .format(self.ext),recursive=True)
        assert len(self.rollout_paths) > 0,\
            "ERROR: Rollout directory did not find any files"
        return self.rollout_paths

    def flatten_for_interestingness_v1(self):
        """Flatten data dictionary to maintain
        same data format as previously specified
        by original interestingness module

        :returns: EnvironmentData: All environment interaction data

        """
        frames = []
        observations = []
        actions = []
        new_episodes = []

        for e,v in self.episodes.items():
            for i in range(len(v['data'])):
                data = v['data'][i]

                frames.append(data['frame'])
                observations.append(data['observation'])
                actions.append(data['action'])
                new_episodes.append(data['new_episode'])

        return EnvironmentData(frames = frames,
                               observations = observations,
                               actions = actions,
                               new_episodes = new_episodes)


    def collect_all_data(self):
        """Load rollouts

        :returns: EnvironmentData: All environment interaction data

        """
        paths = self._get_rollout_paths()

        for i in  tqdm(range(len(paths))):
            rollout = paths[i]
            episode_id = rollout.split("/")[-1]
            matches = re.match(self.rollout_path_regex,episode_id)
            if matches:
                components = matches.groups()
                epochs = int(components[0])
                steps = int(components[1])
                total_reward = int(components[2].replace("n",'-'))

                episode_data = self.read_compressed(rollout)

                self.add_episode(episode_data,
                                             episode_id,
                                             rollout,
                                             steps,
                                             total_reward,
                                             epochs)

        self.out_root = '{}{}_{}_{}'.format(self.outdir,
                                self.agent_name,
                                self.framework,
                                self.env_name)

        # Return flattened EnvironmentData variable
        return self.flatten_for_interestingness_v1()

    def save(self):
        """Save self as a pickle file

        NOTE: This is not currently used since
        interestingness is run while this object
        is still in memory

        """

        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root)

        num_episodes = len(self.episodes)
        filename = "{}/cameleon_ixdrl_env_{}_{}_{}_eps{}.{}"\
            .format(
                    self.out_root,
                    self.agent_name,
                    self.framework,
                    self.env_name,
                    num_episodes,
                    self.ext)

        self.write_compressed(self,filename)



##################################################################################
##   Main Method
##################################################################################



