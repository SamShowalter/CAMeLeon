#################################################################################
#
#             Project Title:  Port saved rollout data into interestingness
#                             agent object
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
from collections import OrderedDict
import re
from tqdm import tqdm

sys.path.append("../interestingness-xdrl")
from interestingness_xdrl import InteractionDataPoint
from interestingness_xdrl.agents import Agent
from interestingness_xdrl.analysis import InterestingnessAnalysisStorageDP

from cameleon.utils.general import _write_pkl, _read_pkl, _write_hkl,\
    _read_hkl, _load_metadata, _save_metadata

#######################################################################
# Cameleon Interestingness Agent
#######################################################################

class CameleonInterestingnessAgent(Agent):

    """Cameleon Interestingness agent"""

    def __init__(self,
                 rollout_dir,
                 env_name,
                 agent_name,
                 framework,
                 outdir = "data/interestingness/",
                 action_factors = ['left','right','up','down'],
                 rollout_regex =r'[a-z\d]*_cp(\d+)_s(\d+)_r*(.+).[ph]kl',
                 use_hickle = False,
                 seed = None):
        Agent.__init__(self, seed)

        self.outdir = outdir

        # Make if it does not exist
        if not os.path.exists(outdir):
            os.makedirs(self.outdir)

        self.rollout_dir = rollout_dir
        self.num_episodes = 0
        self.use_hickle = use_hickle
        self.write_compressed = _write_hkl if use_hickle else _write_pkl
        self.read_compressed = _read_hkl if use_hickle else _read_pkl
        self.ext = "hkl" if use_hickle else "pkl"
        self.rollout_path_regex = rollout_regex
        self.env_name = env_name
        self.agent_name = agent_name
        self.framework = framework
        self.action_factors = action_factors
        self.episodes = OrderedDict()

        #Validate rollout directory
        self._validate_rollout_dir()
        self.metadata = _load_metadata(self.rollout_dir)


    def _init_episode(self,
                      episode_id,
                      episode_filepath,
                      steps,
                      reward,
                      epochs):
        """Initialize an episode to store

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
                                     "name":episode_id,
                                     "train_epochs":epochs,
                                     "total_reward":reward,
                                     "filepath":episode_filepath}

        return self.episodes[episode_id]

    def _add_timestep(self, timestep,
                      timestep_data, episode,name = None):
        """Add timestep to episode
        artifact

        :timestep: Int:       Timestep
        :timestep_data: Dict: Timestep-specific data
        :episode: Dict:       subdictionary of self.episodes

        """

        t = timestep_data

        assert t.get('action_dist',None),\
            "ERROR: Rollouts do not have information necessary to"\
            " conduct interestingness. This may be due to running "\
            "rollouts from a non-checkpointed model."

        episode[timestep] = InteractionDataPoint(
                                obs              = t['observation'],
                                action           = t["action"],
                                reward           = t["reward"],
                                action_probs     = t["action_dist"],
                                new_episode      = (timestep == 0),
                                action_factors   = self.action_factors,

                                # Not everyone has these, so default is None
                                rollout_name     = name,
                                rollout_timestep = timestep,
                                rollout_tag      = name.split("_")[0],
                                value            = t.get("value_function",None),
                                action_values    = t.get("q_values",None),
                                next_obs         = t.get("next_observation",None),
                                next_rwds        = t.get("next_reward",None)
                                                )

        # Things we should probably add to the data point
        episode[timestep].model_checkpoint = int(name.split("_")[1].replace("cp",""))
        episode[timestep].encoded_env = t["info"]["env"]
        episode[timestep].q_values = t.get("q_values",None)
        episode[timestep].action_logits = t.get("action_logits",None)

        # Get link to the analysis storage object
        analysis = InterestingnessAnalysisStorageDP(name,
                                                    timestep,
                                                    episode[timestep])
        episode[timestep].interestingness = analysis



    def add_episode(self,
                      episode_data,
                      episode_id,
                      episode_filepath,
                      steps,
                      reward,
                      epochs):
        """Add episode data to agent store

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
        self.num_episodes += 1
        for i in range(len(episode_data)):
            self._add_timestep(i,
                               episode_data[i],
                               episode_store,
                               name = episode['name'])


    def flatten_for_interestingness_v1(self):
        """Flatten data dictionary to maintain
        same data format as previously specified
        by original interestingness module

        :returns: list[InteractionDataPoint]: Interaction Data

        """
        interaction_data = []

        for e,v in self.episodes.items():
            for i in range(len(v['data'])):
                data = v['data'][i]
                interaction_data.append(data)

        return interaction_data

    def _validate_rollout_dir(self):
        """Validate rollout directory

        """
        assert (os.path.exists(self.rollout_dir) and
                os.path.isdir(self.rollout_dir) and
                os.path.exists(f"{self.rollout_dir}/metadata.json"))

    def _get_rollout_paths(self):
        """Get path to all rollout pickle
        files in directory

        :returns: List[str]: Rollout path list

        """
        self.rollout_paths = glob.glob(self.rollout_dir + "/**/*.{}"\
                                       .format(self.ext),recursive=True)
        assert len(self.rollout_paths) > 0,\
            "ERROR: Rollout directory did not find any files"
        return self.rollout_paths

    def get_interaction_datapoints(self):
        """Load rollouts for interestingness

        :returns: list[InteractionDataPoint]: Interaction Data

        """
        paths = self._get_rollout_paths()

        for i in  tqdm(range(len(paths))):
            rollout = paths[i]
            episode_id = rollout.split("/")[-1]
            matches = re.search(self.rollout_path_regex,episode_id)
            if matches:
                components = matches.groups()
                epochs = int(components[0])
                steps = int(components[1])
                total_reward = int(components[2].replace("n",'-'))
                episode_id = episode_id.replace(".{}".format(self.ext),"")

                episode_data = self.read_compressed(rollout)

                self.add_episode(episode_data,
                                episode_id,
                                rollout,
                                steps,
                                total_reward,
                                epochs)

        self.out_root = '{}{}_{}_{}'.format(
                                self.outdir,
                                self.agent_name,
                                self.framework,
                                self.env_name)

        # return self.episodes
        self.num_episodes = len(self.episodes)
        return self.flatten_for_interestingness_v1()



    def get_counterfactuals(self):
        """Get counterfactuals

        :returns: TODO

        """
        pass

    def save(self):
        """Save self as a pickle file

        NOTE: This is not currently used since
        interestingness is run while this object
        is still in memory

        """

        if not os.path.exists(self.out_root):
            os.makedirs(self.out_root)

        num_episodes = len(self.episodes)
        filename = "{}/cameleon_ixdrl_agent_{}_{}_{}_eps{}.{}"\
            .format(self.out_root,
                    self.agent_name,
                    self.framework,
                    self.env_name,
                    num_episodes,
                    self.ext)

        self.write_compressed(self,filename)



##################################################################################
##   Main Method
##################################################################################



