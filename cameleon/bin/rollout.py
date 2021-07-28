#################################################################################
#
#             Project Title:  Recording Engine for Collecting Episodes from Cameleon
#             Author:         Sam Showalter
#             Date:           2021-07-14
#
#             Source: This was taken from rollout.py in RLlib and altered slightly.
#                     Original file found here:
#                     https://github.com/ray-project/ray/blob/master/rllib/rollout.py
#
#################################################################################

# General stuff
import argparse
import collections
import copy
import glob
import os
# os.environ["FORCE_CUDA"]="1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_HOME"]="/usr/local/cuda"
import sys
from datetime import datetime as dt

import ray.rllib.models.modelv2


# Ray stuff
import ray
import ray.cloudpickle as cloudpickle
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.tune.utils import merge_dicts
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR,register_env

# Gym stuff
import gym
import gym_minigrid.envs
from gym import wrappers as gym_wrappers

# Custom imports
import cameleon.envs
from cameleon.utils.env import load_env, str2bool, str2model, \
    str2wrapper, wrap_env, str2str, cameleon_logger_creator, str2dict, str2framework
from cameleon.callbacks.agent.rllib import RLlibCallbacks

#######################################################################
# Create parser
#######################################################################

def create_parser(parser_creator=None):
    """
    Create argparse argument list
    """
    parser_creator = parser_creator or argparse.ArgumentParser

    parser = parser_creator(formatter_class=argparse.RawDescriptionHelpFormatter,
                            description="Roll out a reinforcement learning agent given a checkpoint.")
    parser.add_argument("--checkpoint",type=str2str,default=None,nargs="?",help="(Optional) checkpoint from which to roll out. "
                                                                "If none given, will use an initial (untrained) Trainer.")

    required_named = parser.add_argument_group("required named arguments")

    required_named.add_argument("--run",type=str,required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's `DQN` or `PPO`), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument("--env",type=str,
        help="The environment specifier to use. This could be an openAI gym "
        "specifier (e.g. `CartPole-v0`) or a full class-path (e.g. "
        "`ray.rllib.examples.env.simple_corridor.SimpleCorridor`).")
    parser.add_argument("--local-mode",default = False,action="store_true",
        help="Run ray in local mode for easier debugging.")
    parser.add_argument("--no-render",default=False,type=str2bool,
        help="Suppress rendering of the environment.")
    parser.add_argument("--use-hickle",default=False,type=str2bool,
        help="Use gzip hickle over more standard pickle compression")
    parser.add_argument("--no-frame",default=True,type=str2bool,
        help="Whether or not to store frames from rollouts. Can be a huge memory burden")
    parser.add_argument("--framework",default=None,type=str2framework,
        help="Framework for model in which rollouts given. This should be provided by config in most cases")
    parser.add_argument("--store-video",type=str2bool,default=True,
        help="Specifies the directory into which videos of all episode "
        "rollouts will be stored.")
    parser.add_argument("--steps",default=10000,type = int,
        help="Number of timesteps to roll out. Rollout will also stop if "
        "`--episodes` limit is reached first. A value of 0 means no "
        "limitation on the number of timesteps run.")
    parser.add_argument("--episodes",default=0,type = int,
        help="Number of complete episodes to roll out. Rollout will also stop "
        "if `--steps` (timesteps) limit is reached first. A value of 0 means "
        "no limitation on the number of episodes run.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument("--config",default="{}",type=str2dict,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Gets merged with loaded configuration from checkpoint file and "
        "`evaluation_config` settings therein.")
    parser.add_argument('--num-workers', default = 4, type = int,
        help="Number of rollout workers to utilize during training")
    parser.add_argument('--num-gpus', default = 1,type = int,
        help="Number of GPUs to utilize during training. Generally this is not bottleneck, so 1 is often sufficient")
    parser.add_argument("--save-info",default=False,action="store_true",
        help="Save the info field generated by the step() method, "
        "as well as the action, observations, rewards and done fields.")
    parser.add_argument("--use-shelve",default=False,action="store_true",
        help="Save rollouts into a python shelf file (will save each episode "
        "as it is generated). An output filename must be set using --out.")
    parser.add_argument("--track-progress",default=False,action="store_true",
        help="Write progress to a temporary file (updated "
        "after each episode). An output filename must be set using --out; "
        "the progress file will live in the same folder.")
    parser.add_argument('--wrappers', default="", type = str2wrapper,  help=
                  """
                    Wrappers to encode the environment observation in different ways. Wrappers will be executed left to right,
                        and the options are as follows (example: 'encoding_only,canniballs_one_hot'):
                  - partial_obs.{obs_size}:         Partial observability - must include size (odd int)
                  - encoding_only:                  Provides only the encoded representation of the environment
                  - rgb_only:                       Provides only the RGB screen of environment
                  - canniballs_one_hot              Canniballs specific one-hot
                  """)
    return parser

#######################################################################
# Helper Functions
#######################################################################

class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # If num_episodes is set, stop if limit reached.
    if num_episodes and episodes >= num_episodes:
        return False
    # If num_steps is set, stop if limit reached.
    elif num_steps and steps >= num_steps:
        return False
    # Otherwise, keep going.
    return True

def get_rollout_tags(writer_files,
                     writer_dict):
    """Get tags for making rollout tag
    folder names

    :writer_files: dict:

    """
    # Fill writer dict with tags
    for file in writer_files:
        tag = "_".join(file.split("/")[-1].split("_")[:2])
        writer_dict[tag] = file

def delete_dummy_video_files(pid_max,
                             subdirs):
    """OpenAI's video creator creates one dummy
    video file per worker. This removes it.
    The id is always the last (max)

    :pid_max: dict: max pid filenames

    """

    # Remove all dummy files
    for pid,file in pid_max['files'].items():

        # Get episode
        ep = pid_max[pid]

        # Remove garbage subdir
        del subdirs[(pid,ep)]
        os.remove(file)

def build_writer_dir(args):
    """Build writer directory

    :args: Argparse args

    """
    # Add writer to environment
    args.writer_dir = "{}{}_{}_{}_ep{}_{}/".format(args.out,
                                                args.run,
                                                args.config['framework'],
                                                args.env,
                                                args.episodes,
                                                dt.now().strftime("%Y.%m.%d"))

    if not os.path.exists(args.writer_dir):
        os.makedirs(args.writer_dir)


def bundle_rollouts(subdirs,
                    writer_dict,
                    writer_dir,
                    ext = "hkl",
                    monitor = True):
    """Bundle rollout pkl files and videos
    into different folders. Only runs if
    videos are created. Otherwise pickles
    stay loose in the directory

    :subdirs: dict: Subdirectory names

    """

    # Bundle everything together
    for (pid,ep),rollout_subdir in subdirs.items():


        # Suffix and filepath - I know, this is messy.
        writer_file = writer_dict["{}_ep{}".format(pid,ep)]
        writer_suffix = "_".join(writer_file.split("/")[-1].split("_")[-2:])
        rollout_subdir = "{}_{}".format(rollout_subdir,
                                        writer_suffix.replace(".{}"\
                                                              .format(ext),""))

        #Make directory if it does not exist
        if not os.path.exists(rollout_subdir):
            os.mkdir(rollout_subdir)


        os.replace(writer_file,
                "{}/{}_ep{}_{}".format(
                                        rollout_subdir,
                                        pid,ep,writer_suffix))

        # Only move if they exist
        if monitor:
            os.replace("{}{}_ep{}_video.mp4".format(writer_dir,pid,ep),
                    "{}/{}_ep{}_{}_video.mp4".format(
                                            rollout_subdir,
                                            pid,ep,
                                            writer_suffix.replace(".{}"\
                                                              .format(ext),"")))

def rename_video_files(writer_dir,
                       video_files,
                       subdirs,
                       pid_max):
    """Rename video files to match writer
    output. OpenAI names the videos first

    :writer_dir: str: Writer directory for files
    :video_files: List: List of video files
    :subdirs: Dict: Dictionary of subdirectories to make
    :pid_max: Dict: Dictionary of dummy video ids

    """
    for v in video_files:
        # Only do this for non-processes files
        # Remove any dummy files at end
        try:

            # Get name components based on openai naming convention
            name_components = v.split("/")[-1].split(".")[-3:]

            # Get PID and episode
            pid = int(name_components[0])
            ep = int(name_components[1].replace('video',''))

            # Make rollout sub directory
            rollout_subdir = "{}{}_ep{}".format(writer_dir,pid,ep)
            subdirs[(pid,ep)] = rollout_subdir

            new_video_name = "{}{}_ep{}_video.mp4".format(
                                          writer_dir,
                                          pid,
                                          ep)

            #Keep track of which files to remove
            pid_max[pid] = max(ep,
                               pid_max.get(pid,0))
            if pid_max[pid] == ep:
                pid_max['files'][pid] = new_video_name

            os.rename(v, new_video_name)

        except Exception as e:
            # print("Process failed for filepath {}"\
            #       .format(v))
            pass

def check_for_saved_config(args):
    """Check for saved configuration

    :args: Argparse arguments
    :returns: Saved config file with merged updates

    """

    # Load configuration from checkpoint file.
    config_path = ""
    args.save_info = True
    config = None

    # If there is a checkpoint, find parameters
    if args.checkpoint:
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.pkl")
        # Try parent directory.
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

    # Load the config from pickled.
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = cloudpickle.load(f)
    # If no pkl file found, require command line `--config`.
    else:
        # If no config in given checkpoint -> Error.
        if args.checkpoint:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no `--config` given on command "
                "line!")

        # Use default config for given agent.
        _, config = get_trainer_class(args.run, return_config=True)


    # Make sure worker 0 has an Env.
    config["create_env_on_driver"] = True

    # Merge with `evaluation_config` (first try from command line, then from
    # pkl file).
    evaluation_config = copy.deepcopy(
        args.config.get("evaluation_config", config.get(
            "evaluation_config", {})))
    config = merge_dicts(config, evaluation_config)
    # Merge with command line `--config` settings (if not already the same
    # anyways).

    # Adds any custom arguments here
    config = merge_dicts(config, args.config)

    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    # Make sure we have evaluation workers.
    if not config.get("evaluation_num_workers"):
        config["evaluation_num_workers"] = config.get("num_workers", 0)
    if not config.get("evaluation_num_episodes"):
        config["evaluation_num_episodes"] = 1

    return config

#######################################################################
# Load config file
#######################################################################

def load_config(args):
    """Load configuration given all inputs

    :args: Argparse args
    :returns: Configj

    """

    args.config = check_for_saved_config(args)
    config = args.config

    #Build writer directory
    build_writer_dir(args)

    # Set render environment and number of workers
    # as well as number of gpus
    config["render_env"] = not args.no_render
    config["num_workers"] = args.num_workers
    config["num_gpus"] = args.num_gpus

    # Only set if  provided explicitly and there is not checkpoint
    if args.framework:
        # Specifying framework here only if one is explicity provided
        config["framework"] = args.framework

    # Allow user to specify a video output path.
    # Determine the video output directory.
    if args.store_video:
        # Add writer to environment
        args.video_dir = os.path.expanduser(args.writer_dir)
        config["record_env"] = args.video_dir

    else:
        config["monitor"] = False

    return config



#######################################################################
# Cleanup Script
#######################################################################


def cleanup(monitor,
            writer_dir,
            ext = "hkl"):
    """

    Clean up folders where artifacts saved. Lots of meaningless
    stuff gets generated by RLlib and clutters the space. This also
    renames everything more intuitively

    TODO: This is a bandaid for some artifacts I can't control easily.
    Probably not a great thing to have long-term

    :outdir: str:     Output directory for pickle artifact
    :writer_dir: str: Directory where artifacts saved
    :video_dir: str:  Video directory for rollouts

    """

    # Relevant files
    video_files = glob.glob(writer_dir+"*.mp4")
    writer_files = glob.glob(writer_dir+"*.{}".format(ext))
    unneeded_jsons = glob.glob(writer_dir+"*.json")

    # Remove unneeded JSON files
    for f in unneeded_jsons:
        os.remove(f)

    # Useful storage objects
    pid_max = {}
    pid_max['files'] = {}
    writer_dict = {}
    subdirs = {}

    # Rename video files
    rename_video_files(writer_dir,
                       video_files,
                       subdirs,
                       pid_max)

    # Delete the dummy video files made by OpenAi monitor
    delete_dummy_video_files(pid_max,
                             subdirs)

    # Get folder names for all rollouts (by tag)
    get_rollout_tags(writer_files,
                     writer_dict)

    # Bundle all of the artifacts together
    bundle_rollouts(subdirs,
                    writer_dict,
                    writer_dir,
                    monitor = monitor,
                    ext = ext)


#######################################################################
# Rollout information
#######################################################################

def rollout(agent,
            env,
            env_name,
            num_steps,
            num_episodes=0,
            no_render=True,
            video_dir=None):
    """
    Rollout execution function. This was largely inherited from RLlib.

    :agent: Agent:        Rllib agent
    :env:   Env:          Gym environment
    :env_name: str:       Env id / name
    :num_steps: Int:      number of steps
    :num_episodes: Int:   Number of episodes
    :no_render: bool:     Whether to render environment for visual inspection
    :video_dir: str:      Video storage path

    """


    policy_agent_mapping = default_policy_agent_mapping
    # Normal case: Agent was setup correctly with an evaluation WorkerSet,
    # which we will now use to rollout.
    if hasattr(agent, "evaluation_workers") and isinstance(
            agent.evaluation_workers, WorkerSet):
        steps = 0
        episodes = 0

        # print(env.agent)
        while keep_going(steps, num_steps, episodes, num_episodes):
            eval_result = agent.evaluate()["evaluation"]

            # Increase timestep and episode counters.
            eps = agent.config["evaluation_num_episodes"]
            episodes += eps
            steps += eps * eval_result["episode_len_mean"]
            # Print out results and continue.
            print("Episode #{}: reward: {}".format(
                episodes, eval_result["episode_reward_mean"]))
        return

    # Agent has no evaluation workers, but RolloutWorkers.
    elif hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]
        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    # Agent has neither evaluation- nor rollout workers.
    else:

        from gym import envs
        if envs.registry.env_specs.get(agent.config["env"]):
            # if environment is gym environment, load from gym
            env = gym.make(agent.config["env"])
        else:
            # if environment registered ray environment, load from ray
            env_creator = _global_registry.get(ENV_CREATOR,
                                               agent.config["env"])
            env_context = EnvContext(
                agent.config["env_config"] or {}, worker_index=0)
            env = env_creator(env_context)
        multiagent = False
        try:
            policy_map = {DEFAULT_POLICY_ID: agent.policy}
        except AttributeError:
            raise AttributeError(
                "Agent ({}) does not have a `policy` property! This is needed "
                "for performing (trained) agent rollouts.".format(agent))
        use_lstm = {DEFAULT_POLICY_ID: False}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    # If monitoring has been requested, manually wrap our environment with a
    # gym monitor, which is set to record every episode.
    if video_dir:
        env = gym_wrappers.Monitor(
            env=env,
            directory=video_dir,
            video_callable=lambda _: True,
            force=True)

    steps = 0
    episodes = 0
    while keep_going(steps, num_steps, episodes, num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and keep_going(steps, num_steps, episodes,
                                      num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                # action = agent.compute_action(a_obs)
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(
                    r for r in reward.values() if r is not None)
            else:
                reward_total += reward
            if not no_render:
                env.render()
            steps += 1
            obs = next_obs

        print("Episode #{}: reward: {}".format(episodes, reward_total))
        if done:
            episodes += 1


########################################################################
## Run method for rollout information
########################################################################

def run(args, parser):
    """
    Run rollouts with arguments and
    argparse parser

    """

    # Load config from saved dictionary
    config = load_config(args)

    # Make sure configuration has the correct outpath
    config['callbacks'] = lambda: RLlibCallbacks(outdir = args.writer_dir,
                                                    model = args.run,
                                                    framework = config['framework'],
                                                    no_frame = args.no_frame,
                                                    use_hickle = args.use_hickle)
    # import ray.rllib.models.modelv2
    #Spin up Ray
    ray.init(local_mode=args.local_mode)

    # Set up environment
    env = gym.make(args.env)

    # Wrap environment
    env = wrap_env(env, args.wrappers)

    # Register environment with Ray
    register_env(args.env, lambda config: env)

    # Create the model Trainer from config.
    cls = get_trainable_cls(args.run)

    # # Instantiate agent with update
    # config.update({
    # "num_cpus_for_driver": 1,
    # "num_workers": 2,
    # "num_gpus": 0,
    # "num_cpus_per_worker": 1,
    # "num_gpus_per_worker": 1,
    # "num_envs_per_worker": 1,
    # })

    agent = cls(env=args.env, config=config,
                logger_creator = cameleon_logger_creator(
                                    args.writer_dir))

    # Restore agent if needed
    if args.checkpoint:
        agent.restore(args.checkpoint)

    # Do the actual rollout.
    rollout(agent,env, args.env,
                args.steps, args.episodes,
                args.no_render, args.video_dir)

    # Stop the agent
    agent.stop()

    # Get the gross files out of there
    cleanup(config['monitor'],
            args.writer_dir,
            "hkl" if args.use_hickle else "pkl")


#######################################################################
# Main - Run rollout parsing engine
#######################################################################



if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # --use_shelve w/o --out option.
    if args.use_shelve and not args.out:
        raise ValueError(
            "If you set --use-shelve, you must provide an output file via "
            "--out as well!")
    # --track-progress w/o --out option.
    if args.track_progress and not args.out:
        raise ValueError(
            "If you set --track-progress, you must provide an output file via "
            "--out as well!")

    run(args, parser)
