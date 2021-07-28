################################################################################
#
#             Project Title:  Training API for RLlib and Cameleon
#             Author:         Sam Showalter
#             Date:           2021-07-07
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

# Logistical Packages
import os
import sys
import logging
from tqdm import tqdm, trange
import argparse
import importlib
from datetime import datetime as dt

# DL and distributed libs
import ray
import torch

# Environment libs
import gym
import gym_minigrid
import cameleon.envs
from gym import envs

# Import all ray models for zoo and registration for ray
from ray.tune.registry import register_env
from ray import tune
from cameleon.callbacks.rllib.tune_progress import CameleonRLlibTuneReporter
from cameleon.utils.env import str2bool, str2dict, dict2str, str2wrapper,str2model, wrap_env,load_env, update_config, cameleon_logger_creator, str2framework
from cameleon.utils.general import _write_pkl

#####################################################################################
# Argparse formation
#####################################################################################
parser = argparse.ArgumentParser(description='Cameleon Training API with RLlib')
parser.add_argument('--model', default=None,required = True, type = str, help='SAC, PPO, PG, A2C, A3C, IMPALA, ES, DDPG, DQN, MARWIL, APEX, or APEX_DDPG')
parser.add_argument('--outdir', default='../models/', help='Directory to output results')
parser.add_argument('--env_name', default = None, required = True, help = "Any argument registered with gym, including Gym Minigrid and Cameleon Environments")
parser.add_argument('--num_epochs', default = 100,type =int, help="Number of training iterations for algorithm")
parser.add_argument('--num_workers', default = 4, type = int,help="Number of rollout workers to utilize during training")
parser.add_argument('--num_gpus', default = 1,type = int, help="Number of GPUs to utilize during training. Generally this is not bottleneck, so 1 is often sufficient")
parser.add_argument('--checkpoint_epochs', default = 5, type = int, help="Number of epochs before a checkpoints is saved")
parser.add_argument('--config',default = None,type = str2dict,help = 'JSON string configuration for RLlib training')
parser.add_argument('--model_dir', default = None, help = "Model directory, if a pretrained system exists already")
parser.add_argument('--framework', default = "tf2",type=str2framework, help = "Deep learning framework to use on backend. Important that this is one of ['tf2','torch']")
parser.add_argument('--verbose', default = True, type=str2bool, help = "Determine if output should be verbose")
parser.add_argument('--tune', default = False, type=str2bool, help = "Determine if the tune wrapper from Ray should be used for training")
parser.add_argument('--wrappers', default="", type = str2wrapper,  help=
                  """
                    Wrappers to encode the environment observation in different ways. Wrappers will be executed left to right, and the options are as follows (example: 'encoding_only,canniballs_one_hot'):
                  - partial_obs.{obs_size}:         Partial observability - must include size (odd int)
                  - encoding_only:                  Provides only the encoded representation of the environment
                  - rgb_only:                       Provides only the RGB screen of environment
                  - canniballs_one_hot              Canniballs specific one-hot
                  """)

#######################################################################
# Helper arguments for argparse execution
#######################################################################

def train_agent_tune(agent,
                     args,
                     config):
    """Train agent with tune. Can be useful for hyperparameter
    tuning and memory management

    :agent: Trainable:    Trainable object with Ray API
    :args: Argparse.Args: Input arguments
    :config: Dict:        Configuration dict for agent

    """

    # Get names for everything
    config["env"] = args.env_name
    name = args.env_name
    trial_dirname_lambda = lambda trial: args.tune_dirname

    # Run experiment
    tune.run(
        agent,
        config = config,
        stop = {"training_iteration":args.num_epochs},

        # Naming conventions
        local_dir = args.outdir_root + "/tune",
        name = name,
        trial_dirname_creator=trial_dirname_lambda,

        # Can set the resources per trial
        # resources_per_trial = None
        progress_reporter=CameleonRLlibTuneReporter(
            args.num_epochs,
            print_intermediate_tables = True),

        #Checkpoint frequency
        checkpoint_freq=args.checkpoint_epochs,

        # Set verbosity to 2
        verbose = 1,

        restore = args.model_dir,

        # resume = True,

        # scheduler=None
        checkpoint_at_end=True,
    )

def train_agent_standalone(agent,
                           args,
                           config):
    """Train the RL agent on the chosen environment

    :num_epochs: Int:        Number of training epochs
    :checkpoint_epochs: Int: Number of epochs between checkpoints
    :outdir: str:            Output directory for saved models
    :verbose: bool:          If the execution should print to console

    """
    #Status callback
    status = "{:2d} reward min/mean/max {:6.2f}/{:6.2f}/{:6.2f} mean_len {:4.2f} saved {}"

    # Iterate through training loop with update messages
    for n in tqdm(range(args.num_epochs),leave = False):
        # Train agent
        result = agent.train()
        chkpt_file = "None saved"

        # Save checkpoint if required
        if (n % args.checkpoint_epochs) == 0:
            chkpt_file = agent.save(args.outdir)

        #Log information
        logging.info(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))

        #Print if needed
        if args.verbose:

            tqdm.write(status.format(
                    n + 1,
                    result["episode_reward_min"],
                    result["episode_reward_mean"],
                    result["episode_reward_max"],
                    result["episode_len_mean"],
                    chkpt_file
                    ))

def train_agent(agent,
                args,
                config,
                tune = False):
    """TODO: Docstring for train_agent.
    :returns: TODO

    """
    if tune:
        train_agent_tune(agent,
                         args,
                         config)
    else:
        assert args.framework in ['tf','tf2'],\
            "ERROR: Currently torch cannot run outside of tune"

        train_agent_standalone(agent,
                               args,
                               config)


#######################################################################
# Main method for argparse
#######################################################################

def main():
    """Main method for argparse and rllib training

    """

    #Parse all arguments
    args = parser.parse_args()

    # Initialize Ray - and try to prevent OOM
    ray.init(object_store_memory=3.5e9)

    # Set up environment
    env = gym.make(args.env_name)

    # Wrap environment
    env = wrap_env(env, args.wrappers)

    # Register environment with Ray
    register_env(args.env_name, lambda config: env)

    # Set model and config
    model, config = str2model(args.model, config = True)

    #Add to config for compute resources
    config['num_workers'] = args.num_workers
    config['num_gpus'] = args.num_gpus
    config['framework'] = args.framework

    #Update config if one was passed
    if args.config:
        config = update_config(config, args.config)


    # Update outdir
    args.outdir_root = args.outdir
    args.outdir = "{}{}_{}_{}_{}".format(args.outdir,
                                    args.model,
                                    args.framework,
                                    args.env_name,
                                    dt.now().strftime("%Y.%m.%d"))

    args.tune_dirname = "{}_{}_{}".format(
                                    args.model,
                                    args.framework,
                                    dt.now().strftime("%Y.%m.%d"))


    # Set up agent
    if not args.tune:
        agent = model(env = args.env_name,
                  config = config,
                  logger_creator=cameleon_logger_creator(
                            args.outdir))

        # Change to pretrained model if needed
        if args.model_dir:
            agent.restore(args.model_dir)
    else:
        agent = args.model


    # Train the agent
    train_agent(agent,
                args,
                config,
                tune = args.tune)

#######################################################################
# Main method
#######################################################################

if __name__ == "__main__":
    main()

