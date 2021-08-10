
#################################################################################
#
#             Project Title:  Analyze interestingness data from Cameleon
#             Author:         Sam Showalter
#             Date:           2021-07-26
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import logging
import re
import numpy as np
import argparse

sys.path.append("../interestingness-xdrl/")
from interestingness_xdrl.analysis.config import AnalysisConfiguration
from interestingness_xdrl.analysis.full import FullAnalysis

from cameleon.utils.env import str2framework, str2list, str2bool,str2dict
from cameleon.interestingness.agent import CameleonInterestingnessAgent
from cameleon.interestingness.environment import CameleonInterestingnessEnvironment

#################################################################################
#   User defined
#################################################################################

parser = argparse.ArgumentParser(description='Port Cameleon Rollouts into Interestingnes-xdrl for analysis')

# Required arguments
parser.add_argument('--rollouts-path', default = None, required = True,help="Path to rollout directory with saved episodes")
parser.add_argument('--model-name', default=None,required = True, type = str, help='SAC, PPO, PG, A2C, A3C, IMPALA, ES, DDPG, DQN, MARWIL, APEX, or APEX_DDPG')
parser.add_argument('--env-name', default = None, required = True, help = "Any env registered with gym, including Gym Minigrid and Cameleon Environments")

# Optional arguments
parser.add_argument('--framework', default = "tf2",type=str2framework, help = "Deep learning framework to use on backend. Important that this is one of ['tf2','torch']")
parser.add_argument('--use-hickle', default = False,type=str2bool, help = "Whether or not to read in rollouts that are from hickle v. pickle")
parser.add_argument('--outdir', default='data/interestingness/',help='Directory to output results')
parser.add_argument('--action-factors', default='direction',type=str2list, help='Semantic groupings of actions. In grid worlds, only direction is present.')
parser.add_argument('--analysis-config', default = None, type=str2dict, help='Interesting analysis JSON-style config (python Dictionary)')
parser.add_argument('--img-format', default = 'pdf', help='Format of images to be saved during analysis.')
parser.add_argument('--clear', default = False, help='Whether to clear output directories before generating results.')

#################################################################################
#   Helper Functions
#################################################################################

def _get_ancillary_execution_info(args):
    """Get other information about rollouts, like
    random seed, etc

    :args: Argparse.Args: User-defined arguments

    """
    match = re.findall(r'_rs(\d+)_w(\d+)',args.rollouts_path)[0]
    args.random_seed = int(match[0])
    args.num_workers = int(match[1])

def make_ixdrl_subdir(args):
    """TODO: Docstring for make_ixdrl_subdir.

    :args: Argparse.Args: User-defined arguments

    """
    args.outdir = "{}/cameleon_ixdrl_ep{}_rs{}_w{}".\
        format(args.outdir,
               args.num_episodes,
               args.random_seed,
               args.num_workers)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)


#################################################################################
#   Main Method
#################################################################################

def main():
    """Main method run for argparse

    :args: Argparse.Args: User provided arguments

    """

    # Get arguments from input
    args = parser.parse_args()
    _get_ancillary_execution_info(args)

    # Instantiate agent
    agent = CameleonInterestingnessAgent(args.rollouts_path,
                                            args.env_name,
                                            args.model_name,
                                            args.framework,
                                            outdir = args.outdir,
                                            action_factors=args.action_factors,
                                            use_hickle=args.use_hickle)

    # Instantiate Environment
    env = CameleonInterestingnessEnvironment(args.rollouts_path,
                                            args.env_name,
                                            args.model_name,
                                            args.framework,
                                            outdir = args.outdir,
                                            action_factors = args.action_factors,
                                            use_hickle = args.use_hickle)

    # Load rollouts
    print("Getting agent interaction data")
    interaction_data = agent.get_interaction_datapoints()

    # Load rollouts
    print("\nGetting environment data")
    env_data = env.collect_all_data()

    # load analysis config
    config = None
    if (args.analysis_config) and (not os.path.isfile(args.analysis_config)):
        raise ValueError('Could not find analysis configuration file in {}'.format(args.analysis_config))
    elif (args.analysis_config):
        config = AnalysisConfiguration.load_json(args.config)
        logging.info('Loaded analysis configuration file from: {}'.format(args.config))
        config.save_json(os.path.join(args.out_dir, os.path.basename(args.config)))
    else:
        config = AnalysisConfiguration()

    # creates full analysis with all analyses
    analysis = FullAnalysis(interaction_data, config, args.img_format)
    logging.info('{} total analyses to be performed...'.format(len(analysis)))


    # Make sure they will sit in same folder
    assert env.out_root == agent.out_root,\
        "ERROR: Environment and Agent for interestingness have"\
        " different output roots: Env: {} - Agent: {}"\
        .format(env.out_root,
                agent.out_root)

    # runs and saves results
    args.outdir = agent.out_root
    args.num_episodes = agent.num_episodes
    make_ixdrl_subdir(args)
    analysis.analyze(args.outdir)
    analysis.save(os.path.join(args.outdir, 'analyses.pkl.gz'))


#######################################################################
# Run the program
#######################################################################

if __name__ == "__main__":
     main()




