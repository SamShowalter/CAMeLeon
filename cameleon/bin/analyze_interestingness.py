
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
import matplotlib.pyplot as plt
import re
import numpy as np
import argparse

sys.path.append("../interestingness-xdrl/")
from interestingness_xdrl.analysis.config import AnalysisConfiguration
from interestingness_xdrl.analysis.full import FullAnalysis

from cameleon.utils.env import str2framework, str2list, str2bool,str2dict, render_encoded_env
from cameleon.utils.general import _load_metadata, _save_metadata
from cameleon.interestingness.agent import CameleonInterestingnessAgent
from cameleon.interestingness.environment import CameleonInterestingnessEnvironment

# Set logging level
logging.basicConfig(level=logging.INFO,
                    format='%(message)s')

#################################################################################
#   User defined
#################################################################################

def create_parser(parser_creator=None):
    """Create arguments parser

    :parser_creator: Argparse.Parser: Argument parser

    """
    parser = argparse.ArgumentParser(description='Port Cameleon Rollouts into Interestingnes-xdrl for analysis')

    # Required arguments
    parser.add_argument('--rollouts-path',required = True,help="Path to rollout directory with saved episodes")

    return create_optional_args(parser)



def create_optional_args(parser):
    """ Add optional arguments to argparse

    :parser: Argparse.Args: User-defined arguments
    :returns: TODO

    """
    # Optional arguments
    parser.add_argument('--use-hickle', default = False,type=str2bool, help = "Whether or not to read in rollouts that are from hickle v. pickle")
    parser.add_argument('--outdir', default='data/interestingness/',help='Directory to output results')
    parser.add_argument('--action-factors', default='direction',type=str2list, help='Semantic groupings of actions. In grid worlds, only direction is present.')
    parser.add_argument('--analysis-config', default = None, type=str2dict, help='Interesting analysis JSON-style config (python Dictionary)')
    parser.add_argument('--analyses', default = 'all', type=str2list, help='Comma-separated string of interestingness analyses to run. You can also specify "all" to run all of them.')
    parser.add_argument('--img-format', default = 'pdf', help='Format of images to be saved during analysis.')
    parser.add_argument('--clear', default = False, help='Whether to clear output directories before generating results.')

    return parser

#################################################################################
#   Helper Functions
#################################################################################

def _get_args_from_metadata(args,metadata):
    """Get other information about rollouts, like
    random seed, etc

    :args: Argparse.Args: User-defined arguments

    """
    rollout_metadata = metadata['rollout']
    train_metadata = metadata['train']
    args.seed = rollout_metadata['seed']
    args.num_workers = rollout_metadata['num_workers']
    args.train_env_name = rollout_metadata['train_env_name']
    args.rollout_env_name = rollout_metadata['rollout_env_name']
    args.model_name = train_metadata['model_name']
    args.framework = train_metadata['framework']

#################################################################################
#   Orch function to analyze interestingness
#################################################################################

def analyze_interestingness(args,parser = None):
    """Analyze interestingness for environment

    :args: Argparse.Args: User-defined arguments

    """

    # Get metadata and update argument parser
    metadata = _load_metadata(args.rollouts_path)
    _get_args_from_metadata(args,metadata)

    # Instantiate agent
    agent = CameleonInterestingnessAgent(args.rollouts_path,
                                            args.train_env_name,
                                            args.model_name,
                                            args.framework,
                                            outdir = args.outdir,
                                            action_factors=args.action_factors,
                                            use_hickle=args.use_hickle)

    # Load rollouts
    logging.info("Getting agent interaction data")
    interaction_data = agent.get_interaction_datapoints()

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
    config.metadata = metadata; config.num_episodes = agent.num_episodes
    analysis = FullAnalysis(interaction_data, config, analyses = args.analyses,img_fmt = args.img_format)
    logging.info('{} total analyses to be performed...'.format(len(analysis)))


    # runs and saves results
    args.outdir = agent.out_root
    analysis.analyze(args.outdir)

    # Get analysis configuration and add to metadata
    # Remove metadata circular dependency
    config.metadata = None
    args.config = config.__dict__
    metadata['interestingness'] = vars(args)
    analysis.save(os.path.join(config.out_root, 'analyses.pkl.gz'))
    _save_metadata(metadata,config.out_root)

#######################################################################
# Main method to run execution with argparse args
#######################################################################

def main():
    """Main method run for argparse

    :args: Argparse.Args: User provided arguments

    """

    # Get arguments from input
    parser = create_parser()
    args = parser.parse_args()

    # Analyze interestingness
    analyze_interestingness(args)


#######################################################################
# Run the program
#######################################################################

if __name__ == "__main__":
     main()




