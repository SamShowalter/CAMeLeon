#################################################################################
#
#             Project Title:  Analyze interestingness for several rollout collections
#             Author:         Sam Showalter
#             Date:           2021-08-16
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import re
import glob
import logging
import datetime as dt
import subprocess
import copy
import shutil
import argparse

# Gym stuff
import gym
import gym_minigrid.envs
from gym import wrappers as gym_wrappers

# Custom imports
import cameleon.envs
from cameleon.utils.parser import str2bool, str2int, str2list, str2dict
from cameleon.bin.analyze_interestingness import create_optional_args, analyze_interestingness


#################################################################################
#   Function-Class Declaration
#################################################################################

def create_parser(parser_creator = None):
    """Create parser for rollout agent experiments
    :returns: Argparser.Args: User-defined arguments

    """

    parser_creator = parser_creator or argparse.ArgumentParser

    parser = parser_creator(formatter_class=argparse.RawDescriptionHelpFormatter,
                            description="Analyze interestingness across several collections of rollouts")
    # Required arguments
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument('--rollout-paths',required = True,type=str2list,
                                help="List of rollout directories on which we want to conduct interestingness analysis")

    parser = create_optional_args(parser)
    return parser


#######################################################################
#    Functions for preprocessing input and validting config
#######################################################################

def _validate_dir(directory):
    """Validate directory exists

    :directory: Make sure directory exists
    :returns: bool: Whether or not dir exists

    """
    return (os.path.exists(directory) and
            os.path.isdir(directory) and
            os.path.exists("{}/metadata.json".format(directory)))

def _validate_dirs(args):
    """Validate all directories given by argparse
    ahead of time

    :args: Argparse.Args: User-defined arguments

    """
    for rollout_path in args.rollout_paths:
        assert _validate_dir(rollout_path),\
            "ERROR: Path does not exist or metadata.json not found:\n - {}"\
            .format(rollout_path)

def _run_interestingness_subexperiment(args,parser):
    """Run single interestingness analysis as part of larger
    experiment

    :args: Argparse.Args: User-defined arguments

    """
    analyze_interestingness(args,parser)

def run_interestingness_experiment(master_args,parser):
    """Run full set of interestingness sub_experiments

    :args: Argparse.Args: User-defined args

    """


    # Set logging level
    logging.basicConfig(level=master_args.log_level,
                        format='%(message)s')

    # Validate given directories
    logging.info("Validating directories")
    _validate_dirs(master_args)
    timestamp = dt.datetime.now().strftime("%Y.%m.%d")

    for rollouts_path in master_args.rollout_paths:
        args = copy.deepcopy(master_args)
        args.rollouts_path = rollouts_path

        # Run the sub-experiment
        logging.info("=========="*7)
        logging.info("Running interestingness experiment for:\nRollout path: {}"\
              .format(args.rollouts_path))
        logging.info("=========="*7)
        _run_interestingness_subexperiment(args,parser)

#######################################################################
# Main method for full experiment
#######################################################################

def main():
    """Run execution

    """
    parser = create_parser()
    args = parser.parse_args()

    # Run rollout experiment
    run_interestingness_experiment(args,parser)

#######################################################################
# Run main method
#######################################################################

if __name__ == "__main__":
    main()








