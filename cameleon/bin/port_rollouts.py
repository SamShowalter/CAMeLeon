#################################################################################
#
#             Project Title:  Port rollouts from Cameleon for Interestingness
#             Author:         Sam Showalter
#             Date:           2021-07-26
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import argparse

sys.path.append("../")
from cameleon.utils.env import str2framework, str2list, str2bool
from cameleon.interestingness.agent import CameleonInterestingnessAgent
from cameleon.interestingness.environment import CameleonInterestingnessEnvironment

#################################################################################
#   User defined
#################################################################################

parser = argparse.ArgumentParser(description='Port Cameleon Rollouts into Interestingnes-xdrl for analysis')

parser.add_argument('--path', default = None, required = True,help="Path to rollout directory with saved episodes")
parser.add_argument('--model', default=None,required = True, type = str, help='SAC, PPO, PG, A2C, A3C, IMPALA, ES, DDPG, DQN, MARWIL, APEX, or APEX_DDPG')
parser.add_argument('--env', default = None, required = True, help = "Any env registered with gym, including Gym Minigrid and Cameleon Environments")
parser.add_argument('--framework', default = "tf2",type=str2framework, help = "Deep learning framework to use on backend. Important that this is one of ['tf2','torch']")
parser.add_argument('--use_hickle', default = False,type=str2bool, help = "Whether or not to read in rollouts that are from hickle v. pickle")
parser.add_argument('--outdir', default='data/interestingness/',help='Directory to output results')
parser.add_argument('--action_factors', default='left,right,up,down',type=str2list, help='Directory to output results')

#################################################################################
#   Main Method
#################################################################################

def main():
    """Main method run for argparse

    :args: Argparse.Args: User provided arguments

    """

    # Get arguments
    args = parser.parse_args()

    agent = CameleonInterestingnessAgent(args.path,
                                            args.env,
                                            args.model,
                                            args.framework,
                                            outdir = args.outdir,
                                            action_factors=args.action_factors,
                                            use_hickle=args.use_hickle)

    env = CameleonInterestingnessEnvironment(args.path,
                                            args.env,
                                            args.model,
                                            args.framework,
                                            outdir = args.outdir,
                                            action_factors = args.action_factors,
                                            use_hickle = args.use_hickle)

    # Load rollouts
    agent.get_interaction_datapoints()

    # Load rollouts
    env.collect_all_data()

    # Make sure they will sit in same folder
    assert env.out_root == agent.out_root,\
        "ERROR: Environment and Agent for interestingness have"\
        "different output roots: Env: {} - Agent: {}"\
        .format(env.out_root,
                agent.out_root)

    # Save rollouts
    agent.save()
    env.save()


#######################################################################
# Run the program
#######################################################################

if __name__ == "__main__":
     main()




