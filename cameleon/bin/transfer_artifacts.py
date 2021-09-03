#################################################################################
#
#             Project Title:  Transfer artifacts CLI Executable
#             Author:         Sam Showalter
#             Date:           2021-08-10
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################


# Logistical Packages
import os
import sys
import logging
import argparse
import importlib
import datetime as dt
from tqdm import tqdm, trange

# Cameleon packages
from cameleon.utils.ftp import CameleonHttpFTP, CAMELEON_DIR_DICT
from cameleon.utils.env import str2bool, str2dict, dict2str, str2list


#####################################################################################
# Argparse formation
#####################################################################################
parser = argparse.ArgumentParser(description='Cameleon Training API with RLlib')

# Required
parser.add_argument('--username', default = None, required = True, help = "Any argument registered with gym, including Gym Minigrid and Cameleon Environments")

# Optional
parser.add_argument('--remote-server-root',
                    default="https://filex.ai.sri.com/caml/cameleon/",
                    help='Directory to output results')
parser.add_argument('--overwrite',
                    default=False,
                    type = str2bool,
                    help='Whether or not to overwrite the existing files')
parser.add_argument('--archive',
                    default="archive",
                    help='Name of the folder where the zipped data should be stored')
parser.add_argument('--project-root',
                    default="../../../",
                    help='Relative directory to execution file to get to project root')
parser.add_argument('--dirs',
                    default=list(CAMELEON_DIR_DICT.keys()),
                    type=str2list,
                    help='List of directory roots to zip and store')
parser.add_argument('--dir-dict',
                    default=CAMELEON_DIR_DICT,
                    type=str2dict,
                    help='Dictionary of directory keys with a regex pattern to match for subdirectories.'\
                    'intended to be a superset of all possible storage directories')
parser.add_argument('--zip-only',
                    default = False,type =str2bool,
                    help="Just zip the data, do not post yet")
parser.add_argument('--post-only', default = False,
                    type =str2bool, help="Just post all existing data in archive")

#################################################################################
#   Main Method
#################################################################################

def main():
    """Main method for running and saving execution data

    """

    #Parse all arguments
    args = parser.parse_args()

    ftp = CameleonHttpFTP(args.username,
                          project_root=args.project_root,
                          remote_server_root=args.remote_server_root,
                          zip_only = args.zip_only,
                          post_only = args.post_only,
                          dirs = args.dirs,
                          dir_dict = args.dir_dict,
                          overwrite=args.overwrite,
                          archive = args.archive)
    # Run file transfer
    ftp.run()


#################################################################################
#   Run main method
#################################################################################



if __name__ == "__main__":
    main()



