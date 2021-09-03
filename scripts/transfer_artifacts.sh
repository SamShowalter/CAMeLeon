#!bin/bash
################################################################################
#
#             Script Title:   Transfer all files to remote server after zipping
#             Author:         Sam Showalter
#             Date:           2021-08-10
#
#################################################################################


#######################################################################
# Set variable names
#######################################################################

USERNAME="showalter"
PROJECT_ROOT="../../../"
REMOTE_SERVER_ROOT="https://filex.ai.sri.com/caml/cameleon/"
POST_ONLY="false"
ZIP_ONLY="false"

DIRS="rollouts,models,models/tune,data/interestingness,data/imago,data/gvf"
# DIRS="data/interestingness"
OVERWRITE="true"
ARCHIVE="archive"

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../"
clear

# Run the script
python -m cameleon.bin.transfer_artifacts \
  --username=$USERNAME \
  --project-root=$PROJECT_ROOT \
  --remote-server-root=$REMOTE_SERVER_ROOT \
  --zip-only=$ZIP_ONLY \
  --post-only=$POST_ONLY \
  --dirs=$DIRS \
  --overwrite=$OVERWRITE \
  --archive=$ARCHIVE
