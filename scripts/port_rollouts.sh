#!/bin/bash
#################################################################################
#
#             Script Title:   Port rollouts from Cameleon to Interestingness-xdrl
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################

#######################################################################
# Set variable names
#######################################################################

OUTPUT_DIR="data/interestingness/"
ENV="Cameleon-Canniballs-Easy-12x12-v0"
# ROLLOUTS_PATH="rollouts/DQN_tf2_Cameleon-Canniballs-Easy-12x12-v0_ep100_2021.07.27/"
ROLLOUTS_PATH="rollouts/hickle_gzip_wframe_APPO_tf2_Cameleon-Canniballs-Easy-12x12-v0_ep1_2021.07.27/"
MODEL="APPO"
FRAMEWORK="tf2"
ACTION_FACTORS="left,right,up,down"
USE_HICKLE="true"

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../"
clear

# Run the script
python -m cameleon.bin.port_rollouts \
  --env=$ENV \
  --model=$MODEL \
  --outdir=$OUTPUT_DIR \
  --framework=$FRAMEWORK \
  --path=$ROLLOUTS_PATH \
  --use_hickle=$USE_HICKLE \
  --action_factors=$ACTION_FACTORS \
