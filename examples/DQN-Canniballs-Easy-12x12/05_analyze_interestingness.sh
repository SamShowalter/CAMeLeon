#!/bin/bash
#################################################################################
#
#             Script Title:   Analyze interestingness from rollouts
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################

#######################################################################
# Set variable names
#######################################################################

# Date timestamp and rollout path information / output directory
DATE=`date "+%Y.%m.%d"`
ROLLOUTS_PATH="rollouts/DQN_torch_Cameleon-Canniballs-Medium-12x12-v0_ep100_ts0_rs42_w4_$DATE/"
OUTPUT_DIR="data/interestingness/"
ANALYSES="value,reward,action-value,execution-uncertainty,execution-value"
ACTION_FACTORS="direction"

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../../"
clear

# Run the script
python -m cameleon.bin.analyze_interestingness \
  --outdir=$OUTPUT_DIR \
  --rollouts-path=$ROLLOUTS_PATH \
  --analyses=$ANALYSES \
  --action-factors=$ACTION_FACTORS \
