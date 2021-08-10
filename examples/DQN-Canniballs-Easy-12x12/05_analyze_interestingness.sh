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

# Model and environment information
ENV_NAME="Cameleon-Canniballs-Easy-12x12-v0"
MODEL_NAME="DQN"
FRAMEWORK="torch"

# Date timestamp and rollout path information / output directory
DATE=`date "+%Y.%m.%d"`
ROLLOUTS_PATH="rollouts/DQN_torch_Cameleon-Canniballs-Medium-12x12-v0_ep100_ts0_rs42_w5_$DATE/"
OUTPUT_DIR="data/interestingness/"
ACTION_FACTORS="direction"
USE_HICKLE="true"

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../../"
clear

# Run the script
python -m cameleon.bin.analyze_interestingness \
  --env-name=$ENV_NAME \
  --model-name=$MODEL_NAME \
  --outdir=$OUTPUT_DIR \
  --framework=$FRAMEWORK \
  --rollouts-path=$ROLLOUTS_PATH \
  --use-hickle=$USE_HICKLE \
  --action-factors=$ACTION_FACTORS \
