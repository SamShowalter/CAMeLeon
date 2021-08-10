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

OUTPUT_DIR="data/interestingness/"
ENV_NAME="Cameleon-Canniballs-Easy-12x12-v0"
# ROLLOUTS_PATH="rollouts/DQN_tf2_Cameleon-Canniballs-Easy-12x12-v0_ep100_2021.07.27/"
ROLLOUTS_PATH="rollouts/DQN_torch_Cameleon-Canniballs-Medium-12x12-v0_ep1_ts0_rs42_w5_2021.08.09/"
MODEL_NAME="DQN"
FRAMEWORK="torch"
ACTION_FACTORS="direction"
USE_HICKLE="true"
ANALYSIS_CONFIG="{}"
IMG_FORMAT="pdf"
CLEAR="false"

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../"
clear

# Run the script
python -m cameleon.bin.analyze_interestingness \
  --env-name=$ENV \
  --model-name=$MODEL_NAME \
  --outdir=$OUTPUT_DIR \
  --framework=$FRAMEWORK \
  --rollouts-path=$ROLLOUTS_PATH \
  --use-hickle=$USE_HICKLE \
  --action-factors=$ACTION_FACTORS \
  --analysis-config=$ANALYSIS_CONFIG \
  --img-format=$IMG_FORMAT \
  --clear=$CLEAR
