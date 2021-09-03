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
ROLLOUTS_PATH="rollouts/APPO_torch_Cameleon-Canniballs-Easy-Fake-Corner-Disruption-12x12-v0_ep100_ts0_rs42_w1_2021.09.02"
ACTION_FACTORS="direction"
ANALYSES="value,execution-uncertainty,execution-value"
# Check this if not rolling out
USE_HICKLE="false"
ANALYSIS_CONFIG="{}"
IMG_FORMAT="pdf"
CLEAR="false"


# Need to remove whitespaces
ANALYSIS_CONFIG="${ANALYSIS_CONFIG//[$'\t\r\n ']}"

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../"
clear

# Run the script
python -m cameleon.bin.analyze_interestingness \
  --outdir=$OUTPUT_DIR \
  --rollouts-path=$ROLLOUTS_PATH \
  --analyses=$ANALYSES \
  --use-hickle=$USE_HICKLE \
  --action-factors=$ACTION_FACTORS \
  --analysis-config=$ANALYSIS_CONFIG \
  --img-format=$IMG_FORMAT \
  --clear=$CLEAR
