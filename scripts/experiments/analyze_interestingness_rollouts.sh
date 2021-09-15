#!/bin/bash
#################################################################################
#
#             Script Title:   Analyze interestingness from many rollouts as experiment
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################

#######################################################################
# Set variable names
#######################################################################

ROLLOUT_PATHS="""
rollouts/APPO_torch_Cameleon-Canniballs-Easy-Corner-Disruption-12x12-v0_ep10_ts0_rs42_w10_2021.09.07
"""

OUTPUT_DIR="data/interestingness/"
FRAMEWORK="torch"
ACTION_FACTORS="direction"
USE_HICKLE="false"
ANALYSIS_CONFIG="{}"
ANALYSES="value"
IMG_FORMAT="pdf"
CLEAR="false"
PLOT="false"
LOG_LEVEL="info"


# Need to remove whitespaces
ROLLOUT_PATHS="${ROLLOUT_PATHS//[$'\t\r\n ']}"
ANALYSIS_CONFIG="${ANALYSIS_CONFIG//[$'\t\r\n ']}"

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../../"
clear

# Run the script
python -m cameleon.bin.experiments.analyze_interestingness_rollouts \
  --rollout-paths=$ROLLOUT_PATHS \
  --analyses=$ANALYSES \
  --outdir=$OUTPUT_DIR \
  --use-hickle=$USE_HICKLE \
  --action-factors=$ACTION_FACTORS \
  --analysis-config=$ANALYSIS_CONFIG \
  --img-format=$IMG_FORMAT \
  --clear=$CLEAR \
  --plot=$PLOT \
  --log-level=$LOG_LEVEL
