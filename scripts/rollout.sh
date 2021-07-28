#!/bin/bash
#################################################################################
#
#             Script Title:   Rollout bash script for environment and agent
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################

#######################################################################
# Set variable names
#######################################################################

OUTPUT_DIR="rollouts/"
STORE_VIDEO="true"
ENV="Cameleon-Canniballs-Medium-12x12-v0"
# Usually makes 5x what you put here, not sure why
EPISODES=1
# If set to zero, episodes / steps has no influence
STEPS=0
# CHECKPOINT="models/DQN_Cameleon-Canniballs-12x12-v0_2021.07.19/checkpoint_best/checkpoint-best"
# If it says params.pkl cannot be found, then this path is wrong (params.pkl exists, just somewhere else)
# If it says out of memory, then wait until training is done
# CHECKPOINT="models/DONT_DELETE_DQN_tf2_Cameleon-Canniballs-12x12-v0_2021.07.22/checkpoint_009962/checkpoint-9962"
CHECKPOINT="models/tune/Cameleon-Canniballs-Medium-12x12-v0/APPO_torch_2021.07.27/checkpoint_004840/checkpoint-4840"
NO_RENDER="true"
RUN="APPO"
WRAPPERS="canniballs_one_hot,encoding_only"
NUM_WORKERS=4
NUM_GPUS=1
USE_HICKLE="true"
NO_FRAME="false"
# Rollout process can pick up trained config if a checkpoint is given
# otherwise, specify information here
CONFIG="{}"

# CONFIG="""{
# 'model':{'dim':12,
#          'conv_filters':[[16,[4,4],1],
#                          [32,[3,3],2],
#                          [512,[6,6],1]]
#         }
# }"""

# Need to remove whitespaces
CONFIG="${CONFIG//[$'\t\r\n ']}"

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../"
clear

# Run the script
python -m cameleon.bin.rollout \
  --run=$RUN \
  --env=$ENV \
  --out=$MODEL \
  --checkpoint=$CHECKPOINT \
  --episodes=$EPISODES \
  --steps=$STEPS \
  --out=$OUTPUT_DIR \
  --store-video=$STORE_VIDEO \
  --no-render=$NO_RENDER \
  --wrappers=$WRAPPERS \
  --num-workers=$NUM_WORKERS \
  --num-gpus=$NUM_GPUS \
  --no-frame=$NO_FRAME \
  --use-hickle=$USE_HICKLE \
  --config=$CONFIG 
