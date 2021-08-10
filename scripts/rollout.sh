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
ENV_NAME="Cameleon-Canniballs-Medium-12x12-v0"

# Usually makes 5x what you put here, not sure why
EPISODES=1
TIMESTEPS=0

CHECKPOINT_PATH="models/DQN_torch_Cameleon-Canniballs-Medium-12x12-v0_2021.08.05/checkpoint_008000/checkpoint-8000"
NO_RENDER="true"
MODEL_NAME="DQN"
WRAPPERS="canniballs_one_hot,encoding_only"
NUM_WORKERS=5
NUM_GPUS=1
SEED=42
USE_HICKLE="true"
NO_FRAME="true"
STORE_IMAGO="false"
IMAGO_DIR="data/imago/"
IMAGO_FEATURES="observation,action_dist,action_logits,value_function"
BUNDLE_ONLY="false"
BUNDLE_ONLY_DIR=""

# Rollout process can pick up trained config if a checkpoint is given
# otherwise, specify information here
CONFIG="{}"

CONFIG="""{
'framework':'torch',
'model':{'dim':12,
         'conv_filters':[[16,[4,4],1],
                         [32,[3,3],2],
                         [512,[6,6],1]]
        }
}"""


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
  --model-name=$MODEL_NAME \
  --env-name=$ENV_NAME \
  --checkpoint-path=$CHECKPOINT_PATH \
  --num-episodes=$EPISODES \
  --num-timesteps=$TIMESTEPS \
  --outdir=$OUTPUT_DIR \
  --store-video=$STORE_VIDEO \
  --no-render=$NO_RENDER \
  --wrappers=$WRAPPERS \
  --num-workers=$NUM_WORKERS \
  --num-gpus=$NUM_GPUS \
  --seed=$SEED \
  --no-frame=$NO_FRAME \
  --use-hickle=$USE_HICKLE \
  --store-imago=$STORE_IMAGO \
  --imago-dir=$IMAGO_DIR \
  --imago-features=$IMAGO_FEATURES \
  --bundle-only=$BUNDLE_ONLY \
  --bundle-only-dir=$BUNDLE_ONLY_DIR \
  --config=$CONFIG 