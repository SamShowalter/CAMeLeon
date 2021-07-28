#!bin/bash
################################################################################
#
#             Script Title:   Training bash script for environment
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################


#######################################################################
# Set variable names
#######################################################################

OUTPUT_DIR="models/"
ENV_NAME="Cameleon-Canniballs-Medium-12x12-v0"
NUM_EPOCHS=10000
MODEL="DQN"
WRAPPERS="canniballs_one_hot,encoding_only"
CHECKPOINT_EPOCHS=20
MODEL_DIR=""
# MODEL_DIR="models/tune/Cameleon-Canniballs-Medium-12x12-v0/APPO_torch_2021.07.27/checkpoint_002780/checkpoint-2780"
NUM_WORKERS=10
NUM_GPUS=1
FRAMEWORK="torch"
VERBOSE="true"
TUNE="true"
CONFIG="""{
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
python -m cameleon.bin.train \
  --num_epochs=$NUM_EPOCHS \
  --env_name=$ENV_NAME \
  --model=$MODEL \
  --wrappers=$WRAPPERS \
  --num_workers=$NUM_WORKERS \
  --num_gpus=$NUM_GPUS \
  --checkpoint_epochs=$CHECKPOINT_EPOCHS \
  --outdir=$OUTPUT_DIR \
  --model_dir=$MODEL_DIR \
  --framework=$FRAMEWORK \
  --tune=$TUNE \
  --config=$CONFIG \
  --verbose=$VERBOSE 
