#!/bin/bash

# gets constants
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/0_constants.sh"

GENERATE=false    # whether to generate the scenario config files
REPLAY=true       # whether to generate replay files
QUICK_RESET=false # whether to merge several episodes in the same replay file
UPLOAD=true       # whether to upload the results to the AIC filex server

SC2_SCENARIOS_DIR="sc2scenarios"
SC2_DATA_DIR="$OUTPUT_DIR/sc2data"
FILEX="$FILEX/replays"
NUM_SCENARIOS=10000
SCENARIOS_RANGE="0,$NUM_SCENARIOS"
SEED=314
SAVE_OBS="feature_screen"
PARALLEL=24 #12

# change to project root directory (in case invoked from other dir)
cd "$DIR/../../.." || exit
clear

# prompt user if upload
if $UPLOAD; then
  read -p "AIC filex username: " USERNAME
  IFS= read -s -p "Password: " PASSWORD
fi

for ((s = 0; s < ${#SCENARIOS[@]}; s++)); do
  SCENARIOS_FILE="$SC2_DATA_DIR/${SCENARIOS[$s]}.json"

  # first generate scenarios
  if $GENERATE; then
    SCENARIOS_FILE="$SC2_DATA_DIR/${SCENARIOS[$s]}_$NUM_SCENARIOS.json"
    python -m sc2scenarios.bin.sample_scenarios \
      --task_module="sc2scenarios.scenarios.assault.spaces.assault_v2" \
      --environment="AssaultTaskEnvironment" \
      --scenario_space="$SC2_SCENARIOS_DIR/configs/${SCENARIOS[$s]}.json" \
      --output=$SCENARIOS_FILE \
      --samples=$NUM_SCENARIOS \
      --seed=$SEED
  fi

  for ((p = 0; p < ${#POLICIES[@]}; p++)); do
    POLICY=${POLICIES[$p]}
    SUBDIR="${SCENARIOS[$s]}_$POLICY"
    REPLAY_DIR="$REPLAYS_DIR/$SUBDIR"

    if $REPLAY; then

      rm -rf "$REPLAY_DIR"
      echo "========================================"
      echo "Generating replays for agent '$POLICY' in scenario '$SCENARIOS_FILE'..."
      python -m sc2scenarios.bin.play_scenarios \
        --sc2data_root=$SC2_DATA_DIR \
        --scenarios="$SCENARIOS_FILE" \
        --index_range=$SCENARIOS_RANGE \
        --task_module="sc2scenarios.scenarios.assault.spaces.assault_v2" \
        --environment="AssaultTaskEnvironment" \
        --policy="$POLICY" \
        --replay_dir="$REPLAY_DIR" \
        --seed=$SEED \
        --quick_reset=$QUICK_RESET \
        --save_observations=$SAVE_OBS \
        --parallel=$PARALLEL

      if $QUICK_RESET; then
        rm -rf "$REPLAY_DIR/selected"
        echo "========================================"
        echo "Selecting replays for agent '$POLICY' from '$REPLAY_DIR'..."

        python -m interestingness_xdrl.bin.util.select_train_replays \
          --replay_sc2_version=latest \
          --replays="$REPLAY_DIR" \
          --output="$REPLAY_DIR/selected"
      fi
    fi

    if $UPLOAD; then
      SOURCE_DIR=$REPLAY_DIR
      TARGET_DIR="$FILEX/$SUBDIR"
      echo "========================================"
      echo "Uploading results to '$TARGET_DIR'..."
      duck -u $USERNAME -p $PASSWORD --nokeychain -v -D "$TARGET_DIR/*" # clear remote directory
      duck -u $USERNAME -p $PASSWORD --nokeychain -v -c $TARGET_DIR # create remote directory
      duck -u $USERNAME -p $PASSWORD --nokeychain -v -e skip --upload $TARGET_DIR $SOURCE_DIR # upload files
    fi

  done

done
