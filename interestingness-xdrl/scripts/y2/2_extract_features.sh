#!/bin/bash

# gets constants
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/0_constants.sh"

EXTRACT_FEATURES=true # whether to run the feature extractor
UPLOAD=true           # whether to upload the results to the AIC filex server

FEATURES_CONF="$OUTPUT_DIR/features_Y2.json"
ENV_SPEC="sc2scenarios.scenarios.assault.spaces.assault_v2.AssaultTaskEnvironment"
FILEX="$FILEX/features"
PARALLEL=12
VERBOSITY=1
CLEAR=true

# change to project root directory (in case invoked from other dir)
cd "$DIR/../../.." || exit
clear

# prompt user if upload
if $UPLOAD; then
  read -p "AIC filex username: " USERNAME
  IFS= read -s -p "Password: " PASSWORD
fi

for ((s = 0; s < ${#SCENARIOS[@]}; s++)); do
  SCENARIO=${SCENARIOS[$s]}

  for ((p = 0; p < ${#POLICIES[@]}; p++)); do
    POLICY=${POLICIES[$p]}
    SUBDIR=$SCENARIO"_"$POLICY
    REPLAY_DIR="$REPLAYS_DIR/$SUBDIR"

    if $EXTRACT_FEATURES; then
      echo "========================================"
      echo "Extracting features for agent '$POLICY' in scenario '$SCENARIO'..."
      python -m feature_extractor.bin.extract_features \
        --replay_sc2_version=latest \
        --config=$FEATURES_CONF \
        --env_spec=$ENV_SPEC \
        --replays="$REPLAY_DIR" \
        --output="$FEATURES_DIR/$SUBDIR" \
        --feature_screen_size=192,144 \
        --feature_minimap_size=72 \
        --feature_camera_width=48 \
        --parallel=$PARALLEL \
        --clear=$CLEAR \
        --verbosity=$VERBOSITY
    fi

    if $UPLOAD; then
      SOURCE_DIR="$FEATURES_DIR/$SUBDIR"
      TARGET_DIR="$FILEX/$SUBDIR"
      echo "========================================"
      echo "Uploading results to '$TARGET_DIR'..."
      duck -u $USERNAME -p $PASSWORD --nokeychain -v -D "$TARGET_DIR/*"                       # clear remote directory
      duck -u $USERNAME -p $PASSWORD --nokeychain -v -c $TARGET_DIR                           # create remote directory
      duck -u $USERNAME -p $PASSWORD --nokeychain -v -e skip --upload $TARGET_DIR $SOURCE_DIR # upload files
    fi

  done

done
