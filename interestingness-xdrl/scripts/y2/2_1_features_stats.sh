#!/bin/bash

# gets constants
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/0_constants.sh"

FEATURE_STATS=true # whether to compute feature stats (counts)
UPLOAD=true        # whether to upload the results to the AIC filex server

FILEX="$FILEX/feature_stats"
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
    STATS_DIR="$OUTPUT_DIR/feature_stats/$SUBDIR"

    if $FEATURE_STATS; then
      echo "========================================"
      echo "Computing features stats for agent '$POLICY' in scenario '$SCENARIO'..."
      python -m feature_extractor.bin.feature_stats \
        --input="$FEATURES_DIR/$SUBDIR/feature-dataset.pkl.gz" \
        --output=$STATS_DIR \
        --verbosity=$VERBOSITY \
        --clear=$CLEAR
    fi

    if $UPLOAD; then
      SOURCE_DIR=$STATS_DIR
      TARGET_DIR="$FILEX/$SUBDIR"
      echo "========================================"
      echo "Uploading results to '$TARGET_DIR'..."
      duck -u $USERNAME -p $PASSWORD --nokeychain -v -D "$TARGET_DIR/*"                       # clear remote directory
      duck -u $USERNAME -p $PASSWORD --nokeychain -v -c $TARGET_DIR                           # create remote directory
      duck -u $USERNAME -p $PASSWORD --nokeychain -v -e skip --upload $TARGET_DIR $SOURCE_DIR # upload files
    fi

  done

done
