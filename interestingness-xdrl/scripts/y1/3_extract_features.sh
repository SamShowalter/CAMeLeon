#!/bin/bash

EXTRACT_RL=true     # whether to extract features for the RL policies
EXTRACT_EXPERT=true # whether to extract features for the expert/manual policies

declare -a SCENARIOS=(
  "RedOutOfPositionAssaultTaskSpace_42_1k_v2"
  "RestrictedMatchupsBlueOutnumberedAssaultTaskSpace_42_1k_v2"
  "RestrictedMatchupsRedOutnumberedAssaultTaskSpace_42_1k_v2"
)

declare -a POLICIES=(
  "RandomActionPolicy"
  "ReaverRandomInit"
  "ReaverAssault0"
)

declare -a EXPERT_POLICIES=(
  "TargetCommandCenterPolicy"
  "NopPolicy"
  "AttackRedForcePolicy"
)

OUTPUT_DIR="output/assault"
REPLAYS_DIR="$OUTPUT_DIR/replays"
FEATURES_CONF="$OUTPUT_DIR/features_config.json"
FEATURES_DIR="$OUTPUT_DIR/feature_extractor"
PARALLEL=12
VERBOSITY=1
CLEAR=true

# change to project root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/../../.." || exit
clear

for ((s = 0; s < ${#SCENARIOS[@]}; s++)); do
  SCENARIO=${SCENARIOS[$s]}

  if $EXTRACT_RL; then
    for ((p = 0; p < ${#POLICIES[@]}; p++)); do
      POLICY=${POLICIES[$p]}
      SUBDIR=$SCENARIO"_"$POLICY
      REPLAY_DIR="$REPLAYS_DIR/$SUBDIR"

      echo "========================================"
      echo "Extracting replays for agent '$POLICY' in scenario '$SCENARIO'..."
      python -m feature_extractor.bin.extract_features \
        --replay_sc2_version=latest \
        --config=$FEATURES_CONF \
        --replays="$REPLAY_DIR" \
        --output="$FEATURES_DIR/$SUBDIR" \
        --feature_screen_size=192,144 \
        --feature_minimap_size=72 \
        --feature_camera_width=48 \
        --parallel=$PARALLEL \
        --clear=$CLEAR \
        --verbosity=$VERBOSITY

    done
  fi

  if $EXTRACT_EXPERT; then
    POLICY=${EXPERT_POLICIES[$s]}
    SUBDIR=$SCENARIO"_"$POLICY
    REPLAY_DIR="$REPLAYS_DIR/$SUBDIR"

    echo "========================================"
    echo "Extracting replays for agent '$POLICY' in scenario '$SCENARIO'..."
    python -m feature_extractor.bin.extract_features \
      --replay_sc2_version=latest \
      --config=$FEATURES_CONF \
      --replays="$REPLAY_DIR" \
      --output="$FEATURES_DIR/$SUBDIR" \
      --feature_screen_size=192,144 \
      --feature_minimap_size=72 \
      --feature_camera_width=48 \
      --parallel=$PARALLEL \
      --clear=$CLEAR \
      --verbosity=$VERBOSITY
  fi

done
