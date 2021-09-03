#!/bin/bash

declare -a SCENARIOS=(
  "RedOutOfPositionAssaultTaskSpace_42_1k_v2"
  "RestrictedMatchupsBlueOutnumberedAssaultTaskSpace_42_1k_v2"
  "RestrictedMatchupsRedOutnumberedAssaultTaskSpace_42_1k_v2"
)

declare -a POLICIES=(
  "RandomActionPolicy"
  "ReaverAssault0"
)

OUTPUT_DIR="output/assault"
INPUT_REPLAYS_DIR="$OUTPUT_DIR/replays"
CLEAR=false #true
ANALYZE=false

for ((s = 0; s < ${#SCENARIOS[@]}; s++)); do

  for ((p = 0; p < ${#POLICIES[@]}; p++)); do
    POLICY=${POLICIES[$p]}
    REPLAY_DIR="$INPUT_REPLAYS_DIR/${SCENARIOS[$s]}_$POLICY"

    echo "========================================"
    echo "Selecting replays for agent '$POLICY' from '$REPLAY_DIR'..."

    python -m interestingness_xdrl.bin.util.select_train_replays \
      --replay_sc2_version=latest \
      --replays="$REPLAY_DIR" \
      --output="$REPLAY_DIR/selected" \
      --analyze=$ANALYZE \
      --clear=$CLEAR
  done
done
