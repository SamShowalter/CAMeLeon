#!/bin/bash

declare -a SCENARIOS=(
  "RedOutOfPositionAssaultTaskSpace_42_1k_v2"
  "RestrictedMatchupsBlueOutnumberedAssaultTaskSpace_42_1k_v2"
  "RestrictedMatchupsRedOutnumberedAssaultTaskSpace_42_1k_v2"
)

declare -a POLICIES=(
  "ReaverRandomInit"
  "ReaverAssault0"
)

OUTPUT_DIR="output/assault"
ANALYSIS_CONF="interestingness-xdrl/config/analysis_y1.json"
ANALYSIS_DIR="$OUTPUT_DIR/analyses"
VERBOSITY=1
CLEAR=true

# change to project root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/../../.."
clear

for ((s = 0; s < ${#SCENARIOS[@]}; s++)); do
  SCENARIOS_FILE="$SCENARIOS_DIR/${SCENARIOS[$s]}.json"

  for ((p = 0; p < ${#POLICIES[@]}; p++)); do
    POLICY=${POLICIES[$p]}
    REPLAY="$REPLAYS_DIR/${SCENARIOS[$s]}_$POLICY.SC2Replay"

    INTERACTION_DATA="$OUTPUT_DIR/interaction_data/${SCENARIOS[$s]}_$POLICY/interaction_data.pkl.gz"

    python -m interestingness_xdrl.bin.analyze \
      --data=$INTERACTION_DATA \
      --config=$ANALYSIS_CONF \
      --output=$ANALYSIS_DIR \
      --clear=$CLEAR \
      --verbosity=$VERBOSITY

  done
done
