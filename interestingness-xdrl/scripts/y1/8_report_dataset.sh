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
VERBOSITY=1
CLEAR=true

# change to project root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/../../.."
clear

for ((s = 0; s < ${#SCENARIOS[@]}; s++)); do
  SCENARIO=${SCENARIOS[$s]}

  for ((p = 0; p < ${#POLICIES[@]}; p++)); do
    POLICY=${POLICIES[$p]}

    SUB_DIR=$SCENARIO"_"$POLICY
    ANALYSIS_FILE="$OUTPUT_DIR/analyses/$SUB_DIR/analyses.pkl.gz"
    INTERACTION_DATA="$OUTPUT_DIR/interaction_data/$SUB_DIR/interaction_data.pkl.gz"
    REPORT_DIR="$OUTPUT_DIR/report-dataset/$SUB_DIR"

    python -m interestingness_xdrl.bin.report.dataset \
      --data=$INTERACTION_DATA \
      --analysis=$ANALYSIS_FILE \
      --output=$REPORT_DIR \
      --clear=$CLEAR \
      --verbosity=$VERBOSITY

  done
done
