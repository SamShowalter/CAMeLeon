#!/bin/bash

OUTPUT_DIR="output/minigames"
SCENARIO=DefeatRoaches #DefeatRoachesCorners #DefeatRoachesCircle
INTERACTION_DATA="$OUTPUT_DIR/interaction_data/$SCENARIO/interaction_data.pkl.gz"
ANALYSIS_FILE="$OUTPUT_DIR/analyses/$SCENARIO/analyses.pkl.gz"
REPORT_DIR="$OUTPUT_DIR/report-dataset/$SCENARIO"
VERBOSITY=1
CLEAR=true

# change to project root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/../../.."
clear

python -m interestingness_xdrl.bin.report.dataset \
  --data=$INTERACTION_DATA \
  --analysis=$ANALYSIS_FILE \
  --output=$REPORT_DIR \
  --clear=$CLEAR \
  --verbosity=$VERBOSITY
