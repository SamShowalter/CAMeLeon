#!/bin/bash

OUTPUT_DIR="output/minigames"
SCENARIO=DefeatRoaches #DefeatRoachesCorners #DefeatRoachesCircle
ANALYSIS_FILE="$OUTPUT_DIR/analyses/$SCENARIO/analyses.pkl.gz"
FEATURES_DATA="$OUTPUT_DIR/feature_extractor/$SCENARIO.csv"
REPORT_DIR="$OUTPUT_DIR/report-features"
VERBOSITY=1
CLEAR=True

python -m interestingness_xdrl.bin.report.sc2_features \
  --analysis=$ANALYSIS_FILE \
  --features=$FEATURES_DATA \
  --output=$REPORT_DIR \
  --clear=$CLEAR \
  --verbosity=$VERBOSITY
