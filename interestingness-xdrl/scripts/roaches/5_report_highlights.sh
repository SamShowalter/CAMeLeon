#!/bin/bash

OUTPUT_DIR="output/minigames"
SCENARIO=DefeatRoaches #DefeatRoachesCorners #DefeatRoachesCircle
INTERACTION_DATA="$OUTPUT_DIR/interaction_data/$SCENARIO/interaction_data.pkl.gz"
ANALYSIS_FILE="$OUTPUT_DIR/analyses/$SCENARIO/analyses.pkl.gz"
REPLAY="$OUTPUT_DIR/replays/$SCENARIO.SC2Replay"
REPORT_DIR="$OUTPUT_DIR/report-highlights"
STEP_MUL=8 #1
FPS=8
SIZE=640,480
VERBOSITY=1
CLEAR=True

python -m interestingness_xdrl.bin.report.highlights \
  --replay_sc2_version=latest \
  --step_mul=$STEP_MUL \
  --data=$INTERACTION_DATA \
  --analysis=$ANALYSIS_FILE \
  --replays=$REPLAY \
  --window_size=$SIZE \
  --fps=$FPS \
  --output=$REPORT_DIR \
  --clear=$CLEAR \
  --verbosity=$VERBOSITY
