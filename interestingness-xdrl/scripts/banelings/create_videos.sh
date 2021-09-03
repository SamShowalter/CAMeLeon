#!/bin/bash

OUTPUT_DIR="output/minigames"
SCENARIO=DefeatZerglingsAndBanelings
REPLAY="$OUTPUT_DIR/replays/$SCENARIO.SC2Replay"
VIDEO_DIR="$OUTPUT_DIR/videos/$SCENARIO"
STEP_MUL=1 #8 #1
SIZE=640,480
FPS=22.5
CRF=18
VERBOSITY=1
CLEAR=True

python -m interestingness_xdrl.bin.record_sc2_video \
  --replay_sc2_version=latest \
  --replays=$REPLAY \
  --output=$VIDEO_DIR \
  --step_mul=$STEP_MUL \
  --window_size=$SIZE \
  --fps=$FPS \
  --crf=$CRF \
  --hide_hud \
  --verbosity=$VERBOSITY \
#  --separate
