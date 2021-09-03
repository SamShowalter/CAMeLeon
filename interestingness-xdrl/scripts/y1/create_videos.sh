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
REPLAYS_DIR="$OUTPUT_DIR/replays"
VIDEO_DIR="$OUTPUT_DIR/videos"
STEP_MUL=8 #1
SIZE=640,480
FPS=22.5
CRF=18
SEPARATE=true #false
VERBOSITY=1

for ((s = 0; s < ${#SCENARIOS[@]}; s++)); do

  for ((p = 0; p < ${#POLICIES[@]}; p++)); do
    POLICY=${POLICIES[$p]}
    REPLAY="$REPLAYS_DIR/${SCENARIOS[$s]}_$POLICY.SC2Replay"

    python -m interestingness_xdrl.bin.record_sc2_video \
      --replay_sc2_version=latest \
      --replays=$REPLAY \
      --output="$VIDEO_DIR/${SCENARIOS[$s]}_$POLICY" \
      --step_mul=$STEP_MUL \
      --window_size=$SIZE \
      --fps=$FPS \
      --crf=$CRF \
      --hide_hud \
      --verbosity=$VERBOSITY \
      --separate=$SEPARATE

  done

done
