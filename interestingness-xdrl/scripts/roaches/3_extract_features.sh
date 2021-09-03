#!/bin/bash

OUTPUT_DIR="output/minigames"
SCENARIO=DefeatRoaches #DefeatRoachesCorners #DefeatRoachesCircle
REPLAY="$OUTPUT_DIR/replays/$SCENARIO.SC2Replay"
FEATURES_CONF="feature-extractor/config/features_roaches.json"
FEATURES_DIR="$OUTPUT_DIR/feature_extractor"
VERBOSITY=1
CLEAR=True

python -m feature_extractor.bin.extract_features \
  --replay_sc2_version=latest \
  --config=$FEATURES_CONF \
  --replays=$REPLAY \
  --output=$FEATURES_DIR \
  --feature_screen_size=64 \
  --feature_minimap_size=64 \
  --feature_camera_width=24 \
  --clear=$CLEAR \
  --verbosity=$VERBOSITY
