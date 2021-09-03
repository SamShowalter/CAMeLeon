#!/bin/bash

OUTPUT_DIR="output/minigames"
SCENARIO=DefeatRoaches #DefeatRoachesCorners #DefeatRoachesCircle
INTERACTION_DATA="$OUTPUT_DIR/interaction_data/$SCENARIO/interaction_data.pkl.gz"
ANALYSIS_CONF="interestingness-xdrl/config/analysis.json"
ANALYSIS_DIR="$OUTPUT_DIR/analyses"
VERBOSITY=1
CLEAR=True

python -m interestingness_xdrl.bin.analyze \
  --data=$INTERACTION_DATA \
  --config=$ANALYSIS_CONF \
  --output=$ANALYSIS_DIR \
  --clear=$CLEAR \
  --verbosity=$VERBOSITY
