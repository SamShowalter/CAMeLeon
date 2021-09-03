#!/bin/bash

OUTPUT_DIR="output/minigames"
SCENARIO=DefeatRoaches #DefeatRoachesCorners #DefeatRoachesCircle
INTERACTION_DATA="$OUTPUT_DIR/interaction_data/$SCENARIO/interaction_data.pkl.gz"
ANALYSIS_FILE="$OUTPUT_DIR/analyses/$SCENARIO/analyses.pkl.gz"
REPLAY="$OUTPUT_DIR/replays/$SCENARIO.SC2Replay"
REAVER_DIR="$OUTPUT_DIR/reaver_policies"
VAE_MODEL_DIR="$OUTPUT_DIR/sc2_vae/DefeatRoaches-rb"
REPORT_DIR="$OUTPUT_DIR/report-counterfactuals"
SIZE=640,480
GPU=True
VERBOSITY=1
CLEAR=True

python -m interestingness_xdrl.bin.report.counterfactuals \
  --replay_sc2_version=latest \
  --obs_features=vae2 \
  --obs_spatial_dim=64 \
  --action_set=screen \
  --action_spatial_dim=16 \
  --results_dir=$REAVER_DIR \
  --experiment=vtrace/squeeze_DefeatRoaches_vae2_0 \
  --env=DefeatRoaches \
  --vae_model=$VAE_MODEL_DIR \
  --data=$INTERACTION_DATA \
  --analysis=$ANALYSIS_FILE \
  --replays=$REPLAY \
  --window_size=$SIZE \
  --output=$REPORT_DIR \
  --gpu=$GPU \
  --clear=$CLEAR \
  --verbosity=$VERBOSITY
