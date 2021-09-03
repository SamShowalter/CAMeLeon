#!/bin/bash

OUTPUT_DIR="output/minigames"
SCENARIO=DefeatRoaches #DefeatRoachesCorners #DefeatRoachesCircle
REPLAY="$OUTPUT_DIR/replays/$SCENARIO.SC2Replay"
REAVER_DIR="$OUTPUT_DIR/reaver_policies"
VAE_MODEL_DIR="$OUTPUT_DIR/sc2_vae/DefeatRoaches-rb"
PE_MODEL="$OUTPUT_DIR/pets/model/vae_7"
GPU=True
VERBOSITY=1
CLEAR=True
USE_DYNAMICS_MODEL=false

if $USE_DYNAMICS_MODEL; then
  python -m interestingness_xdrl.bin.collect.sc2_reaver \
    --replay_sc2_version=latest \
    --results_dir=$REAVER_DIR \
    --experiment=vtrace/squeeze_DefeatRoaches_vae2_0 \
    --obs_features=vae2 \
    --obs_spatial_dim=64 \
    --action_set=screen \
    --action_spatial_dim=16 \
    --env=DefeatRoaches \
    --replays=$REPLAY \
    --output="$OUTPUT_DIR/interaction_data" \
    --gpu=$GPU \
    --clear=$CLEAR \
    --verbosity=$VERBOSITY \
    --vae_model=$VAE_MODEL_DIR \
    --pe_model=$PE_MODEL
else
  python -m interestingness_xdrl.bin.collect.sc2_reaver \
    --replay_sc2_version=latest \
    --results_dir=$REAVER_DIR \
    --experiment=vtrace/squeeze_DefeatRoaches_vae2_0 \
    --obs_features=vae2 \
    --obs_spatial_dim=64 \
    --action_set=screen \
    --action_spatial_dim=16 \
    --env=DefeatRoaches \
    --replays=$REPLAY \
    --output="$OUTPUT_DIR/interaction_data" \
    --gpu=$GPU \
    --clear=$CLEAR \
    --verbosity=$VERBOSITY
fi
