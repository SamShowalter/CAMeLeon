#!/bin/bash

OUTPUT_DIR="output/minigames"
REAVER_DIR="$OUTPUT_DIR/policies"
EXPERIMENT="vtrace/squeeze_DefeatRoaches_vae2_0"
TASK=DefeatRoaches
STEPS=1000

GPU=True
VERBOSITY=1

python -m reaver.run2 \
  --test \
  --traj_len 1 \
  --task $STEPS,$TASK \
  --results_dir=$REAVER_DIR \
  --experiment=$EXPERIMENT \
  --obs_features=vae2 \
  --obs_spatial_dim=64 \
  --action_set=screen \
  --action_spatial_dim=16 \
  --save_replay_episodes 1 \
  --gpu=$GPU \
  --verbosity=$VERBOSITY
