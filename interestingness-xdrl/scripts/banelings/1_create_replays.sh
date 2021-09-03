#!/bin/bash

OUTPUT_DIR="output/minigames"
REAVER_DIR="$OUTPUT_DIR/policies"
EXPERIMENT="DefeatZerglingsAndBanelings_a2c_reaver"
TASK=DefeatZerglingsAndBanelings
STEPS=1000

GPU=True
VERBOSITY=1

python -m reaver.run2 \
  --test \
  --traj_len 1 \
  --task $STEPS,$TASK \
  --results_dir=$REAVER_DIR \
  --experiment=$EXPERIMENT \
  --obs_features=minimal \
  --obs_spatial_dim=16 \
  --action_set=minimal \
  --action_spatial_dim=16 \
  --save_replay_episodes 1 \
  --gpu=$GPU \
  --verbosity=$VERBOSITY
