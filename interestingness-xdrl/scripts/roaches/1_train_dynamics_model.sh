#!/bin/bash

OUTPUT_DIR="output/minigames"
REPLAY="$OUTPUT_DIR/replays/pets"
VAE_DIR="$OUTPUT_DIR/sc2_vae/DefeatRoaches-reaver"
PARALLEL=2

python imago/imago/models/sequential/pets/train_vae.py \
  --replay_sc2_version=latest \
  --feature_screen_size=64 \
  --feature_camera_width=24 \
  --action_set=screen \
  --action_spatial_dim=16 \
  --vae_model=$VAE_DIR \
  --replays=$REPLAY \
  --output="$OUTPUT_DIR/pets/model/vae" \
  --parallel=$PARALLEL \
  --verbosity=1 \
  --clear
