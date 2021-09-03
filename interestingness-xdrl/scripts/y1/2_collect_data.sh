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
SC2_DATA_DIR="$OUTPUT_DIR/sc2data"
OUTPUT_DATA_DIR="$OUTPUT_DIR/interaction_data"
REPLAYS_DIR="$OUTPUT_DIR/replays"
USE_DYNAMICS_MODEL=False
VAE_MODEL_DIR="$OUTPUT_DIR/vae/assault" # TODO
PE_MODEL="$OUTPUT_DIR/pets/model/vae_5" # TODO
BATCH_SIZE=256
SEED=0
CLEAR=true
VERBOSITY=1

for ((s = 0; s < ${#SCENARIOS[@]}; s++)); do
  SCENARIOS_FILE="$SCENARIOS_DIR/${SCENARIOS[$s]}.json"

  for ((p = 0; p < ${#POLICIES[@]}; p++)); do
    POLICY=${POLICIES[$p]}
    REPLAY="$REPLAYS_DIR/${SCENARIOS[$s]}_$POLICY.SC2Replay"

    echo "========================================"
    echo "Collecting interaction data for agent '$POLICY' in scenario '$SCENARIOS_FILE'..."

    if $USE_DYNAMICS_MODEL; then
      python -m interestingness_xdrl.bin.collect.sc2_caml_y1 \
        --replay_sc2_version=latest \
        --sc2data_root=$SC2_DATA_DIR \
        --task_module="sc2scenarios.scenarios.assault.spaces.caml_year1_eval" \
        --environment="AssaultTaskEnvironment" \
        --policy=$POLICY \
        --replays=$REPLAY \
        --crop_to_playable_area \
        --batch_size=$BATCH_SIZE \
        --output=$OUTPUT_DATA_DIR \
        --seed=$SEED \
        --clear=$CLEAR \
        --verbosity=$VERBOSITY \
        --vae_model=$VAE_MODEL_DIR \
        --pe_model=$PE_MODEL
    else
      python -m interestingness_xdrl.bin.collect.sc2_caml_y1 \
        --replay_sc2_version=latest \
        --sc2data_root=$SC2_DATA_DIR \
        --task_module="sc2scenarios.scenarios.assault.spaces.caml_year1_eval" \
        --environment="AssaultTaskEnvironment" \
        --policy=$POLICY \
        --replays=$REPLAY \
        --crop_to_playable_area \
        --batch_size=$BATCH_SIZE \
        --output=$OUTPUT_DATA_DIR \
        --seed=$SEED \
        --clear=$CLEAR \
        --verbosity=$VERBOSITY
    fi
  done
done
