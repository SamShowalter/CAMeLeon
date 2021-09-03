#!/bin/bash

REPLAY=true        # whether to generate replay files
QUICK_RESET=false  #true # whether to merge several episodes in replay files
REPLAY_RL=true     # whether to produce replays for the RL policies
REPLAY_EXPERT=false # whether to produce replays for the expert/manual policies

declare -a SCENARIOS=(
  "RedOutOfPositionAssaultTaskSpace_42_1k_v2"
  "RestrictedMatchupsBlueOutnumberedAssaultTaskSpace_42_1k_v2"
  "RestrictedMatchupsRedOutnumberedAssaultTaskSpace_42_1k_v2"
)

declare -a POLICIES=(
  "RandomActionPolicy"
  "ReaverRandomInit"
  "ReaverAssault0"
)

declare -a EXPERT_POLICIES=(
  "TargetCommandCenterPolicy"
  "NopPolicy"
  "AttackRedForcePolicy"
)

OUTPUT_DIR="output/assault"
SC2_DATA_DIR="$OUTPUT_DIR/sc2data"
SCENARIOS_DIR="$SC2_DATA_DIR/sc2scenarios/caml_year1_eval/scenarios"
SCENARIOS_RANGE=0,1000 #0,10
REPLAYS_DIR="$OUTPUT_DIR/replays"
SEED=0
PARALLEL=12

# change to project root directory (in case invoked from other dir)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR/../../.." || exit
clear

for ((s = 0; s < ${#SCENARIOS[@]}; s++)); do
  SCENARIOS_FILE="$SCENARIOS_DIR/${SCENARIOS[$s]}.json"

  if $REPLAY_RL; then
    for ((p = 0; p < ${#POLICIES[@]}; p++)); do
      POLICY=${POLICIES[$p]}
      REPLAY_DIR="$REPLAYS_DIR/${SCENARIOS[$s]}_$POLICY"

      if $REPLAY; then
        rm -rf "$REPLAY_DIR"
        echo "========================================"
        echo "Generating replays for agent '$POLICY' in scenario '$SCENARIOS_FILE'..."

        python -m sc2scenarios.bin.play_scenarios \
          --sc2data_root=$SC2_DATA_DIR \
          --scenarios="$SCENARIOS_FILE" \
          --index_range=$SCENARIOS_RANGE \
          --task_module="sc2scenarios.scenarios.assault.spaces.caml_year1_eval" \
          --environment="AssaultTaskEnvironment" \
          --policy="$POLICY" \
          --replay_dir="$REPLAY_DIR" \
          --seed=$SEED \
          --crop_to_playable_area \
          --quick_reset=$QUICK_RESET \
          --parallel=$PARALLEL
      fi

      if $QUICK_RESET; then
        rm -rf "$REPLAY_DIR/selected"
        echo "========================================"
        echo "Selecting replays for agent '$POLICY' from '$REPLAY_DIR'..."

        python -m interestingness_xdrl.bin.util.select_train_replays \
          --replay_sc2_version=latest \
          --replays="$REPLAY_DIR" \
          --output="$REPLAY_DIR/selected"
      fi
    done
  fi

  if $REPLAY_EXPERT; then
    POLICY=${EXPERT_POLICIES[$s]}
    REPLAY_DIR="$REPLAYS_DIR/${SCENARIOS[$s]}_$POLICY"

    if $REPLAY; then
      rm -rf "$REPLAY_DIR"
      python -m sc2scenarios.bin.play_scenarios \
        --sc2data_root=$SC2_DATA_DIR \
        --scenarios="$SCENARIOS_FILE" \
        --index_range=$SCENARIOS_RANGE \
        --task_module="sc2scenarios.scenarios.assault.spaces.caml_year1_eval" \
        --environment="AssaultTaskEnvironment" \
        --policy="$POLICY" \
        --replay_dir="$REPLAY_DIR" \
        --seed=$SEED \
        --crop_to_playable_area \
        --quick_reset=$QUICK_RESET \
        --parallel=$PARALLEL
    fi

    if $QUICK_RESET; then
      rm -rf "$REPLAY_DIR/selected"
      echo "========================================"
      echo "Selecting replays for agent '$POLICY' from '$REPLAY_DIR'..."

      python -m interestingness_xdrl.bin.util.select_train_replays \
        --replay_sc2_version=latest \
        --replays="$REPLAY_DIR" \
        --output="$REPLAY_DIR/selected"
    fi
  fi

done
