#!/bin/bash

declare -a SCENARIOS=("MeansOfProduction_42_10k")

#declare -a RL_POLICIES=(
#  "ReaverRandomInit"
#  "ReaverAssault0"
#)

#declare -a EXPERT_POLICIES=(
#  "AttackMoveCenterPolicy"
#  "AttackRedForcePolicy"
#  "TargetCommandCenterPolicy"
#  "RandomActionRandomIntervalAsyncPolicy"
#)

declare -a POLICIES=(
  "ExpertPolicyA"
  "RandomActionRandomIntervalAsyncPolicy"
)

FILEX="https://filex.ai.sri.com/caml/y2"
OUTPUT_DIR="output/y2"
REPLAYS_DIR="$OUTPUT_DIR/replays"
FEATURES_DIR="$OUTPUT_DIR/features"
