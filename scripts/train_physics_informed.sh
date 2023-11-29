#!/bin/bash

args=("$@")

declare -a forcings=(
	"sinusoidal_low_period"
	"sinusoidal_nonstationary"
	"sinusoidal_nonstationary_onesided"
)
declare -a models=("mlp" "bnn" "ensemble")

for forcing in "${forcings[@]}"; do
	for model in "${models[@]}"; do
		python ../src/train.py \
			experiment=${args[0]} \
			model=$model \
			forcing=$forcing \
			data=physics_informed \
			feature_names="[F, DeltaT, DeltaS]"
	done
done
