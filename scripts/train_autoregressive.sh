#!/bin/bash

args=("$@")

declare -a forcings=(
	"sinusoidal_low_period"
	"sinusoidal_nonstationary"
	"sinusoidal_nonstationary_onesided"
)
# declare -a models=("mlp" "bnn" "ensemble" "rnn" "lstm" "gru")
declare -a models=("conv")
window_size=10

for forcing in "${forcings[@]}"; do
	for model in "${models[@]}"; do
		python ../src/train.py \
			experiment=${args[0]} \
			model=$model data=autoregressive \
			forcing=$forcing \
			tags="[$forcing, autoregressive, $model]" \
			++model.net.input_dim=$window_size \
			++data.window_size=$window_size
	done
done
