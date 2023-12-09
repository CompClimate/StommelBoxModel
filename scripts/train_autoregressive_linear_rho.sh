#!/bin/bash

args=("$@")

forcings=$(ls -1 ../configs/s_forcing | sed -e 's/\.yaml$//')
declare -a models=("mlp" "bnn" "ensemble" "rnn" "lstm" "gru" "conv")
window_size=10

for forcing in $forcings; do
	for model in "${models[@]}"; do
		python ../src/train.py \
			experiment=${args[0]} \
			model=$model \
			data=autoregressive \
			s_forcing=$forcing \
			tags="[$forcing, autoregressive, $model]" \
			++data.window_size=$window_size
	done
done
