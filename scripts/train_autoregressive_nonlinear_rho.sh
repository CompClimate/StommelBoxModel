#!/bin/bash

args=("$@")

s_forcings=$(ls -1 ../configs/s_forcing | sed -e 's/\.yaml$//')
t_forcings=$(ls -1 ../configs/t_forcing | sed -e 's/\.yaml$//')
declare -a models=("mlp" "bnn" "ensemble" "rnn" "lstm" "gru" "conv")
window_size=10

for s_forcing in $s_forcings; do
	for t_forcing in $t_forcings; do
		for model in "${models[@]}"; do
			python ../src/train.py \
				experiment=${args[0]} \
				model=$model \
				data=autoregressive \
				s_forcing=$s_forcing \
				t_forcing=$t_forcing \
				density=nonlinear \
				tags="['$forcing', 'autoregressive', '$model']" \
				++data.window_size=$window_size
		done
	done
done
