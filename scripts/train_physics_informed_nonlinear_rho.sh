#!/bin/bash

args=("$@")

s_forcings=$(ls -1 ../configs/s_forcing | sed -e 's/\.yaml$//')
t_forcings=$(ls -1 ../configs/t_forcing | sed -e 's/\.yaml$//')
declare -a models=("mlp" "bnn" "ensemble")

for s_forcing in $s_forcings; do
	for t_forcing in $t_forcings; do
		for model in "${models[@]}"; do
			python ../src/train.py \
				experiment=${args[0]} \
				model=$model \
				s_forcing=$s_forcing \
				t_forcing=$t_forcing \
				data=physics_informed \
				density=nonlinear \
				feature_names="['S_1', 'S_2', 'T_1', 'T_2', 'F_s', 'F_t']" \
				++model.net.input_dim=6
		done
	done
done
