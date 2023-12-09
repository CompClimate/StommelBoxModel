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
				density=nonlinear
		done
	done
done
