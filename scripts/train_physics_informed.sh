#!/bin/bash

args=("$@")

. imports/forcings.sh
. imports/models_physics_informed.sh
data="physics_informed"
models=$(join_by , ${models[@]})

experiment=${args[0]}

for i in "${!s_forcings[@]}"; do
	t_forcing=$(get_t_forcing $i)

	python ../src/train.py -m \
		experiment=$experiment \
		model=$models \
		data=$data \
		s_forcing=${s_forcings[$i]} \
		$t_forcing \
		density=linear,nonlinear &

	wait
done
