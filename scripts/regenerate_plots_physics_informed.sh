#!/bin/bash

args=("$@")

. imports/forcings.sh
. imports/models_physics_informed.sh
data="physics_informed"
models=("bnn" "mlp" "ensemble_mlp")
models=$(join_by , ${models[@]})
s_forcings=("linear_symmetric" "sinusoidal_stationary" "sinusoidal_nonstationary")

experiment=${args[0]}

for i in "${!s_forcings[@]}"; do
	# t_forcing=$(get_t_forcing $i)

	python ../src/train.py -m \
		experiment=$experiment \
		model=$models \
		data=$data \
		s_forcing=${s_forcings[$i]} \
		t_forcing=${s_forcings[$i]} \
		density=linear \
		++train=False \
		logger=none &

		# $t_forcing \
		# density=linear,nonlinear \
	wait
done
