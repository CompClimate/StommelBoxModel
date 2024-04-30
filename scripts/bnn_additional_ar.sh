#!/bin/bash

args=("$@")

s_forcings=("linear_symmetric" "sinusoidal_stationary" "sinusoidal_nonstationary")

data="autoregressive"
model="bnn"

experiment=${args[0]}

# F_1, F_2, F_3
# for i in "${!s_forcings[@]}"; do
# 	python ../src/train.py -m \
# 		experiment=$experiment \
# 		model=$model \
# 		data=$data \
# 		s_forcing=${s_forcings[$i]} \
# 		~t_forcing \
# 		density=linear \
# 		hydra=sweep_timestamp \
# 		++paths.log_dir="../logs/bnn_additional_experiments/linear_density_ar" \
# 		++model.net.prior_sigma='1e-2,1e-3,1e-4,1e-5,1e-6' &
# 	wait
# done

t_forcings=("linear_symmetric" "sinusoidal_stationary" "sinusoidal_nonstationary")

# F_4, F_5, F_6
for i in "${!s_forcings[@]}"; do
	python ../src/train.py -m \
		experiment=$experiment \
		model=$model \
		data=$data \
		s_forcing=${s_forcings[$i]} \
		t_forcing=${t_forcings[$i]} \
		density=nonlinear \
		hydra=sweep_timestamp \
		++paths.log_dir="../logs/bnn_additional_experiments/nonlinear_density_ar" \
		++model.net.prior_sigma='1e-2,1e-3,1e-4,1e-5,1e-6' &
	wait
done
