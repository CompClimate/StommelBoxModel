#!/bin/bash

args=("$@")

s_forcings=$(ls -1 ../configs/s_forcing | sed -e 's/\.yaml$//')
t_forcings=$(ls -1 ../configs/t_forcing | sed -e 's/\.yaml$//')
declare -a autoreg_models=("mlp" "bnn" "ensemble" "rnn" "lstm" "gru" "conv")
declare -a physics_models=("mlp" "bnn" "ensemble")
window_size=10

for s_forcing in $s_forcings; do
	for model in "${autoreg_models[@]}"; do
		python ../src/train.py \
			experiment=${args[0]} \
			model=$model \
			data=autoregressive \
			s_forcing=$s_forcing \
			tags="[$s_forcing, autoregressive, $model]" \
			++data.window_size=$window_size \
			++train=False \
			ckpt_name="'epoch=119-step=2640.ckpt'" \
			logger=none
	done

	for t_forcing in $t_forcings; do
		for model in "${physics_models[@]}"; do
			python ../src/train.py \
				experiment=${args[0]} \
				model=$model \
				s_forcing=$s_forcing \
				t_forcing=$t_forcing \
				data=physics_informed \
				density=nonlinear \
				++train=False \
				ckpt_name="'epoch=119-step=2640.ckpt'" \
				logger=none
		done
	done
done
