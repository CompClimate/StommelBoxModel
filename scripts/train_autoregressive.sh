#!/bin/bash

args=("$@")
window_size=10

for forcing in sinusoidal_{low_period,nonstationary,nonstationary_onesided}; do
	for model in ensemble mlp bnn; do
		python ../src/train.py \
			experiment=${args[0]} \
			model=$model data=autoregressive \
			forcing=$forcing \
			tags="[$forcing, autoregressive, $model]" \
			++model.net.input_dim=$window_size \
			++data.window_size=$window_size
	done
done
