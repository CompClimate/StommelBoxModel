#!/bin/bash

args=("$@")
models=(mlp bnn ensemble)
forcings=(sinusoidal_{low_period,nonstationary,nonstationary_onesided}.yaml)
window_size=10

for forcing in sinusoidal_{low_period,nonstationary,nonstationary_onesided}.yaml; do
	for model in ensemble mlp bnn; do
		python ../src/train.py \
			experiment=${args[0]} \
			model=$model data=autoregressive \
			tags="[$forcing, autoregressive, $model]" \
			feature_names="[F, DeltaT, DeltaS]" \
			++model.net.input_dim=$window_size \
			++data.window_size=$window_size
	done
done
