#!/bin/bash

args=("$@")

for forcing in sinusoidal_{low_period,nonstationary,nonstationary_onesided}; do
	for model in mlp bnn ensemble; do
		python ../src/train.py \
			experiment=${args[0]} \
			model=$model \
			forcing=$forcing \
			data=physics_informed \
			feature_names="[F, DeltaT, DeltaS]"
	done
done
