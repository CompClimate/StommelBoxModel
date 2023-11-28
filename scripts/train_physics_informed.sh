#!/bin/bash

args=("$@")

for forcing in sinusoidal_{low_period,nonstationary,nonstationary_onesided}.yaml; do
	for model in mlp bnn ensemble; do
		python ../src/train.py \
			experiment=${args[0]} \
			model=$model \
			data=physics_informed \
			feature_names="[F, DeltaT, DeltaS]"
	done
done
