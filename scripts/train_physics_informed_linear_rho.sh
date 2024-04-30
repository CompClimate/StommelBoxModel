#!/bin/bash

args=("$@")

forcings=$(ls -1 ../configs/s_forcing | sed -e 's/\.yaml$//')
declare -a models=("mlp" "bnn" "ensemble")

for forcing in $forcings; do
	for model in "${models[@]}"; do
		python ../src/train.py \
			experiment=${args[0]} \
			model=$model \
			s_forcing=$forcing \
			data=physics_informed
	done
done
