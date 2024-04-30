#!/bin/bash

args=("$@")

forcings=$(ls -1 ../configs/forcing | sed -e 's/\.yaml$//')
declare -a models=("mlp" "bnn" "ensemble")

for forcing in $forcings; do
	for model in "${models[@]}"; do
		python ../src/plot_landscape.py \
			model=$model \
			forcing=$forcing \
			training_path="'../logs/train/runs/$forcing/$model/physics_informed/training_path/training_path.pkl'" \
			checkpoint_path="'../logs/train/runs/$forcing/$model/physics_informed/csv/version_0/checkpoints/epoch=119-step=2640.ckpt'"
	done
done
