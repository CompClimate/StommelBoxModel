#!/bin/bash

args=("$@")

forcings=$(ls -1 ../configs/forcing | sed -e 's/\.yaml$//')
declare -a models=("mlp" "bnn" "ensemble" "rnn" "lstm" "gru" "conv")
window_size=10

for forcing in $forcings; do
	for model in "${models[@]}"; do
		python ../src/plot_landscape.py \
			model=$model \
			forcing=$forcing \
			data.autoregressive=True \
			data.window_size=$window_size \
			training_path="'../logs/train/runs/$forcing/$model/autoregressive/training_path/training_path.pkl'" \
			checkpoint_path="'../logs/train/runs/$forcing/$model/autoregressive/csv/version_0/checkpoints/epoch=119-step=2640.ckpt'"
	done
done
