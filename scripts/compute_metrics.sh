#!/bin/bash

declare -a scenarios=("F1" "F2" "F3" "F4" "F5" "F6")
declare -a models=("mlp" "bnn" "ensemble_mlp")
declare -a metrics=("bias")

# AR
for metric in "${metrics[@]}"; do
    for scenario in "${scenarios[@]}"; do
        for model in "${models[@]}"; do
            ckpt_path="logs/train/runs/$scenario/$model/ar/csv/last.ckpt"
            data_path="data/$scenario.pkl"
            python scripts/compute_metric.py \
                job_name="$scenario/$model/ar" \
                metric_type="$metric" \
                ckpt_path=\"$ckpt_path\" \
                data.path=\"$data_path\" \
                model=$model \
                data.window_size=10
        done
    done
done

# PI
for metric in "${metrics[@]}"; do
    for scenario in "${scenarios[@]}"; do
        for model in "${models[@]}"; do
            ckpt_path="logs/train/runs/$scenario/$model/pi/csv/last.ckpt"
            data_path="data/$scenario.pkl"
            python scripts/compute_metric.py \
                job_name="$scenario/$model/pi" \
                metric_type="$metric" \
                ckpt_path=\"$ckpt_path\" \
                data.path=\"$data_path\" \
                model=$model
        done
    done
done
