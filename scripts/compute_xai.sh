#!/bin/bash

declare -a scenarios=("F1" "F2" "F3" "F4" "F5" "F6")
declare -a models=("mlp" "bnn" "ensemble_mlp")
declare -a methods=("shap" "deeplift")

# AR
for scenario in "${scenarios[@]}"; do
    for model in "${models[@]}"; do
        python src/explain.py \
            job_name="$scenario/$model/ar" \
            experiment=training \
            model=$model \
            ckpt_path="\"logs/train/runs/$scenario/$model/ar/csv/last.ckpt\"" \
            data.window_size=10 \
            ++data.path="\"data/$scenario.pkl\""
    done
done

# PI
for scenario in "${scenarios[@]}"; do
    for model in "${models[@]}"; do
        python src/explain.py \
            job_name="$scenario/$model/pi" \
            experiment=training \
            model=$model \
            ckpt_path="\"logs/train/runs/$scenario/$model/pi/csv/last.ckpt\"" \
            ++data.path="\"data/$scenario.pkl\""
    done
done
