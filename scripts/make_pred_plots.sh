#!/bin/bash

declare -a scenarios=("F1" "F2" "F3" "F4" "F5" "F6")
declare -a models=("mlp" "bnn" "ensemble_mlp")

# AR
for scenario in "${scenarios[@]}"; do
    for model in "${models[@]}"; do
        python scripts/plot_csv.py \
            --csv_filenames="[\"logs/metric/runs/$scenario/$model/ar/train_prediction.csv\",\"logs/metric/runs/$scenario/$model/ar/val_prediction.csv\"]" \
            --text.usetex=True - \
            lineplot '["Ground Truth: Train"]' --y="train/y" - \
            lineplot '["Prediction: Train"]' --y="train/pred_mean" - \
            bands "train/pred_mean" "train/pred_std" - \
            lineplot '["Ground Truth: Val"]' --y="val/y" --shift_x=True - \
            lineplot '["Prediction: Val"]' --y="val/pred_mean" --shift_x=True - \
            bands "val/pred_mean" "val/pred_std" --shift_x=True - \
            set '\(\tau\)' '\(q\)' \
            tight_layout \
            legend \
            savefig plots/$scenario/${scenario}_${model}_ar_pred.pdf
    done
done

# PI
for scenario in "${scenarios[@]}"; do
    for model in "${models[@]}"; do
        python scripts/plot_csv.py \
            --csv_filenames="[\"logs/metric/runs/$scenario/$model/pi/train_prediction.csv\",\"logs/metric/runs/$scenario/$model/pi/val_prediction.csv\"]" \
            --text.usetex=True - \
            lineplot '["Ground Truth: Train"]' --y="train/y" - \
            lineplot '["Prediction: Train"]' --y="train/pred_mean" - \
            bands "train/pred_mean" "train/pred_std" - \
            lineplot '["Ground Truth: Val"]' --y="val/y" --shift_x=True - \
            lineplot '["Prediction: Val"]' --y="val/pred_mean" --shift_x=True - \
            bands "val/pred_mean" "val/pred_std" --shift_x=True - \
            set '\(\tau\)' '\(q\)' \
            tight_layout \
            legend \
            savefig plots/$scenario/${scenario}_${model}_pi_pred.pdf
    done
done
