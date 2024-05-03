#!/bin/bash

declare -a scenarios=("F1" "F2" "F3" "F4" "F5" "F6")
declare -a models=("mlp" "bnn" "ensemble_mlp")

# AR
for scenario in "${scenarios[@]}"; do
    for model in "${models[@]}"; do
		echo "=== AR | $scenario | $model ==="
        python scripts/plot_csv.py \
            --csv_filenames="[\"logs/metric/runs/$scenario/$model/ar/train_bias.csv\",\"logs/metric/runs/$scenario/$model/ar/val_bias.csv\"]" \
            --text.usetex=True - \
        	lineplot '["Training"]' --y="train/bias" - \
			lineplot '["Validation"]' --y="val/bias" --shift_x=True - \
			set "\(\tau\)" "Bias" \
            tight_layout \
            legend \
            savefig plots/$scenario/${scenario}_${model}_ar_bias.pdf
    done
done

# PI
for scenario in "${scenarios[@]}"; do
    for model in "${models[@]}"; do
		echo "=== PI | $scenario | $model ==="
        python scripts/plot_csv.py \
            --csv_filenames="[\"logs/metric/runs/$scenario/$model/pi/train_bias.csv\",\"logs/metric/runs/$scenario/$model/pi/val_bias.csv\"]" \
            --text.usetex=True - \
            lineplot '["Training"]' --y="train/bias" - \
			lineplot '["Validation"]' --y="val/bias" --shift_x=True - \
			set "\(\tau\)" "Bias" \
            tight_layout \
            legend \
            savefig plots/$scenario/${scenario}_${model}_pi_bias.pdf
    done
done
