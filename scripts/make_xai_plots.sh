#!/bin/bash

declare -a scenarios=("F1" "F2" "F3" "F4" "F5" "F6")
declare -a models=("mlp" "bnn" "ensemble_mlp")
declare -a methods=("shap" "deeplift")

# AR
for scenario in "${scenarios[@]}"; do
    for model in "${models[@]}"; do
        for method in "${methods[@]}"; do
            python scripts/plot_csv.py \
                --csv_filenames="[\"logs/explain/runs/$scenario/$model/ar/explanations.csv\"]" \
                --text.usetex=True - \
                shap_heatmap $method \
                tight_layout \
                savefig plots/$scenario/${scenario}_${model}_ar_${method}.pdf
                # shap_heatmap "[\"${method}_1\",\"${method}_2\",\"${method}_3\",\"${method}_4\",\"${method}_5\",\"${method}_6\"]" \
                # --feature_names="[\"\(S_1\)\",\"\(S_2\)\",\"\(T_1\)\",\"\(T_2\)\",\"\(F_s\)\",\"\(F_t\)\"]" - \
        done
    done
done

# PI
for scenario in "${scenarios[@]}"; do
    for model in "${models[@]}"; do
        for method in "${methods[@]}"; do
            python scripts/plot_csv.py \
                --csv_filenames="[\"logs/explain/runs/$scenario/$model/pi/explanations.csv\"]" \
                --text.usetex=True - \
                shap_heatmap $method \
                tight_layout \
                savefig plots/$scenario/${scenario}_${model}_pi_${method}.pdf
                # shap_heatmap "[\"${method}_1\",\"${method}_2\",\"${method}_3\",\"${method}_4\",\"${method}_5\",\"${method}_6\"]" \
                # --feature_names="[\"\(S_1\)\",\"\(S_2\)\",\"\(T_1\)\",\"\(T_2\)\",\"\(F_s\)\",\"\(F_t\)\"]" - \
        done
    done
done
