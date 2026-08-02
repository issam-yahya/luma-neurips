#!/bin/bash
# =============================================================================
# LUMA / MinimalTS  --  seed-stability sweep (seeds 2021-2025)
# =============================================================================
# Reruns every (dataset, pred_len) config from reproduce/configs_luma_best.csv
# across 5 seeds, to verify the reported numbers aren't a cherry-picked draw.
# reproduce/README.md already documents that seeds 2021-2025 vary MSE by only
# ~0.3%; this script is what produces the raw per-seed numbers behind that
# claim, in the scripts/reproduce/ layout.
#
# Usage:
#   bash scripts/reproduce/run_five_seeds.sh                # all datasets
#   bash scripts/reproduce/run_five_seeds.sh ETTh1 traffic  # subset
#   GPU=1 bash scripts/reproduce/run_five_seeds.sh
#   SEEDS="2021 2022" bash scripts/reproduce/run_five_seeds.sh
#
# Outputs:
#   logs_reproduce_multiseed/<dataset>_<pred>_p<P>_r<R>_seed<seed>.log
#   logs_reproduce_multiseed/multiseed_results.csv
#     columns: dataset,pred_len,P,R,seed,mse,mae,params
#
# Then aggregate:
#   python scripts/reproduce/aggregate_seeds.py
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

CONFIG="${REPO_DIR}/reproduce/configs_luma_best.csv"
LOGDIR="${REPO_DIR}/logs_reproduce_multiseed"
RESULTS="${LOGDIR}/multiseed_results.csv"
mkdir -p "${LOGDIR}"

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

read -ra SEED_LIST <<< "${SEEDS:-2021 2022 2023 2024 2025}"

FILTER=" $* "

echo "dataset,pred_len,P,R,seed,mse,mae,params" > "${RESULTS}"

tail -n +2 "${CONFIG}" | while IFS=, read -r dataset pred_len period_len rank seq_len \
    label_len enc_in d_model dropout learning_rate train_epochs batch_size patience \
    data root_path data_path freq ref_mse ref_mae ref_params; do

    if [ "$#" -gt 0 ] && [[ "${FILTER}" != *" ${dataset} "* ]]; then
        continue
    fi

    for seed in "${SEED_LIST[@]}"; do
        tag="${dataset}_${pred_len}_p${period_len}_r${rank}_seed${seed}"
        LOGFILE="${LOGDIR}/${tag}.log"

        if [ -s "${LOGFILE}" ] && grep -q 'mse:' "${LOGFILE}"; then
            echo "[skip] ${tag} (already done)"
        else
            echo "[run ] ${tag}  seq=${seq_len} lr=${learning_rate} bs=${batch_size} ep=${train_epochs}"
            python -u run.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --model MinimalTS \
                --model_id "${tag}" \
                --data "${data}" \
                --root_path "${root_path}" \
                --data_path "${data_path}" \
                --features M \
                --freq "${freq}" \
                --seq_len "${seq_len}" \
                --label_len "${label_len}" \
                --pred_len "${pred_len}" \
                --enc_in "${enc_in}" \
                --dec_in "${enc_in}" \
                --c_out "${enc_in}" \
                --d_model "${d_model}" \
                --dropout "${dropout}" \
                --period_len "${period_len}" \
                --rank "${rank}" \
                --learning_rate "${learning_rate}" \
                --train_epochs "${train_epochs}" \
                --batch_size "${batch_size}" \
                --patience "${patience}" \
                --lradj type1 \
                --loss MSE \
                --des Exp \
                --itr 1 \
                --seed "${seed}" \
                > "${LOGFILE}" 2>&1
        fi

        mse=$(grep -oP 'mse:\s*\K[0-9.]+' "${LOGFILE}" | tail -1)
        mae=$(grep -oP 'mae:\s*\K[0-9.]+' "${LOGFILE}" | tail -1)
        params=$(grep -oiP 'total\s+trainable\s+params:\s*\K[0-9,]+' "${LOGFILE}" | tail -1 | tr -d ,)
        echo "${dataset},${pred_len},${period_len},${rank},${seed},${mse:-NA},${mae:-NA},${params:-NA}" >> "${RESULTS}"
        echo "       -> seed=${seed} mse=${mse:-NA} mae=${mae:-NA}"
    done
done

echo
echo "Done. Results: ${RESULTS}"
echo "Aggregate with:  python scripts/reproduce/aggregate_seeds.py"
