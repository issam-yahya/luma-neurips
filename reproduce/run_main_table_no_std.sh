#!/bin/bash
# =============================================================================
# LUMA / MinimalTS  --  main results table reproduction driver
# =============================================================================
# Reproduces every reported best-configuration result (the main table) by
# re-running run.py with the exact hyperparameters recorded in the original
# training logs (reproduce/configs_luma_best.csv).
#
# Usage:
#   bash scripts/reproduce/run_main_table.sh                 # all datasets
#   bash scripts/reproduce/run_main_table.sh ETTh1 ETTh2     # only these
#   GPU=1 bash scripts/reproduce/run_main_table.sh           # pick CUDA device
#   SEED=2021 bash scripts/reproduce/run_main_table.sh       # override seed
#
# Outputs:
#   logs_reproduce/<dataset>_<pred>_p<P>_r<R>.log    one log per config
#   logs_reproduce/reproduced_results.csv            dataset,pred_len,P,R,mse,mae,params
#
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

CONFIG="${REPO_DIR}/reproduce/configs_luma_best.csv"
LOGDIR="${REPO_DIR}/logs_reproduce"
RESULTS="${LOGDIR}/reproduced_results.csv"
mkdir -p "${LOGDIR}"

GPU="${GPU:-0}"
SEED="${SEED:-2021}"         # seed the reported best numbers were produced with
                              # (verified: seeds 2021-2025 reproduce within ~0.3% MSE)
export CUDA_VISIBLE_DEVICES="${GPU}"

FILTER=" $* "

echo "dataset,pred_len,P,R,mse,mae,params" > "${RESULTS}"

tail -n +2 "${CONFIG}" | while IFS=, read -r dataset pred_len period_len rank seq_len \
    label_len enc_in d_model dropout learning_rate train_epochs batch_size patience \
    data root_path data_path freq ref_mse ref_mae ref_params; do

    if [ "$#" -gt 0 ] && [[ "${FILTER}" != *" ${dataset} "* ]]; then
        continue
    fi

    tag="${dataset}_${pred_len}_p${period_len}_r${rank}"
    LOGFILE="${LOGDIR}/${tag}.log"

    if [ -s "${LOGFILE}" ] && grep -q 'mse:' "${LOGFILE}"; then
        echo "[skip] ${tag} (already done)"
    else
        echo "[run ] ${tag}  seq=${seq_len} lr=${learning_rate} bs=${batch_size} ep=${train_epochs}"
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --model LUMA \
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
            --seed "${SEED}" \
            > "${LOGFILE}" 2>&1
    fi

    mse=$(grep -oP 'mse:\s*\K[0-9.]+' "${LOGFILE}" | tail -1)
    mae=$(grep -oP 'mae:\s*\K[0-9.]+' "${LOGFILE}" | tail -1)
    params=$(grep -oiP 'total\s+trainable\s+params:\s*\K[0-9,]+' "${LOGFILE}" | tail -1 | tr -d ,)
    echo "${dataset},${pred_len},${period_len},${rank},${mse:-NA},${mae:-NA},${params:-NA}" >> "${RESULTS}"
    echo "       -> mse=${mse:-NA} mae=${mae:-NA}"
done

echo
echo "Done. Results: ${RESULTS}"
echo "Verify with:  python reproduce/check_reproduction.py"
