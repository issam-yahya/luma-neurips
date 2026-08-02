#!/bin/bash
# =============================================================================
# LUMA / MinimalTS  --  fixed lookback (seq_len=720) ablation
# =============================================================================
# Reruns every (dataset, pred_len) config from configs_luma_best.csv but forces
# seq_len=720 for all of them, regardless of what seq_len the original best
# config used. This isolates the effect of period_len/rank from lookback
# length -- all main-table configs already use seq_len=720, so this script
# mainly matters if configs_luma_best.csv is later extended with configs that
# used a different lookback.
#
# Usage:
#   bash scripts/reproduce/run_fixed_lookback.sh                # all datasets
#   bash scripts/reproduce/run_fixed_lookback.sh ETTh1 traffic  # subset
#   GPU=1 bash scripts/reproduce/run_fixed_lookback.sh
#   SEED=2021 bash scripts/reproduce/run_fixed_lookback.sh
#
# Outputs:
#   logs_reproduce_fixed_lookback/<dataset>_<pred>_p<P>_r<R>_sl720.log
#   logs_reproduce_fixed_lookback/reproduced_results_sl720.csv
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

FIXED_SEQ_LEN=720

CONFIG="${REPO_DIR}/reproduce/configs_luma_best.csv"
LOGDIR="${REPO_DIR}/logs_reproduce_fixed_lookback"
RESULTS="${LOGDIR}/reproduced_results_sl${FIXED_SEQ_LEN}.csv"
mkdir -p "${LOGDIR}"

GPU="${GPU:-0}"
SEED="${SEED:-2021}"
export CUDA_VISIBLE_DEVICES="${GPU}"

FILTER=" $* "

echo "dataset,pred_len,P,R,seq_len,mse,mae,params" > "${RESULTS}"

tail -n +2 "${CONFIG}" | while IFS=, read -r dataset pred_len period_len rank seq_len \
    label_len enc_in d_model dropout learning_rate train_epochs batch_size patience \
    data root_path data_path freq ref_mse ref_mae ref_params; do

    if [ "$#" -gt 0 ] && [[ "${FILTER}" != *" ${dataset} "* ]]; then
        continue
    fi

    tag="${dataset}_${pred_len}_p${period_len}_r${rank}_sl${FIXED_SEQ_LEN}"
    LOGFILE="${LOGDIR}/${tag}.log"

    if [ -s "${LOGFILE}" ] && grep -q 'mse:' "${LOGFILE}"; then
        echo "[skip] ${tag} (already done)"
    else
        echo "[run ] ${tag}  seq=${FIXED_SEQ_LEN} (orig=${seq_len}) lr=${learning_rate} bs=${batch_size} ep=${train_epochs}"
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
            --seq_len "${FIXED_SEQ_LEN}" \
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
    echo "${dataset},${pred_len},${period_len},${rank},${FIXED_SEQ_LEN},${mse:-NA},${mae:-NA},${params:-NA}" >> "${RESULTS}"
    echo "       -> mse=${mse:-NA} mae=${mae:-NA}"
done

echo
echo "Done. Results: ${RESULTS}"
