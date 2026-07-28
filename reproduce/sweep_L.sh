#!/bin/bash
# Sweep L (seq_len) only, keeping the current best (P, R) per (dataset, pred_len)
# from reproduce/configs_luma_best.csv.
#
# Usage:
#   bash reproduce/sweep_L.sh                     # ETTh2 ETTm2 default, L in 96 192 336 720
#   DATASETS="ETTh2" LS="336 512 720" bash ...
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

GPU="${GPU:-0}"
SEED="${SEED:-2021}"
export CUDA_VISIBLE_DEVICES="${GPU}"

LOGDIR="${REPO_DIR}/logs_sweep"
RESULTS="${LOGDIR}/sweep_L_results.csv"
mkdir -p "${LOGDIR}"
echo "dataset,pred_len,seq_len,period_len,rank,mse,mae" > "${RESULTS}"

DATASETS="${DATASETS:-ETTh2 ETTm2}"
LS="${LS:-96 192 336 720}"

CONFIG="${REPO_DIR}/reproduce/configs_luma_best.csv"

for ds in ${DATASETS}; do
  # gather (pred_len, P, R) triples for this dataset from the config CSV
  mapfile -t rows < <(awk -F, -v d="${ds}" 'NR>1 && $1==d {print $2","$3","$4}' "${CONFIG}")
  case "${ds}" in
    ETTh1|ETTh2) freq=h ;;
    ETTm1|ETTm2) freq=t ;;
    *) freq=h ;;
  esac
  for row in "${rows[@]}"; do
    IFS=, read -r pl P R <<< "${row}"
    for L in ${LS}; do
      tag="${ds}_pred${pl}_L${L}_p${P}_r${R}"
      LOGFILE="${LOGDIR}/${tag}.log"
      if [ -s "${LOGFILE}" ] && grep -q 'mse:' "${LOGFILE}"; then
        echo "[skip] ${tag}"
      else
        echo "[run ] ${tag}"
        python -u run.py --task_name long_term_forecast --is_training 1 \
          --model LUMA --model_id "${tag}" \
          --data ${ds} --root_path ./dataset/ETT-small/ --data_path ${ds}.csv \
          --features M --freq ${freq} \
          --seq_len ${L} --label_len 48 --pred_len ${pl} \
          --enc_in 7 --dec_in 7 --c_out 7 --d_model 512 --dropout 0.001 \
          --period_len ${P} --rank ${R} \
          --learning_rate 0.02 --train_epochs 30 --batch_size 256 --patience 5 \
          --lradj type1 --loss MSE --des Exp --itr 1 --seed ${SEED} \
          > "${LOGFILE}" 2>&1
      fi
      mse=$(grep -oP 'mse:\s*\K[0-9.]+' "${LOGFILE}" | tail -1)
      mae=$(grep -oP 'mae:\s*\K[0-9.]+' "${LOGFILE}" | tail -1)
      echo "${ds},${pl},${L},${P},${R},${mse:-NA},${mae:-NA}" >> "${RESULTS}"
      echo "       -> mse=${mse:-NA}"
    done
  done
done
echo "Done. Results: ${RESULTS}"
