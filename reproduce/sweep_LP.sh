#!/bin/bash
# Sweep (L, P) for ETTh2 and ETTm2 to find the config that matches the paper.
# R is kept at the current best (10 for pred_len<720; 9 for ETTm2/720).
# Logs go to logs_sweep/. A summary CSV is written for post-processing.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

GPU="${GPU:-0}"
SEED="${SEED:-2021}"
export CUDA_VISIBLE_DEVICES="${GPU}"
LOGDIR="${REPO_DIR}/logs_sweep"
RESULTS="${LOGDIR}/sweep_results.csv"
mkdir -p "${LOGDIR}"
echo "dataset,pred_len,seq_len,period_len,rank,mse,mae" > "${RESULTS}"

DATASETS="${DATASETS:-ETTh2 ETTm2}"
LS="${LS:-96 192 336 720}"
PS="${PS:-2 4 8 12 24}"

for ds in ${DATASETS}; do
  case "${ds}" in
    ETTh1|ETTh2) freq=h ;;
    ETTm1|ETTm2) freq=t ;;
  esac
  for pl in 96 192 336 720; do
    # keep the (already-selected) rank per pred_len
    case "${pl}" in 720) R=9;; *) R=10;; esac
    for L in ${LS}; do
      for P in ${PS}; do
        # sanity: skip if L / P < 1
        [ $(( L / P )) -lt 1 ] && continue
        tag="${ds}_L${L}_pred${pl}_p${P}_r${R}"
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
done
echo "Done. Sweep results: ${RESULTS}"
