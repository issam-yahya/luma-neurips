#!/bin/bash
# LUMA (LUMA) on ETTm2 -- best config per horizon
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
SEED="${SEED:-2021}"
mkdir -p ./logs

model_name=LUMA
root_path=./dataset/ETT-small/
data_path=ETTm2.csv
data=ETTm2
seq_len=336

declare -A P R
P[96]=2;  R[96]=10
P[192]=2; R[192]=10
P[336]=2; R[336]=10
P[720]=8; R[720]=9

for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --model ${model_name} --model_id "${data}_${seq_len}_${pred_len}_p${P[$pred_len]}_r${R[$pred_len]}" \
    --data ${data} --root_path ${root_path} --data_path ${data_path} \
    --features M --freq t \
    --seq_len ${seq_len} --label_len 48 --pred_len ${pred_len} \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 512 --dropout 0.001 \
    --period_len ${P[$pred_len]} --rank ${R[$pred_len]} \
    --learning_rate 0.03 --train_epochs 30 --batch_size 256 --patience 5 \
    --lradj type1 --loss MSE --des Exp --itr 1 --seed ${SEED} \
    2>&1 | tee ./logs/ETTm2_${pred_len}_p${P[$pred_len]}_r${R[$pred_len]}.log
done
