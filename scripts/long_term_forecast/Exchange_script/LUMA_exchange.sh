#!/bin/bash
# LUMA (LUMA) on exchange_rate -- best config per horizon
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
SEED="${SEED:-2021}"
mkdir -p ./logs

model_name=LUMA
root_path=./dataset/exchange_rate/
data_path=exchange_rate.csv
data=custom
seq_len=720

declare -A P R
P[96]=2;  R[96]=4
P[192]=2; R[192]=6
P[336]=4; R[336]=10
P[720]=2; R[720]=10

for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --model ${model_name} --model_id "exchange_${seq_len}_${pred_len}_p${P[$pred_len]}_r${R[$pred_len]}" \
    --data ${data} --root_path ${root_path} --data_path ${data_path} \
    --features M --freq d \
    --seq_len ${seq_len} --label_len 48 --pred_len ${pred_len} \
    --enc_in 8 --dec_in 8 --c_out 8 --d_model 512 --dropout 0.1 \
    --period_len ${P[$pred_len]} --rank ${R[$pred_len]} \
    --learning_rate 0.005 --train_epochs 30 --batch_size 32 --patience 5 \
    --lradj type1 --loss MSE --des Exp --itr 1 --seed ${SEED} \
    2>&1 | tee ./logs/exchange_${pred_len}_p${P[$pred_len]}_r${R[$pred_len]}.log
done
