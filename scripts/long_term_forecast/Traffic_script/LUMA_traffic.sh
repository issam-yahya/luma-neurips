#!/bin/bash
# LUMA (LUMA) on traffic -- best config per horizon
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
SEED="${SEED:-2021}"
mkdir -p ./logs

model_name=LUMA
root_path=./dataset/traffic/
data_path=traffic.csv
data=custom
seq_len=720

declare -A P R
P[96]=24;  R[96]=8
P[192]=24; R[192]=10
P[336]=24; R[336]=12
P[720]=24; R[720]=10

for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --model ${model_name} --model_id "traffic_${seq_len}_${pred_len}_p${P[$pred_len]}_r${R[$pred_len]}" \
    --data ${data} --root_path ${root_path} --data_path ${data_path} \
    --features M --freq h \
    --seq_len ${seq_len} --label_len 48 --pred_len ${pred_len} \
    --enc_in 862 --dec_in 862 --c_out 862 --d_model 512 --dropout 0.1 \
    --period_len ${P[$pred_len]} --rank ${R[$pred_len]} \
    --learning_rate 0.03 --train_epochs 30 --batch_size 128 --patience 5 \
    --lradj type1 --loss MSE --des Exp --itr 1 --seed ${SEED} \
    2>&1 | tee ./logs/traffic_${pred_len}_p${P[$pred_len]}_r${R[$pred_len]}.log
done
