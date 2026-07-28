#!/bin/bash
# LUMA (LUMA) on weather -- best config per horizon
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
SEED="${SEED:-2021}"
mkdir -p ./logs

model_name=LUMA
root_path=./dataset/weather/
data_path=weather.csv
data=custom
seq_len=720

# P=2, R=8 is best across all horizons
for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --model ${model_name} --model_id "weather_${seq_len}_${pred_len}_p2_r8" \
    --data ${data} --root_path ${root_path} --data_path ${data_path} \
    --features M --freq t \
    --seq_len ${seq_len} --label_len 48 --pred_len ${pred_len} \
    --enc_in 21 --dec_in 21 --c_out 21 --d_model 512 --dropout 0.1 \
    --period_len 2 --rank 8 \
    --learning_rate 0.02 --train_epochs 30 --batch_size 256 --patience 5 \
    --lradj type1 --loss MSE --des Exp --itr 1 --seed ${SEED} \
    2>&1 | tee ./logs/weather_${pred_len}_p2_r8.log
done
