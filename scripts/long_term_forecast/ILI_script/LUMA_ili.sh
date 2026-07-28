#!/bin/bash
# LUMA (LUMA) on illness (ILI) -- best config per horizon
# Note: seq_len=104 and label_len=18 (small dataset), and freq must be 'h'
# (the loader crashes with freq='w').
set -u
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
SEED="${SEED:-2021}"
mkdir -p ./logs

model_name=LUMA
root_path=./dataset/illness/
data_path=national_illness.csv
data=custom
seq_len=104

# pred_len -> (period_len, rank, lr, epochs, batch_size, patience)
declare -A P R LR EP BS PAT
P[24]=2;  R[24]=6;  LR[24]=0.03;  EP[24]=50;  BS[24]=16; PAT[24]=10
P[36]=2;  R[36]=8;  LR[36]=0.03;  EP[36]=50;  BS[36]=16; PAT[36]=10
P[48]=2;  R[48]=10; LR[48]=0.01;  EP[48]=100; BS[48]=8;  PAT[48]=15
P[60]=4;  R[60]=16; LR[60]=0.01;  EP[60]=100; BS[60]=8;  PAT[60]=15

for pred_len in 24 36 48 60; do
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --model ${model_name} --model_id "ili_${seq_len}_${pred_len}_p${P[$pred_len]}_r${R[$pred_len]}" \
    --data ${data} --root_path ${root_path} --data_path ${data_path} \
    --features M --freq h \
    --seq_len ${seq_len} --label_len 18 --pred_len ${pred_len} \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 512 --dropout 0.1 \
    --period_len ${P[$pred_len]} --rank ${R[$pred_len]} \
    --learning_rate ${LR[$pred_len]} --train_epochs ${EP[$pred_len]} \
    --batch_size ${BS[$pred_len]} --patience ${PAT[$pred_len]} \
    --lradj type1 --loss MSE --des Exp --itr 1 --seed ${SEED} \
    2>&1 | tee ./logs/ili_${pred_len}_p${P[$pred_len]}_r${R[$pred_len]}.log
done
