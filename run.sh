#!/bin/bash

python ./src/main.py \
    --n_in 2 \
    --n_h 200 \
    --n_out 1 \
    --num_layers 1 \
    --task Random \
    --init_scheme RandomInit \
    --seq 10 \
    --numTr 1000 \
    --numVl 1 \
    --numTe 5000 \
    --batch_size_tr 1000 \
    --batch_size_vl 1 \
    --batch_size_te 1000 \
    --t1 2 \
    --t2 2 \
    --num_epochs 5000 \
    --learning_rate 0.01 \
    --optimizerFn Adam \
    --lossFn mse \
    --mode test \
    --checkpoint_freq 100 \
    --seed 1 \
    --projectName "rnn-test-adamtest2" \
    --logger "wandb" \
    --performance_samples 9 \
    --activation_fn tanh \
    --log_freq 1
