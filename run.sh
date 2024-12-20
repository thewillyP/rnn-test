#!/bin/bash

python ./src/main.py \
    --n_in 2 \
    --n_h 30 \
    --n_out 1 \
    --num_layers 1 \
    --task Random \
    --randomType Uniform \
    --seq 100 \
    --numTr 4 \
    --numVl 2 \
    --numTe 5000 \
    --batch_size_tr 4 \
    --batch_size_vl 2 \
    --batch_size_te 5000 \
    --t1 3 \
    --t2 5 \
    --num_epochs 200 \
    --learning_rate 0.01 \
    --optimizerFn SGD \
    --lossFn mse \
    --mode experiment \
    --checkpoint_freq 20 \
    --seed 2 \
    --projectName "mlr-test" \
    --logger "wandb" \
    --performance_samples 9 \
    --init_scheme StaticRandomInit \
    --activation_fn tanh \
    --log_freq 1 \
    --meta_learning_rate 0.0002 \
    --l2_regularization 0.0 \
    --is_oho 1 \
    --time_chunk_size 10
