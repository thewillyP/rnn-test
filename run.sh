#!/bin/bash


python ./src/main.py \
    --n_in 2 \
    --n_h 200 \
    --n_out 1 \
    --num_layers 1 \
    --task Random \
    --randomType Uniform \
    --seq 10 \
    --numTr 1000 \
    --numVl 1000 \
    --numTe 5000 \
    --batch_size_tr 1000 \
    --batch_size_vl 1000 \
    --batch_size_te 5000 \
    --t1 3 \
    --t2 5 \
    --num_epochs 3500 \
    --learning_rate 0.1 \
    --optimizerFn SGD \
    --lossFn mse \
    --mode test \
    --checkpoint_freq 100 \
    --seed 9 \
    --projectName "rnn-test-ohotest2" \
    --logger "wandb" \
    --performance_samples 9 \
    --init_scheme ZeroInit \
    --activation_fn relu \
    --log_freq 1 \
    --meta_learning_rate 0.0001 \
    --l2_regularization 0 \
    --is_oho 0