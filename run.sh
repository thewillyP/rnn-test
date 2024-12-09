#!/bin/bash


python ./src/main.py \
    --n_in 2 \
    --n_h 30 \
    --n_out 1 \
    --num_layers 1 \
    --task Random \
    --init_scheme RandomInit \
    --seq 30 \
    --numTr 1000 \
    --numVl 10000 \
    --numTe 10000 \
    --batch_size_tr 100 \
    --batch_size_vl 100 \
    --batch_size_te 100 \
    --t1 5 \
    --t2 9 \
    --num_epochs 100 \
    --learning_rate 0.1 \
    --optimizerFn SGD \
    --lossFn cross_entropy \
    --mode test \
    --checkpoint_freq 100 \
    --seed 1 \
    --projectName "rnn-test-ffwdtest" \
    --logger "wandb" \
    --performance_samples 9 \
    --activation_fn tanh \
    --log_freq 1 \
    --meta_learning_rate 0.00001 \
    --l2_regularization 0.0 \
    --is_oho 1
