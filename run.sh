#!/bin/bash


python ./src/main.py \
    --n_in 2 \
    --n_h 200 \
    --n_out 1 \
    --num_layers 1 \
    --task Random \
    --randomType Uniform \
    --seq 100 \
    --numTr 10 \
    --numVl 10 \
    --numTe 5000 \
    --batch_size_tr 10 \
    --batch_size_vl 10 \
    --batch_size_te 5000 \
    --t1 5 \
    --t2 9 \
    --num_epochs 3500 \
    --learning_rate 0.1 \
    --optimizerFn SGD \
    --lossFn mse \
    --mode test \
    --checkpoint_freq 100 \
    --seed 9 \
    --projectName "rnn-test-ohotest2" \
    --logger "prettyprint" \
    --performance_samples 9 \
    --init_scheme ZeroInit \
    --activation_fn relu \
    --log_freq 2 \
    --meta_learning_rate 0.0001 \
    --l2_regularization 0 \
    --is_oho 1 \
    --time_chunk_size 10