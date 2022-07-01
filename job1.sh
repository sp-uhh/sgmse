#!/bin/sh

export CUDA_VISIBLE_DEVICES="0,1,2,3"

spec_abs_exponent=0.1   
for spec_factor in 0.7 0.8 0.9
do
    python train.py --base_dir /data/richter/wsj0_chime3/ --batch_size 8 --gpus 4 --eval_start 100 --max_epochs_110 --num_eval_files 20 --spec_abs_exponent $spec_abs_exponent --spec_factor $spec_factor
done

spec_abs_exponent=0.2   
for spec_factor in 0.4 0.5 0.6
do
    python train.py --base_dir /data/richter/wsj0_chime3/ --batch_size 8 --gpus 4 --eval_start 100 --max_epochs_110 --num_eval_files 20 --spec_abs_exponent $spec_abs_exponent --spec_factor $spec_factor
done

spec_abs_exponent=0.3   
for spec_factor in 0.3 0.4 0.5
do
    python train.py --base_dir /data/richter/wsj0_chime3/ --batch_size 8 --gpus 4 --eval_start 100 --max_epochs_110 --num_eval_files 20 --spec_abs_exponent $spec_abs_exponent --spec_factor $spec_factor
done

spec_abs_exponent=0.4   
for spec_factor in 0.2 0.25 0.3
do
    python train.py --base_dir /data/richter/wsj0_chime3/ --batch_size 8 --gpus 4 --eval_start 100 --max_epochs_110 --num_eval_files 20 --spec_abs_exponent $spec_abs_exponent --spec_factor $spec_factor
done

spec_abs_exponent=0.5  
for spec_factor in 0.15 0.2 0.25
do
    python train.py --base_dir /data/richter/wsj0_chime3/ --batch_size 8 --gpus 4 --eval_start 100 --max_epochs_110 --num_eval_files 20 --spec_abs_exponent $spec_abs_exponent --spec_factor $spec_factor
done

spec_abs_exponent=0.6  
for spec_factor in 0.1 0.15 0.2
do
    python train.py --base_dir /data/richter/wsj0_chime3/ --batch_size 8 --gpus 4 --eval_start 100 --max_epochs_110 --num_eval_files 20 --spec_abs_exponent $spec_abs_exponent --spec_factor $spec_factor
done

spec_abs_exponent=0.7
for spec_factor in 0.07 0.1 
do
    python train.py --base_dir /data/richter/wsj0_chime3/ --batch_size 8 --gpus 4 --eval_start 100 --max_epochs_110 --num_eval_files 20 --spec_abs_exponent $spec_abs_exponent --spec_factor $spec_factor
done

spec_abs_exponent=0.8
for spec_factor in 0.05 0.1
do
    python train.py --base_dir /data/richter/wsj0_chime3/ --batch_size 8 --gpus 4 --eval_start 100 --max_epochs_110 --num_eval_files 20 --spec_abs_exponent $spec_abs_exponent --spec_factor $spec_factor
done

spec_abs_exponent=0.9
for spec_factor in 0.04 0.06
do
    python train.py --base_dir /data/richter/wsj0_chime3/ --batch_size 8 --gpus 4 --eval_start 100 --max_epochs_110 --num_eval_files 20 --spec_abs_exponent $spec_abs_exponent --spec_factor $spec_factor
done

spec_abs_exponent=1.0
for spec_factor in 0.03 0.05
do
    python train.py --base_dir /data/richter/wsj0_chime3/ --batch_size 8 --gpus 4 --eval_start 100 --max_epochs_110 --num_eval_files 20 --spec_abs_exponent $spec_abs_exponent --spec_factor $spec_factor
done