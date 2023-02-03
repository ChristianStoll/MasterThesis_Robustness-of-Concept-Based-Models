#!/usr/bin/env bash
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 100 --use_slot false --vis false --channel 2048 --dataset_dir ../data/CUB_200_2011/ --num_workers 0
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 100 --use_slot true --use_pre true --loss_status 1 --slots_per_class 5 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 100 --use_slot true --use_pre true --loss_status -1 --slots_per_class 3 --power 2 --to_k_layer 3 --lambda_value 1. --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0