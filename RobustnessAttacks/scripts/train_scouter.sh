#!/usr/bin/env bash
# scouter 100 seed 1
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 100 --use_slot false --vis false --channel 2048 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 100_classes_seed1/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 100 --use_slot true --use_pre true --loss_status 1 --slots_per_class 5 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 100_classes_seed1/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 100 --use_slot true --use_pre true --loss_status -1 --slots_per_class 3 --power 2 --to_k_layer 3 --lambda_value 1. --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 100_classes_seed1/
# scouter 100 seed 2
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 100 --use_slot false --vis false --channel 2048 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 100_classes_seed2/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 100 --use_slot true --use_pre true --loss_status 1 --slots_per_class 5 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 100_classes_seed2/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 100 --use_slot true --use_pre true --loss_status -1 --slots_per_class 3 --power 2 --to_k_layer 3 --lambda_value 1. --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 100_classes_seed2/
# scouter 100 seed 3
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 100 --use_slot false --vis false --channel 2048 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 100_classes_seed3/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 100 --use_slot true --use_pre true --loss_status 1 --slots_per_class 5 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 100_classes_seed3/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 100 --use_slot true --use_pre true --loss_status -1 --slots_per_class 3 --power 2 --to_k_layer 3 --lambda_value 1. --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 100_classes_seed3/

#!/usr/bin/env bash
# scouter 25 1 slot, var lambda
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot false --vis false --channel 2048 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lv_seed1/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot true --use_pre true --loss_status 1 --slots_per_class 1 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lv_seed1/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot true --use_pre true --loss_status -1 --slots_per_class 1 --power 2 --to_k_layer 3 --lambda_value 1. --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lv_seed1/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot false --vis false --channel 2048 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lv_seed2/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot true --use_pre true --loss_status 1 --slots_per_class 1 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lv_seed2/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot true --use_pre true --loss_status -1 --slots_per_class 1 --power 2 --to_k_layer 3 --lambda_value 1. --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lv_seed2/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot false --vis false --channel 2048 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lv_seed3/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot true --use_pre true --loss_status 1 --slots_per_class 1 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lv_seed3/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot true --use_pre true --loss_status -1 --slots_per_class 1 --power 2 --to_k_layer 3 --lambda_value 1. --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lv_seed3/

# scouter 25 1 slot, lambda 10
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot false --vis false --channel 2048 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lc_seed1/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot true --use_pre true --loss_status 1 --slots_per_class 1 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lc_seed1/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot true --use_pre true --loss_status -1 --slots_per_class 1 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lc_seed1/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot false --vis false --channel 2048 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lc_seed2/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot true --use_pre true --loss_status 1 --slots_per_class 1 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lc_seed2/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot true --use_pre true --loss_status -1 --slots_per_class 1 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lc_seed2/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot false --vis false --channel 2048 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lc_seed3/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot true --use_pre true --loss_status 1 --slots_per_class 1 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lc_seed3/
python3 train.py --dataset CUB200 --model resnest50d --batch_size 32 --epochs 150 --num_classes 25 --use_slot true --use_pre true --loss_status -1 --slots_per_class 1 --power 2 --to_k_layer 3 --lambda_value 10 --vis false --channel 2048 --freeze_layers 2 --dataset_dir ../data/CUB_200_2011/ --num_workers 0 --output_dir 25_classes_s1_lc_seed3/