#!/usr/bin/env bash
python3 ConceptBottleneck/experiments.py cub Standard --seed 1 -ckpt 1 -log_dir StandardNoBNModel_Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -data_dir data/CUB_processed/class_attr_data_10 -b 32 -weight_decay 0.0004 -lr 0.01 -scheduler_step 20
python3 ConceptBottleneck/experiments.py cub Standard --seed 2 -ckpt 1 -log_dir StandardNoBNModel_Seed2/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -data_dir data/CUB_processed/class_attr_data_10 -b 32 -weight_decay 0.0004 -lr 0.01 -scheduler_step 20
python3 ConceptBottleneck/experiments.py cub Standard --seed 3 -ckpt 1 -log_dir StandardNoBNModel_Seed3/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -data_dir data/CUB_processed/class_attr_data_10 -b 32 -weight_decay 0.0004 -lr 0.01 -scheduler_step 20