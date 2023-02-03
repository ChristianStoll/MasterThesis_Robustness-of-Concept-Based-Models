

# ======================= Source code, datasets and preprocessing =======================
# Download the source code and datasets from our codalab worksheet ( https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2 ).
# You should have `CUB_200_2011`, `CUB_processed`, `places365`, `pretrained`, `src` all availble on the path during experiment runs.
# Each python script outputs to a folder, change `-log_dir` or `-out_dir` if you would like different output folders.

# Experiments

### Standard Model Linear Probe
python3 CUB/generate_new_data.py ExtractProbeRepresentations --model_path Joint0Model_Seed1/outputs/best_model_1.pth --layer_idx -1 --data_dir CUB_processed/class_attr_data_10 --out_dir Joint0Model_Seed1_ExtractProbeRep
python3 CUB/generate_new_data.py ExtractProbeRepresentations --model_path Joint0Model_Seed2/outputs/best_model_2.pth --layer_idx -1 --data_dir CUB_processed/class_attr_data_10 --out_dir Joint0Model_Seed2_ExtractProbeRep
python3 CUB/generate_new_data.py ExtractProbeRepresentations --model_path Joint0Model_Seed3/outputs/best_model_3.pth --layer_idx -1 --data_dir CUB_processed/class_attr_data_10 --out_dir Joint0Model_Seed3_ExtractProbeRep
python3 CUB/probe.py -data_dir Joint0Model_Seed1_ExtractProbeRep -n_attributes 112 -log_dir Joint0Model_Seed1_LinearProbe/outputs -lr 0.001 -scheduler_step 1000 -weight_decay 0.00004
python3 CUB/probe.py -data_dir Joint0Model_Seed2_ExtractProbeRep -n_attributes 112 -log_dir Joint0Model_Seed2_LinearProbe/outputs -lr 0.001 -scheduler_step 1000 -weight_decay 0.00004
python3 CUB/probe.py -data_dir Joint0Model_Seed3_ExtractProbeRep -n_attributes 112 -log_dir Joint0Model_Seed3_LinearProbe/outputs -lr 0.001 -scheduler_step 1000 -weight_decay 0.00004
python3 CUB/probe.py -data_dirs Joint0Model_Seed1_ExtractProbeRep Joint0Model_Seed2_ExtractProbeRep Joint0Model_Seed3_ExtractProbeRep -log_dir Joint0Model_LinearProbe -eval -model_dirs Joint0Model_Seed1_LinearProbe/outputs/best_model.pth Joint0Model_Seed2_LinearProbe/outputs/best_model.pth Joint0Model_Seed3_LinearProbe/outputs/best_model.pth

### Standard No Bottleneck Model
python3 experiments.py cub Standard --seed 1 -ckpt 1 -log_dir StandardNoBNModel_Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -data_dir CUB_processed/class_attr_data_10 -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 20
python3 experiments.py cub Standard --seed 2 -ckpt 1 -log_dir StandardNoBNModel_Seed2/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -data_dir CUB_processed/class_attr_data_10 -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 20
python3 experiments.py cub Standard --seed 3 -ckpt 1 -log_dir StandardNoBNModel_Seed3/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -data_dir CUB_processed/class_attr_data_10 -b 64 -weight_decay 0.0004 -lr 0.01 -scheduler_step 20
python3 CUB/inference.py -model_dirs StandardNoBNModel_Seed1/outputs/best_model_1.pth StandardNoBNModel_Seed2/outputs/best_model_2.pth StandardNoBNModel_Seed3/outputs/best_model_3.pth -eval_data test -data_dir CUB_processed/class_attr_data_10 -log_dir StandardNoBNModel/outputs

### Joint Model
#### Concept loss weight = 0.001
python3 experiments.py cub Joint --seed 1 -ckpt 1 -log_dir Joint0.001Model_Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.001 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -end2end
python3 experiments.py cub Joint --seed 2 -ckpt 1 -log_dir Joint0.001Model_Seed2/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.001 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -end2end
python3 experiments.py cub Joint --seed 3 -ckpt 1 -log_dir Joint0.001Model_Seed3/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.001 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -end2end
python3 CUB/inference.py -model_dirs Joint0.001Model_Seed1/outputs/best_model_1.pth Joint0.001Model_Seed2/outputs/best_model_2.pth Joint0.001Model_Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir CUB_processed/class_attr_data_10 -log_dir Joint0.001Model/outputs
#### Concept loss weight = 0.01
python3 experiments.py cub Joint --seed 1 -ckpt 1 -log_dir Joint0.01Model__Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end
python3 experiments.py cub Joint --seed 2 -ckpt 1 -log_dir Joint0.01Model__Seed2/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end
python3 experiments.py cub Joint --seed 3 -ckpt 1 -log_dir Joint0.01Model_Seed3/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -end2end
python3 CUB/inference.py -model_dirs Joint0.01Model__Seed1/outputs/best_model_1.pth Joint0.01Model__Seed2/outputs/best_model_2.pth -eval_data test -use_attr -n_attributes 112 -data_dir CUB_processed/class_attr_data_10 -feature_group_results -log_dir Joint0.01Model/outputs
#### Concept loss weight = 0.01, with Sigmoid
python3 experiments.py cub Joint --seed 1 -ckpt 1 -log_dir Joint0.01SigmoidModel__Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end -use_sigmoid
python3 experiments.py cub Joint --seed 2 -ckpt 1 -log_dir Joint0.01SigmoidModel__Seed2/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end -use_sigmoid
python3 experiments.py cub Joint --seed 3 -ckpt 1 -log_dir Joint0.01SigmoidModel__Seed3/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.0004 -lr 0.001 -scheduler_step 1000 -end2end -use_sigmoid
#### Concept loss weight = 0.1
python3 experiments.py cub Joint --seed 1 -ckpt 1 -log_dir Joint0.1Model__Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.1 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -end2end
python3 experiments.py cub Joint --seed 2 -ckpt 1 -log_dir Joint0.1Model__Seed2/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.1 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -end2end
python3 experiments.py cub Joint --seed 3 -ckpt 1 -log_dir Joint0.1Model__Seed3/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.1 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -end2end
python3 CUB/inference.py -model_dirs Joint0.1Model__Seed1/outputs/best_model_1.pth Joint0.1Model__Seed2/outputs/best_model_2.pth Joint0.1Model__Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir CUB_processed/class_attr_data_10 -log_dir Joint0.1Model/outputs
#### Concept loss weight = 1
python3 experiments.py cub Joint --seed 1 -ckpt 1 -log_dir Joint1Model__Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 1 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -end2end
python3 experiments.py cub Joint --seed 2 -ckpt 1 -log_dir Joint1Model__Seed2/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 1 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -end2end
python3 experiments.py cub Joint --seed 3 -ckpt 1 -log_dir Joint1Model__Seed3/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 1 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -end2end
python3 CUB/inference.py -model_dirs Joint1Model__Seed1/outputs/best_model_1.pth Joint1Model__Seed2/outputs/best_model_2.pth Joint1Model__Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir CUB_processed/class_attr_data_10 -log_dir Joint1Model/outputs

### Multitask Model
python3 experiments.py cub Multitask --seed 1 -ckpt 1 -log_dir MultitaskModel_Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20
python3 experiments.py cub Multitask --seed 2 -ckpt 1 -log_dir MultitaskModel_Seed2/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20
python3 experiments.py cub Multitask --seed 3 -ckpt 1 -log_dir MultitaskModel_Seed3/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -attr_loss_weight 0.01 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20
python3 CUB/inference.py -model_dirs MultitaskModel_Seed1/outputs/best_model_1.pth MultitaskModel_Seed2/outputs/best_model_2.pth MultitaskModel_Seed3/outputs/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir CUB_processed/class_attr_data_10 -log_dir MultitaskModel/outputs