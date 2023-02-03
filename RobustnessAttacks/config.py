import os
# General
BASE_PATH = '/'.join(os.getcwd()
                     .replace("\\", '/')
                     .split('/')[:os.getcwd().replace("\\", '/').split('/').index('Robustness-of-Concept-Based-Models')+1])

N_ATTRIBUTES = 312
n_attributes = 112
N_CLASSES = 200

# Training
UPWEIGHT_RATIO = 9.0
MIN_LR = 0.0001
LR_DECAY_SIZE = 0.1

COCO_INSTANCE_CATEGORY_NAMES = [  #  =
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
BIRD_IDX = COCO_INSTANCE_CATEGORY_NAMES.index('bird')

# Directories
model_dirs = {
    'cb_standard': 'cub_models/StandardNoBNModel_Seed',
    'inception-v3_200': 'cub_models/StandardNoBNModel_Seed',
    'cb_concept': 'cub_models/ConceptModel_Seed',
    'cb_independent': 'cub_models/IndependentModel_WithVal_Seed',
    'cb_sequential': 'cub_models/SequentialModel_WithVal_Seed',
    'cub_black_cb_standard': 'cub_black_models/StandardNoBNModel_Seed',
    'cub_black_inception-v3_200': 'cub_black_models/StandardNoBNModel_Seed',
    'inception-v3_25': 'cub_models/StandardNoBNModel25_Seed',
    'scouter100+': 'scouter/100_classes_seed+/CUB200_use_slot_checkpoint.pth',
    'scouter100-': 'scouter/100_classes_seed+/CUB200_use_slot_negative_checkpoint.pth',
    'scouter25+': 'scouter/25_classes_seed+/CUB200_use_slot_checkpoint.pth',
    'scouter25-': 'scouter/25_classes_seed+/CUB200_use_slot_negative_checkpoint.pth',
    'scouter25+_s1_lv': 'scouter/25_classes_s1_lv_seed+/CUB200_use_slot_checkpoint.pth',
    'scouter25-_s1_lv': 'scouter/25_classes_s1_lv_seed+/CUB200_use_slot_negative_checkpoint.pth',
    'scouter25+_s1_lc': 'scouter/25_classes_s1_lc_seed+/CUB200_use_slot_checkpoint.pth',
    'scouter25-_s1_lc': 'scouter/25_classes_s1_lc_seed+/CUB200_use_slot_negative_checkpoint.pth',
    'vit_s2mall_p16_224': 'vit/vit_small_patch16_224/nodrop/seed',
    'vit_small_p32_224': 'vit/vit_small_patch32_224/nodrop/seed',
    'vit_base_p16_24': 'vit/vit_base_patch16_224/nodrop/seed',
    'MaskBottleneck_Concept': 'MaskBottleneck/Concept_+/Seed',
    'MaskBottleneck_Independent': 'MaskBottleneck/Independent/Seed'
}
