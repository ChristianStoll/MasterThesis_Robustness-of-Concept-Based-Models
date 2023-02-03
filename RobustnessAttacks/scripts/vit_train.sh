# These 3 models were the ones in the thesis!
# vit s p32 224 dropout
CUDA_VISIBLE_DEVICES=0 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch32_224 --pretrained True --num-classes 200 --batch_size 64 --seed 1 --epochs 300 --img_size 224 --output models/vit/vit_small_patch32_224/drop/seed1/ --drop 0.1
CUDA_VISIBLE_DEVICES=1 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch32_224 --pretrained True --num-classes 200 --batch_size 64 --seed 2 --epochs 300 --img_size 224 --output models/vit/vit_small_patch32_224/drop/seed2/ --drop 0.1
CUDA_VISIBLE_DEVICES=2 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch32_224 --pretrained True --num-classes 200 --batch_size 64 --seed 3 --epochs 300 --img_size 224 --output models/vit/vit_small_patch32_224/drop/seed3/ --drop 0.1

# vit base p16 224 dropout
CUDA_VISIBLE_DEVICES=1 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_base_patch16_224 --pretrained True --num-classes 200 --batch_size 32 --seed 1 --epochs 300 --img_size 224 --output models/vit/vit_base_patch16_224/drop/seed1/ --drop 0.1
CUDA_VISIBLE_DEVICES=2 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_base_patch16_224 --pretrained True --num-classes 200 --batch_size 32 --seed 2 --epochs 300 --img_size 224 --output models/vit/vit_base_patch16_224/drop/seed2/ --drop 0.1
CUDA_VISIBLE_DEVICES=3 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_base_patch16_224 --pretrained True --num-classes 200 --batch_size 32 --seed 3 --epochs 300 --img_size 224 --output models/vit/vit_base_patch16_224/drop/seed3/ --drop 0.1

# vit base p16 224 w/o dropout
CUDA_VISIBLE_DEVICES=1 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_base_patch16_224 --pretrained True --num-classes 200 --batch_size 32 --seed 1 --epochs 300 --img_size 224 --output models/vit/vit_base_patch16_224/nodrop/seed1/
CUDA_VISIBLE_DEVICES=2 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_base_patch16_224 --pretrained True --num-classes 200 --batch_size 32 --seed 2 --epochs 300 --img_size 224 --output models/vit/vit_base_patch16_224/nodrop/seed2/
CUDA_VISIBLE_DEVICES=3 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_base_patch16_224 --pretrained True --num-classes 200 --batch_size 32 --seed 3 --epochs 300 --img_size 224 --output models/vit/vit_base_patch16_224/nodrop/seed3/



# several attempts to train ViT models

CUDA_VISIBLE_DEVICES=0 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_base_patch16_224 --pretrained True --num-classes 200 --batch_size 32 --seed 1 --output models/vit/vitb_p16_224/seed1/


CUDA_VISIBLE_DEVICES=3 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch16_224 --pretrained True --num-classes 200 --batch_size 64 --seed 42 --img_size 224 --output models/vit/vits_p16_224/seed2/
CUDA_VISIBLE_DEVICES=0 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_large_patch16_224 --pretrained True --num-classes 200 --batch_size 16 --seed 1 --img_size 224 --output models/vit/vitl_p16_224/seed1/

CUDA_VISIBLE_DEVICES=3 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch16_224 --pretrained True --num-classes 200 --batch_size 64 --seed 45 --epochs 10 --img_size 224 --output models/vit/vits_p16_224/seed1_test/
CUDA_VISIBLE_DEVICES=1 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch16_384 --pretrained True --num-classes 200 --batch_size 16 --seed 45 --epochs 10 --img_size 384 --output models/vit/vit_small_patch16_384/seed1_test/

CUDA_VISIBLE_DEVICES=2 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch32_224 --pretrained True --num-classes 200 --batch_size 32 --seed 45 --epochs 10 --img_size 224 --output models/vit/vit_small_patch32_224/seed1_test/
CUDA_VISIBLE_DEVICES=1 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch16_224 --pretrained True --num-classes 200 --batch_size 32 --seed 45 --epochs 10 --img_size 224 --eval-metric loss --output models/vit/vit_small_patch32_384/seed1_test/

CUDA_VISIBLE_DEVICES=0 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch16_224 --pretrained True --num-classes 200 --batch_size 64 --seed 3 --epochs 300 --img_size 224 --eval-metric loss --drop 0.1 --output models/vit/vit_small_patch16_224/seed3_test/
CUDA_VISIBLE_DEVICES=1 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch16_224 --pretrained True --num-classes 200 --batch_size 64 --seed 25 --epochs 300 --img_size 224 --eval-metric loss --output models/vit/vit_small_patch16_224/seed3_test/

CUDA_VISIBLE_DEVICES=0 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch16_224 --pretrained True --num-classes 200 --batch_size 64 --seed 1 --epochs 300 --img_size 224 --drop 0.1 --output models/vit/vit_small_patch16_224/seed4_test/
CUDA_VISIBLE_DEVICES=2 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch16_224 --pretrained True --num-classes 200 --batch_size 64 --seed 1 --epochs 300 --img_size 224 --output models/vit/vit_small_patch16_224/seed4_test/

# vit s p32 224 no dropout
CUDA_VISIBLE_DEVICES=0 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch32_224 --pretrained True --num-classes 200 --batch_size 64 --seed 1 --epochs 300 --img_size 224 --output models/vit/vit_small_patch32_224/seed1/
CUDA_VISIBLE_DEVICES=1 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch32_224 --pretrained True --num-classes 200 --batch_size 64 --seed 2 --epochs 300 --img_size 224 --output models/vit/vit_small_patch32_224/seed2/
CUDA_VISIBLE_DEVICES=2 python3 ViT_train.py --data_dir data/CUB_200_2011/ --model vit_small_patch32_224 --pretrained True --num-classes 200 --batch_size 64 --seed 3 --epochs 300 --img_size 224 --output models/vit/vit_small_patch32_224/seed3/

