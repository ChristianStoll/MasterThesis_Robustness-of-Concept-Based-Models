CUDA_VISIBLE_DEVICES=1 python3 workstation_attacks.py --task attack_scouter_variations --batch_size 32
CUDA_VISIBLE_DEVICES=3 python3 workstation_attacks.py --task attack_img_XtoC --batch_size 64

CUDA_VISIBLE_DEVICES=1 python3 workstation_attacks.py --task attack_e2e_cub_black_segmentations --batch_size 32
CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task attack_e2e_cub_black_segmentations_sparsefool_step_variations --batch_size 64

CUDA_VISIBLE_DEVICES=1 python3 workstation_attacks.py --task test_vit --batch_size 32
CUDA_VISIBLE_DEVICES=1 python3 workstation_attacks.py --task find_reason --batch_size 64

CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task test_vit_img_size --batch_size 64
CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task attack_vit --batch_size 32
CUDA_VISIBLE_DEVICES=1 python3 workstation_attacks.py --task attack_inception25 --batch_size 64
CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task attack_Hybrid_XtoY --batch_size 64
CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task test_Hybrid --batch_size 64

CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task travBirds_tests --batch_size 64


CUDA_VISIBLE_DEVICES=1 python3 workstation_attacks.py --task attack_img_XtoC_hybrid --batch_size 128
CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task attack_Hybrid_XtoY --batch_size 64



CUDA_VISIBLE_DEVICES=1 python3 MaskPipelineAttack.py --task travBirds_maskrcnn

CUDA_VISIBLE_DEVICES=1 python3 MaskPipelineAttack.py --task predict_adversarial_stage2_noseg --batch_size 64
CUDA_VISIBLE_DEVICES=0 python3 MaskPipelineAttack.py --task predict_adversarial_stage2_useseg --batch_size 64
CUDA_VISIBLE_DEVICES=1 python3 MaskPipelineAttack.py --task predict_birds_stage2_noseg --batch_size 64
CUDA_VISIBLE_DEVICES=1 python3 MaskPipelineAttack.py --task predict_birds_stage2_useseg --batch_size 64




CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task attack_vit --batch_size 16

CUDA_VISIBLE_DEVICES=1 python3 workstation_attacks.py --task attack_mask_stage2_XtoC --batch_size 64
CUDA_VISIBLE_DEVICES=3 python3 workstation_attacks.py --task attack_mask_stage2_XtoY_useseg --batch_size 64
CUDA_VISIBLE_DEVICES=3 python3 workstation_attacks.py --task attack_mask_stage2_XtoY_noseg --batch_size 64



CUDA_VISIBLE_DEVICES=3 python3 workstation_attacks.py --task attack_mask_stage2_CtoY --batch_size 8


CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task test_travBirds_ViT --batch_size 64
CUDA_VISIBLE_DEVICES=1 python3 workstation_attacks.py --task mask_s2_XtoY_sparsefool_noseg --batch_size 8
CUDA_VISIBLE_DEVICES=1 python3 workstation_attacks.py --task mask_s2_XtoY_sparsefool_seg --batch_size 8

CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task attack_mask_stage2_XtoY_useseg_sparsefool --batch_size 8
CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task attack_mask_stage2_XtoY_useseg_fgsm --batch_size 32
CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task attack_mask_stage2_XtoY_useseg_bim --batch_size 64
CUDA_VISIBLE_DEVICES=0 python3 workstation_attacks.py --task attack_mask_stage2_XtoY_useseg_pgd --batch_size 16