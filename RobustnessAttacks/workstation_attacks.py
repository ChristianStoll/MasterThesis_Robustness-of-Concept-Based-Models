import os
import torch
import sys
import random
import json
import attacks as atk
import tests
from rtpt import RTPT
import argparse
import utils
from config import BASE_PATH

torch.set_num_threads(4)
n_attributes = 112
n_classes = 200
random.seed(123)



def attack_images_XtoCtoY(args):
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    model_names = ['inception-v3_200', 'cb_sequential', 'cb_independent']
    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    seeds = [1, 2, 3]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names)*len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    accs = atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    with open(f'{BASE_PATH}/outputs/results_attack_XtoCtoY_varBIM_fixPGD.json', 'w') as f:
        json.dump(data, f)


def attack_concepts_CtoY(args):
    model_names = ['cb_sequential', 'cb_independent']
    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    seeds = [1, 2, 3]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names)*len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    for use_sigmoid in [True, False]:
        # model_accuracies (no iwp/rfiwp included), initially_wrong_predictions, right_from_initially_wrong_predictions
        accs, iwp, rfiwp = atk.attack_concepts(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks,
                                               use_sigmoid=use_sigmoid, num_examples=None,
                                               batch_size=args.batch_size, rtpt=rtpt)

        data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs,
                'initially_wrong_predictions': iwp, 'right_from_initially_wrong_predictions': rfiwp}

        with open(f'{BASE_PATH}/outputs/results_attack_CtoY_sigmoid_' + str(use_sigmoid) + '.json', 'w') as f:
            json.dump(data, f)


def attack_img_XtoC_cb(args):
    attacks = ['fgsm', 'bim', 'pgd']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    seeds = [1, 2, 3]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(attacks)*len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    accs, confusion_matrix, num_c_wrong, total_num_concepts = \
        atk.attack_image_XtoC(seeds=seeds, eps=eps, attacks=attacks, num_examples=None,  batch_size=args.batch_size,
                              rtpt=rtpt)

    data = {'models': ['cb_concept'], 'total_num_concepts': total_num_concepts, 'seeds': seeds, 'attacks': attacks,
            'epsilon': eps, 'accuracies': accs, 'num_c_wrong': num_c_wrong, 'confusion_matrix': confusion_matrix}

    with open(f'{BASE_PATH}/outputs/results_attack_XtoC.json', 'w') as f:
        json.dump(data, f)


def test_scouter_chkpts(args):
    #model_names = ['scouter100+', 'scouter100-', 'scouter25+_s1_lv', 'scouter25-_s1_lv', 'scouter25+_s1_lc', 'scouter25-_s1_lc']
    model_names = ['scouter100+', 'scouter100-']

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names)*3)
    # Start the RTPT tracking
    rtpt.start()

    tests.test_scouter(model_names=model_names, batch_size=args.batch_size, rtpt=rtpt)


def attack_scouter(args):
    model_names = ['scouter25+', 'scouter25-']
    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    seeds = [1, 2, 3]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names)*len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    accs = atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    with open(f'{BASE_PATH}/outputs/results_scouter_atk.json', 'w') as f:
        json.dump(data, f)


def attack_scouter_variations(args):
    # 'scouter25+_s1_lv' is the same as 'scouter25+_s1_lc' -> no need to attack it
    model_names = ['scouter25-_s1_lv', 'scouter25+_s1_lc', 'scouter25-_s1_lc']
    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    seeds = [1, 2, 3]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names)*len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    accs = atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    with open(f'{BASE_PATH}/outputs/results_scouter_variations_atk.json', 'w') as f:
        json.dump(data, f)


def attack_e2e_cub_black(args):
    model_names = ['cub_black_inception-v3_200']
    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    seeds = [1, 2, 3]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names)*len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    accs = atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    with open(f'{BASE_PATH}/outputs/results_attack_e2e_cub_black.json', 'w') as f:
        json.dump(data, f)


def attack_e2e_cub_black_segmentations(args):
    model_names = ['cub_black_inception-v3_200']
    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    seeds = [1, 2, 3]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names)*len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    accs = atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt, use_segmentation=True)

    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    with open(f'{BASE_PATH}/outputs/results_segmentation_attack_e2e_cub_black.json', 'w') as f:
        json.dump(data, f)


def attack_e2e_cub_black_segmentations_sparsefool_step_variations(args):
    model_names = ['cub_black_inception-v3_200']
    attacks = ['sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    seeds = [1, 2, 3]
    steps = [2, 3, 4, 5]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names)*len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    accs = []
    for step in steps:
        acc = atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                               batch_size=args.batch_size, rtpt=rtpt, use_segmentation=True)
        accs.append(acc)
    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs,
            'steps': steps}

    with open(f'{BASE_PATH}/outputs/results_e2e_segmentation_atk_sparsefool_step_variations.json', 'w') as f:
        json.dump(data, f)


def find_reason(args):
    """
    attack XtoY for Concept Bottleneck models to store more data than in normal attack setups
    :param args:
    :return:
    """
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    model_names = ['cb_sequential', 'cb_independent']
    attacks = ['fgsm']
    seeds = [1, 2, 3]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names)*len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    use_attr_to_check_concepts = True
    if use_attr_to_check_concepts:
        accs, dataset_stats, num_wrong, conf_matrix, num_imgs = \
            atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                             batch_size=args.batch_size, rtpt=rtpt, save_perturbations=False,
                             use_attr_to_check_concepts=use_attr_to_check_concepts)

        data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs,
                'dataset_stats': dataset_stats, 'num_concepts_wrong': num_wrong, 'num_imgs': num_imgs,
                'confusion_matrix': conf_matrix}
    else:
        accs, dataset_stats = \
            atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                             batch_size=args.batch_size, rtpt=rtpt, save_perturbations=False,
                             use_attr_to_check_concepts=use_attr_to_check_concepts)

        data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs,
                'dataset_stats': dataset_stats}

    with open(f'{BASE_PATH}/outputs/find_reason_all_BN_models.json', 'w') as f:
        json.dump(data, f)


def attack_ViT(args):
    os.makedirs(f'{BASE_PATH}/outputs/', exist_ok=True)
    model_names = ['vit_base_p16_224', 'vit_small_p32_224', 'vit_small_p16_224']
    seeds = [1, 2, 3]
    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names) * len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    accs = atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    os.makedirs(f'{BASE_PATH}/outputs/', exist_ok=True)
    with open(f'{BASE_PATH}/outputs/results_vit_XtoY.json', 'w') as f:
        json.dump(data, f)


def attack_inception25(args):
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    model_names = ['inception-v3_25']
    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    seeds = [1, 2, 3]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names)*len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    accs = atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    with open(f'{BASE_PATH}/outputs/results_attack_Inception25.json', 'w') as f:
        json.dump(data, f)


def test_Hybrid(args):
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    model_names = ['HybridIndependent']
    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    seeds = [1]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names) * len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    accs = atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    with open(f'{BASE_PATH}/outputs/results_attack_HybridIndependent.json', 'w') as f:
        json.dump(data, f)


def travBirds_tests(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=5*3)
    # Start the RTPT tracking
    rtpt.start()

    # baseline test for 1 seed as sanity check
    model_names = ['inception-v3_200']
    accs = atk.travellingBirds_test(model_names=model_names, seeds=[1], batch_size=args.batch_size,
                             image_dirs=['CUB_black', 'cub', 'CUB_fixed/test', 'CUB_random'])

    seeds = [1, 2, 3]
    model_names = ['inception-v3_200', 'cb_sequential', 'cb_independent']

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=len(model_names) * len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    accs = atk.travellingBirds_test(model_names=model_names, seeds=[1, 2, 3], batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': ['cb_standard', 'cb_sequential', 'cb_independent'], 'seeds': seeds,
            'datasets': ['CUB', 'CUB_black', 'CUB_fixed', 'CUB_random'], 'accuracies': accs}

    with open(f'{BASE_PATH}/outputs/results_birds_cb.json', 'w') as f:
        json.dump(data, f)

    seeds = [1, 2, 3]
    # inception trained on cub black
    model_names = ['cub_black_inception-v3_200']
    accs = atk.travellingBirds_test(model_names=model_names, seeds=seeds, batch_size=args.batch_size,
                                    image_dirs=['CUB_black', 'cub', 'CUB_fixed/test', 'CUB_random'])

    data = {'models': model_names, 'seeds': seeds, 'datasets': ['CUB_black', 'CUB', 'CUB_fixed', 'CUB_random'],
            'accuracies': accs}
    with open(f'{BASE_PATH}/outputs/results_birds_cub_black_inception.json', 'w') as f:
        json.dump(data, f)

    # inception 25 on cub
    model_names = ['inception-v3_25']
    accs = atk.travellingBirds_test(model_names=model_names, seeds=seeds, batch_size=args.batch_size)

    data = {'models': model_names, 'seeds': seeds, 'datasets': ['CUB', 'CUB_black', 'CUB_fixed', 'CUB_random'],
            'accuracies': accs}
    with open(f'{BASE_PATH}/outputs/results_birds_inception_25.json', 'w') as f:
        json.dump(data, f)

    model_names = ['scouter25+', 'scouter25-']
    accs = atk.travellingBirds_test(model_names=model_names, seeds=seeds, batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': model_names, 'seeds': seeds, 'datasets': ['CUB', 'CUB_black', 'CUB_fixed', 'CUB_random'],
            'accuracies': accs}

    with open(f'{BASE_PATH}/outputs/results_scouter_birds.json', 'w') as f:
        json.dump(data, f)

    # variations of hyperparams of scouter
    model_names = ['scouter25-_s1_lv', 'scouter25+_s1_lc', 'scouter25-_s1_lc']
    accs = atk.travellingBirds_test(model_names=model_names, seeds=seeds, batch_size=args.batch_size)

    data = {'models': model_names, 'seeds': seeds, 'datasets': ['CUB', 'CUB_black', 'CUB_fixed', 'CUB_random'],
            'accuracies': accs}
    with open('outputs/results_birds_scouter_variations.json', 'w') as f:
        json.dump(data, f)

    # ViT small p16 224
    model_names = ['vits_dropout_300ep_v2_drop', 'vits_dropout_300ep_v2_nodrop']
    accs = atk.travellingBirds_test(model_names=model_names, seeds=seeds, batch_size=args.batch_size)

    data = {'models': model_names, 'seeds': seeds, 'datasets': ['CUB', 'CUB_black', 'CUB_fixed', 'CUB_random'],
            'accuracies': accs}
    with open('outputs/results_birds_ViT.json', 'w') as f:
        json.dump(data, f)

    model_names = ['vit_base_p16_224', 'vit_small_p32_224', 'vit_small_p16_224']
    seeds = [1, 2, 3]

    accs = atk.travellingBirds_test(model_names=model_names, seeds=seeds, batch_size=args.batch_size)

    data = {'models': model_names, 'seeds': seeds, 'datasets': ['CUB', 'CUB_black', 'CUB_fixed', 'CUB_random'],
            'accuracies': accs}
    with open(f'{BASE_PATH}/outputs/results_birds_ViT_new.json', 'w') as f:
        json.dump(data, f)


def setup_cb_models():
    """
        this function is required to setup the model dicts of the models of "Concept Bottleneck Models,
        https://arxiv.org/abs/2007.04612
        to be able to store & use the state dict of the models
    :return:
    """
    model_setup_names = ['standard', 'independent', 'sequential', 'concept']
    seeds = [1, 2, 3]
    utils.setup_model_dicts(model_names=model_setup_names, seeds=seeds)


def attack_mask_stage2_XtoY_noseg(args):
    seeds = [1, 2, 3]
    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=12)
    # Start the RTPT tracking
    rtpt.start()

    model_names = ['Mask_Independent_cropbb', 'Mask_Independent_segbb', 'Mask_Sequential_cropbb', 'Mask_Sequential_segbb']
    accs = atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    with open(f'{BASE_PATH}/outputs/results_mask_stage2_XtoY.json', 'w') as f:
        json.dump(data, f)


def attack_mask_stage2_XtoY_useseg(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=3)
    # Start the RTPT tracking
    rtpt.start()
    seeds = [1, 2, 3]
    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]

    # second part using segmented images (compare with Inception black)
    model_names_black = ['Mask_Independent_cropbb_useseg', 'Mask_Independent_segbb_useseg', 'Mask_Sequential_cropbb_useseg',
                         'Mask_Sequential_segbb_useseg']

    accs = atk.attack_image(model_names=model_names_black, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt, use_segmentation=True)

    data = {'models': model_names_black, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}
    with open(f'{BASE_PATH}/outputs/results_mask_stage2_XtoY_useseg.json', 'w') as f:
        json.dump(data, f)


def attack_mask_stage2_XtoY_useseg_fgsm(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=3)
    # Start the RTPT tracking
    rtpt.start()
    seeds = [1, 2, 3]
    attacks = ['fgsm']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]

    # second part using segmented images (compare with Inception black)
    model_names_black = ['Mask_Independent_cropbb_useseg', 'Mask_Independent_segbb_useseg', 'Mask_Sequential_cropbb_useseg',
                         'Mask_Sequential_segbb_useseg']

    accs = atk.attack_image(model_names=model_names_black, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt, use_segmentation=True)

    data = {'models': model_names_black, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}
    with open(f'{BASE_PATH}/outputs/results_mask_stage2_XtoY_useseg_fgsm.json', 'w') as f:
        json.dump(data, f)

def attack_mask_stage2_XtoY_useseg_bim(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=3)
    # Start the RTPT tracking
    rtpt.start()
    seeds = [1, 2, 3]
    attacks = ['bim']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]

    # second part using segmented images (compare with Inception black)
    model_names_black = ['Mask_Independent_cropbb_useseg', 'Mask_Independent_segbb_useseg', 'Mask_Sequential_cropbb_useseg',
                         'Mask_Sequential_segbb_useseg']

    accs = atk.attack_image(model_names=model_names_black, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt, use_segmentation=True)

    data = {'models': model_names_black, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}
    with open(f'{BASE_PATH}/outputs/results_mask_stage2_XtoY_useseg_bim.json', 'w') as f:
        json.dump(data, f)

def attack_mask_stage2_XtoY_useseg_pgd(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=3)
    # Start the RTPT tracking
    rtpt.start()
    seeds = [1, 2, 3]
    attacks = ['pgd']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]

    # second part using segmented images (compare with Inception black)
    model_names_black = ['Mask_Independent_cropbb_useseg', 'Mask_Independent_segbb_useseg', 'Mask_Sequential_cropbb_useseg',
                         'Mask_Sequential_segbb_useseg']

    accs = atk.attack_image(model_names=model_names_black, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt, use_segmentation=True)

    data = {'models': model_names_black, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}
    with open(f'{BASE_PATH}/outputs/results_mask_stage2_XtoY_useseg_pgd.json', 'w') as f:
        json.dump(data, f)

def attack_mask_stage2_XtoY_useseg_sparsefool(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=3)
    # Start the RTPT tracking
    rtpt.start()
    seeds = [1, 2, 3]
    attacks = ['sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]

    # second part using segmented images (compare with Inception black)
    model_names_black = ['Mask_Independent_cropbb_useseg', 'Mask_Independent_segbb_useseg', 'Mask_Sequential_cropbb_useseg',
                         'Mask_Sequential_segbb_useseg']

    accs = atk.attack_image(model_names=model_names_black, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt, use_segmentation=True)

    data = {'models': model_names_black, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}
    with open(f'{BASE_PATH}/outputs/results_mask_stage2_XtoY_useseg_sparsefool.json', 'w') as f:
        json.dump(data, f)


def attack_mask_stage2_XtoC(args):
    seeds = [1, 2, 3]
    attacks = ['fgsm', 'bim', 'pgd']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    print(f'Check the path: {BASE_PATH}/outputs/results_mask_stage2_XtoC.json')

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=12)
    # Start the RTPT tracking
    rtpt.start()

    model_names = ['Mask_Concept_cropbb', 'Mask_Concept_segbb']
    accs, confusion_matrix, num_c_wrong, total_num_concepts = \
        atk.attack_image_XtoC(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                              batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': model_names, 'total_num_concepts': total_num_concepts, 'seeds': seeds, 'attacks': attacks,
            'epsilon': eps, 'accuracies': accs, 'num_c_wrong': num_c_wrong, 'confusion_matrix': confusion_matrix}

    with open(f'{BASE_PATH}/outputs/results_mask_stage2_XtoC.json', 'w') as f:
        json.dump(data, f)

    # second part using segmented images (compare with Inception black)
    model_names_black = ['Mask_Concept_cropbb_useseg', 'Mask_Concept_segbb_useseg']
    accs, confusion_matrix, num_c_wrong, total_num_concepts = \
        atk.attack_image_XtoC(model_names=model_names_black, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                              batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': model_names_black, 'total_num_concepts': total_num_concepts, 'seeds': seeds, 'attacks': attacks,
            'epsilon': eps, 'accuracies': accs, 'num_c_wrong': num_c_wrong, 'confusion_matrix': confusion_matrix}

    with open(f'{BASE_PATH}/outputs/results_mask_stage2_XtoC_useseg.json', 'w') as f:
        json.dump(data, f)


def attack_mask_stage2_CtoY(args):
    model_names = ['Mask_Independent_cropbb', 'Mask_Independent_segbb', 'Mask_Sequential_cropbb',
                   'Mask_Sequential_segbb', 'Mask_Independent_cropbb_useseg', 'Mask_Independent_segbb_useseg',
                   'Mask_Sequential_cropbb_useseg', 'Mask_Sequential_segbb_useseg']

    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    seeds = [1, 2, 3]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=2*len(model_names)*len(seeds))
    # Start the RTPT tracking
    rtpt.start()

    for use_sigmoid in [True, False]:
        # model_accuracies (no iwp/rfiwp included), initially_wrong_predictions, right_from_initially_wrong_predictions
        accs, iwp, rfiwp = atk.attack_concepts(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks,
                                               use_sigmoid=use_sigmoid, num_examples=None, batch_size=args.batch_size,
                                               rtpt=rtpt)

        data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs,
                'initially_wrong_predictions': iwp, 'right_from_initially_wrong_predictions': rfiwp}

        with open(f'{BASE_PATH}/outputs/results_attack_mask_CtoY_sigmoid_' + str(use_sigmoid) + '.json', 'w') as f:
            json.dump(data, f)


def mask_s2_XtoY_sparsefool_noseg(args):
    seeds = [1, 2, 3]
    attacks = ['sparsefool', 'sparsefool', 'sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=9)
    # Start the RTPT tracking
    rtpt.start()

    model_names = ['Mask_Independent_cropbb', 'Mask_Independent_segbb', 'Mask_Sequential_cropbb', 'Mask_Sequential_segbb']
    accs = atk.attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt)

    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    with open(f'{BASE_PATH}/outputs/results_mask_stage2_XtoY_sparsefool_noseg2.json', 'w') as f:
        json.dump(data, f)


def mask_s2_XtoY_sparsefool_seg(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=args.task, max_iterations=9)
    # Start the RTPT tracking
    rtpt.start()
    seeds = [1, 2, 3]
    attacks = ['sparsefool', 'sparsefool', 'sparsefool']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]

    # second part using segmented images (compare with Inception black)
    model_names_black = ['Mask_Independent_cropbb_useseg', 'Mask_Independent_segbb_useseg',
                         'Mask_Sequential_cropbb_useseg',
                         'Mask_Sequential_segbb_useseg']

    accs = atk.attack_image(model_names=model_names_black, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                            batch_size=args.batch_size, rtpt=rtpt, use_segmentation=True)

    data = {'models': model_names_black, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    with open(f'{BASE_PATH}/outputs/results_mask_stage2_XtoY_sparsefool_seg2.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    pkl_file_path = 'CUB_200_2011/CUB_processed/class_attr_data_10/'

    parser = argparse.ArgumentParser('Workstation Attacks', add_help=False)
    # self defined
    parser.add_argument('--task', default='', type=str)
    parser.add_argument('--batch_size', default=64, type=int)

    args = parser.parse_args()

    """model_setup_names = ['cb_standard', 'cb_independent', 'cb_sequential', 'concept']
    seeds = [1, 2, 3]

    setup_model_dicts(model_names=model_setup_names, seeds=seeds)
    exit()"""

    tasks = sys.argv[2]
    tasks = tasks.split(',')
    print('Tasks: ', tasks)
    for task in tasks:
        print(task)
        if task == 'attack_concepts_CtoY':
            attack_concepts_CtoY(args)
        elif task == 'attack_img_XtoC_cb':
            attack_img_XtoC_cb(args)
        elif task == 'attack_image_XtoCtoY':
            attack_images_XtoCtoY(args)
        elif task == 'test_scouter_chkpts':
            test_scouter_chkpts(args)
        elif task == 'attack_scouter':
            attack_scouter(args)
        elif task == 'attack_e2e_cub_black':
            attack_e2e_cub_black(args)
        elif task == 'attack_scouter_variations':
            attack_scouter_variations(args)
        elif task == 'attack_e2e_cub_black_segmentations':
            attack_e2e_cub_black_segmentations(args)
        elif task == 'attack_e2e_cub_black_segmentations_sparsefool_step_variations':
            attack_e2e_cub_black_segmentations_sparsefool_step_variations(args)
        elif task == 'find_reason':
            find_reason(args)
        elif task == 'attack_vit':
            attack_ViT(args)
        elif task == 'attack_inception25':
            attack_inception25(args)
        elif task == 'travBirds_tests':
            travBirds_tests(args)
        elif task == 'test_Hybrid':
            test_Hybrid(args)
        elif task == 'setup_cb_models':
            setup_cb_models()
        elif task == 'attack_mask_stage2_XtoC':
            attack_mask_stage2_XtoC(args)
        elif task == 'attack_mask_stage2_XtoY_noseg':
            attack_mask_stage2_XtoY_noseg(args)
        elif task == 'attack_mask_stage2_XtoY_useseg':
            attack_mask_stage2_XtoY_useseg(args)
        elif task == 'attack_mask_stage2_CtoY':
            attack_mask_stage2_CtoY(args)
        elif task == 'mask_s2_XtoY_sparsefool_noseg':
            mask_s2_XtoY_sparsefool_noseg(args)
        elif task == 'mask_s2_XtoY_sparsefool_seg':
            mask_s2_XtoY_sparsefool_seg(args)
        elif task == 'attack_mask_stage2_XtoY_useseg_fgsm':
            attack_mask_stage2_XtoY_useseg_fgsm(args)
        elif task == 'attack_mask_stage2_XtoY_useseg_bim':
            attack_mask_stage2_XtoY_useseg_bim(args)
        elif task == 'attack_mask_stage2_XtoY_useseg_pgd':
            attack_mask_stage2_XtoY_useseg_pgd(args)
        elif task == 'attack_mask_stage2_XtoY_useseg_sparsefool':
            attack_mask_stage2_XtoY_useseg_sparsefool(args)