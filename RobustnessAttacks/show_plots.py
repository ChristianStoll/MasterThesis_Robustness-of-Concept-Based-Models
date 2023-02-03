import os
import yaml
import json
import numpy as np

import utils_plot as plot
import utils
from config import BASE_PATH


def load_XtoY_attack_results(filename, max_eps=1.):
    with open(f'{BASE_PATH}/outputs/{filename}.json'.replace("\\", '/'), 'r') as file:
        data = json.load(file)

    models = data['models']
    seeds = data['seeds']
    attacks = data['attacks']
    epsilon = data['epsilon']
    accuracies = data['accuracies']

    # reduce number of epsilon if required
    idx = len(epsilon)
    for i in range(len(epsilon)):
        if epsilon[i] >= max_eps:
            idx = i
            break

    epsilon = epsilon[:idx]
    new_accs = []
    for i in range(len(models)):
        new_accs.append([])
        for seed in range(len(seeds)):
            new_accs[i].append([])
            for j in range(len(attacks)):
                new_accs[i][seed].append(accuracies[i][seed][j][:idx])

    return models, seeds, attacks, epsilon, new_accs, data


def show_cb_XtoC_results():
    with open((BASE_PATH + '/outputs/' + 'results_attack_XtoC.json').replace("\\", '/')) as file:
        data = json.load(file)

    models = data['models']
    seeds = data['seeds']
    attacks = data['attacks']
    epsilon = data['epsilon']
    accuracies = data['accuracies']

    print(models)

    # model 1
    # seed 3
    # attacks 3

    plot.plot_eps_vs_accuracy(epsilons=epsilon, accuracies=[accuracies], attacks=attacks, modeltype_names=models)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=[accuracies], attacks=attacks,
                                       modeltype_names=models, normalize=True)

    confusion_matrix = data['confusion_matrix']

    print(np.array(confusion_matrix).shape)


def show_cb_XtoCtoY_results():
    models, seeds, attacks, epsilon, accuracies, data = \
        load_XtoY_attack_results('results_attack_XtoCtoY_varBIM_fixPGD', max_eps=1)

    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       normalize=True)


def show_cub_black_XtoCtoY_results():
    with open((BASE_PATH + '/outputs/' + 'results_attack_e2e_cub_black.json').replace("\\", '/'), 'r') as file:
        data = json.load(file)

    with open((BASE_PATH + '/outputs/' + 'results_attack_XtoCtoY_varBIM_fixPGD.json').replace("\\", '/'), 'r') as file:
        data2 = json.load(file)

    models = data['models'] + [data2['models'][0]]
    seeds = data['seeds']
    attacks = data['attacks']
    epsilon = data['epsilon']
    accuracies = data['accuracies'] + [data2['accuracies'][0]]

    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       normalize=True)


def show_cb_CtoY_results():
    models, seeds, attacks, epsilon, accuracies, data = \
        load_XtoY_attack_results('results_attack_CtoY_sigmoid_True', max_eps=1)

    # plot.plot_eps_vs_accuracy(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks,
                                       modeltype_names=models)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks,
                                       modeltype_names=models, normalize=True)


def show_cb_birds_results():
    with open(BASE_PATH + '/outputs/' + 'results_birds.json') as file:
        data = json.load(file)

    models = data['models']
    seeds = data['seeds']
    datasets = data['datasets']
    accuracies = data['accuracies']

    #plot.plot_bird_results(model_names=models, datasets=datasets, accuracies=accuracies)
    #plot.plot_bird_results(model_names=models, datasets=datasets, accuracies=accuracies, normalize=True)

    plot.plot_birds_acc_and_norm_results(model_names=models, datasets=datasets, accuracies=accuracies)

    #plot.plot_birds_result_difference(model_names=models, datasets=datasets, accuracies=accuracies, model_type='cb')


def show_vit_adversarial_results():
    fontsize = 'large'
    models_in, _, _, _, accuracies_in, data_in = load_XtoY_attack_results('results_attack_XtoCtoY_varBIM_fixPGD',
                                                                          max_eps=1)
    models, seeds, attacks, epsilon, accuracies, data = load_XtoY_attack_results('results_vit_XtoY', max_eps=1)

    models = [models_in[0]] + models
    accuracies = [accuracies_in[0]] + accuracies

    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       fontsize=fontsize)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       normalize=True, fontsize=fontsize)


def show_vit_birds_results():
    with open(f'{BASE_PATH}/outputs/results_cb_birds.json') as file:
        data_in = json.load(file)

    models_in = data_in['models']
    accuracies_in = data_in['accuracies']

    with open(f'{BASE_PATH}/outputs/results_birds_ViT_new.json') as file:
        data = json.load(file)

    models = data['models']
    datasets = data['datasets']
    accuracies = data['accuracies']

    models = [models_in[0]] + models
    accuracies = [accuracies_in[0]] + accuracies

    plot.plot_bird_results(model_names=models, datasets=datasets, accuracies=accuracies)
    plot.plot_bird_results(model_names=models, datasets=datasets, accuracies=accuracies, normalize=True)
    plot.plot_birds_result_difference(model_names=models, datasets=datasets, accuracies=accuracies)


def show_inception_black_birds_results():
    with open(f'{BASE_PATH}/outputs/results_birds_cub_black_inception.json') as file:
        data = json.load(file)

    models = data['models']
    datasets = data['datasets']
    accuracies = data['accuracies']

    plot.plot_bird_results(model_names=models, datasets=datasets, accuracies=accuracies)
    plot.plot_bird_results(model_names=models, datasets=datasets, accuracies=accuracies, normalize=True)
    plot.plot_birds_result_difference(model_names=models, datasets=datasets, accuracies=accuracies)


def show_scouter_results():
    max_eps = 1
    # load Inception_v3_25 results
    models_in, _, _, _, accuracies_in, data_in = load_XtoY_attack_results('results_attack_Inception25', max_eps=max_eps)

    # load scouter25 results
    models2, seeds, attacks, epsilon, accuracies2, data = load_XtoY_attack_results('results_scouter_atk', max_eps=max_eps)

    models = models_in + models2
    accuracies = accuracies_in + accuracies2

    # plot.plot_eps_vs_accuracy(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       normalize=True)

    # scouter variation results
    # load scouter25 variation results
    models_var, _, _, _, accuracies_var, data_var = load_XtoY_attack_results('results_scouter_variations_atk',
                                                                             max_eps=max_eps)

    models = models_in + models_var
    accuracies = accuracies_in + accuracies_var

    # plot.plot_eps_vs_accuracy(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       fontsize='large')
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       normalize=True, fontsize='large')

    models = models_in + models2 + models_var
    accuracies = accuracies_in + accuracies2 + accuracies_var

    # plot.plot_eps_vs_accuracy(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       fontsize='large')
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       normalize=True, fontsize='large')


def show_scouter_birds_results():
    with open(BASE_PATH + '/outputs/' + 'results_birds_scouter.json') as file:
        data = json.load(file)

    models = data['models']
    datasets = data['datasets']
    accuracies = data['accuracies']

    with open(f'{BASE_PATH}/outputs/results_birds_inception_25.json') as file:
        data_in = json.load(file)

    models = data_in['models'] + models
    accuracies = data_in['accuracies'] + accuracies

    plot.plot_bird_results(model_names=models, datasets=datasets, accuracies=accuracies)
    plot.plot_bird_results(model_names=models, datasets=datasets, accuracies=accuracies, normalize=True)
    plot.plot_birds_result_difference(model_names=models, datasets=datasets, accuracies=accuracies)

    with open(f'{BASE_PATH}/outputs/results_birds_scouter_variations.json') as file:
        data_var = json.load(file)

    models = data_in['models'] + data_var['models']
    accuracies = data_in['accuracies'] + data_var['accuracies']

    plot.plot_bird_results(model_names=models, datasets=datasets, accuracies=accuracies)
    plot.plot_bird_results(model_names=models, datasets=datasets, accuracies=accuracies, normalize=True)
    plot.plot_birds_result_difference(model_names=models, datasets=datasets, accuracies=accuracies)


def show_cb_cub_black_segmentation_results():
    models, seeds, attacks, epsilon, accuracies, data = \
        load_XtoY_attack_results('results_segmentation_attack_e2e_cub_black', max_eps=0.05)

    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       normalize=True)


def show_all_normalized_results():
    models = []
    attacks = []
    epsilon = []
    accuracies = []

    for file in ['results_segmentation_attack_e2e_cub_black', 'results_attack_e2e_cub_black', 'results_scouter_atk']:
        with open(BASE_PATH + '/outputs/' + file + '.json') as file:
            data = json.load(file)

            models += data['models']
            print(data['models'])
            accuracies += data['accuracies']

    file = 'results_attack_XtoCtoY_varBIM_fixPGD'
    with open(BASE_PATH + '/outputs/' + file + '.json') as file:
        data = json.load(file)

        models += data['models']
        accuracies += data['accuracies']
        attacks = data['attacks']
        epsilon = data['epsilon']

    with open(BASE_PATH + '/outputs/' + 'results_vit_nodropout_atk.json') as file:
        data = json.load(file)

        models += data['models']
        accuracies += data['accuracies']

    print(len(models))

    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       normalize=True)


def show_mask_stage2_attack_XtoY_results():
    fontsize = 'large'  # [medium, large]
    max_eps = 1
    # load Inception_v3_25 results
    models_in, _, _, _, accuracies_in, data_in = load_XtoY_attack_results('results_attack_XtoCtoY_varBIM_fixPGD',
                                                                          max_eps=max_eps)

    # load mask stage2 results - not using segmentations
    models, seeds, attacks, epsilon, accuracies, data = \
        load_XtoY_attack_results('results_mask_stage2_XtoY', max_eps=max_eps)

    models = [models_in[0]] + models
    accuracies = [accuracies_in[0]] + accuracies

    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       fontsize=fontsize)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       normalize=True, fontsize=fontsize)

    # load mask stage2 results - using segmentations
    models, seeds, attacks, epsilon, accuracies, data = \
        load_XtoY_attack_results('results_mask_stage2_XtoY_useseg', max_eps=max_eps)

    models = [models_in[0]] + models
    accuracies = [accuracies_in[0]] + accuracies

    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models,
                                       normalize=True)


def show_mask_stage2_XtoC_results():
    with open(f"{BASE_PATH}/outputs/results_mask_stage2_XtoC_useseg.json".replace("\\", '/')) as file:
        data = json.load(file)

    models = data['models']
    attacks = data['attacks']
    epsilon = data['epsilon']
    accuracies = data['accuracies']

    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks,
                                       modeltype_names=models, normalize=False)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks,
                                       modeltype_names=models, normalize=True)

    for i in range(len(models)):
        plot.plot_eps_vs_accuracy(epsilons=epsilon, accuracies=[accuracies[i]], attacks=attacks,
                                  modeltype_names=[models[i]])


def show_mask_stage2_CtoY_results():
    fontsize = 'medium'
    models, seeds, attacks, epsilon, accuracies, data = \
        load_XtoY_attack_results('results_attack_mask_CtoY_sigmoid_True', max_eps=1)

    # plot.plot_eps_vs_accuracy(epsilons=epsilon, accuracies=accuracies, attacks=attacks, modeltype_names=models)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks,
                                       modeltype_names=models, use_seaborn=True, fontsize=fontsize)
    plot.plot_eps_vs_accuracy_multiple(epsilons=epsilon, accuracies=accuracies, attacks=attacks,
                                       modeltype_names=models, normalize=True, use_seaborn=True, fontsize=fontsize)


def show_maskrcnn_stage1_adversarial_results():
    with open(f"{BASE_PATH}/outputs/results_MaskRCNN_attacks.json") as file:
        data = json.load(file)

    models = data['models']
    attacks = data['attacks']
    epsilon = data['epsilon']
    no_bird_pred = data['no_bird_pred']
    total_loss = data['total_loss']
    indiv_losses = data['indiv_losses']
    bird_scores = data['bird_scores']
    top1 = bird_scores['top1_birds']
    avg_birds = bird_scores['avg_birds_detected']
    avg_birds_all = bird_scores['avg_birds_allimgs']

    accuracy_types = ['Top 1', 'Average birds found', 'Average all images']
    accuracies = [top1, avg_birds, avg_birds_all]

    # plot accuracy
    plot.plot_adv_eps_vs_accuracy_maskrcnn(epsilon, accuracies, attacks, models, accuracy_types,
                                           normalize=False, use_seaborn=False, ytitle='Average certainty score')

    # plot number of nobird predictions
    plot.plot_adv_eps_vs_others_maskrcnn(epsilon, no_bird_pred, attacks, models, normalize=False, use_seaborn=False,
                                         ytitle='Images with no bird found')

    # plot number of nobird predictions
    plot.plot_adv_eps_vs_others_maskrcnn(epsilon, total_loss, attacks, models, normalize=False, use_seaborn=False,
                                         ytitle='Total Loss')


def show_maskrcnn_stage1_travBirds_results():
    with open(f"{BASE_PATH}/outputs/results_MaskRCNN_travBirds_CUB.json") as file:
        data_cub = json.load(file)

    with open(f"{BASE_PATH}/outputs/results_MaskRCNN_travBirds.json") as file:
        data = json.load(file)

    models = data['models']
    datasets = data_cub['birds_sets'] + data['birds_sets']
    no_bird_pred = data_cub['no_bird_pred'] + data['no_bird_pred']
    top1 = data_cub['top1_birds'] + data['top1_birds']
    avg_birds = data_cub['avg_birds_detected'] + data['avg_birds_detected']
    avg_birds_all = data_cub['avg_birds_allimgs'] + data['avg_birds_allimgs']

    accuracy_types = ['Top 1', 'Average birds found', 'Average all images']
    accuracies = [top1, avg_birds, avg_birds_all]

    # plot accuracies
    plot.plot_travBirds_bird_accuracies_maskrcnn(accuracy_types, datasets, accuracies, normalize=False,
                                                 ytitle='Average certainty score')

    # plot nobirds bound
    plot.plot_travBirds_bird_not_found_maskrcnn(models, datasets, no_bird_pred, normalize=False, fix_yticks=False,
                                                ytitle='Number of images with no bird found')


def show_mask_stage2_adversarial_forward_pass():
    normalize = True
    max_eps = 1

    models_in, _, _, _, accuracies_in, data_in = load_XtoY_attack_results('results_attack_XtoCtoY_varBIM_fixPGD',
                                                                          max_eps=max_eps)

    # without segmentation
    with open(f"{BASE_PATH}/outputs/results_MaskRCNN_stage2_advpreds_nocm.json") as file:
        data = json.load(file)

    models = data['models']
    attacks = data['attacks']
    epsilon = data['epsilon']
    accuracies = data['accuracies']

    bird_found_acc = data['bird_found_acc']
    nobird_found_acc = data['nobird_found_acc']

    accuracies = [accuracies_in[0]] + accuracies

    fontsize = 'large'
    plot.plot_mask_stage2_eps_vs_accuracy_multiple(epsilon, accuracies, attacks, [models_in[0]] + models,
                                                   normalize=normalize, use_seaborn=False, ylabel='All images Accuracy',
                                                   fontsize=fontsize)
    plot.plot_mask_stage2_eps_vs_accuracy_multiple(epsilon, bird_found_acc, attacks, models, normalize=normalize,
                                                   use_seaborn=False, ylabel='Bird found Accuracy', fontsize=fontsize)
    plot.plot_mask_stage2_eps_vs_accuracy_multiple(epsilon, nobird_found_acc, attacks, models, normalize=normalize,
                                                   use_seaborn=False, ylabel='No bird found Accuracy', fontsize=fontsize)

    # with segmentation
    print('Now with segmentation!')
    max_eps = 1
    # load Inception_v3_25 results
    models_in, _, _, _, accuracies_in, data_in = load_XtoY_attack_results('results_attack_Inception25', max_eps=max_eps)

    with open(f"{BASE_PATH}/outputs/results_MaskRCNN_stage2_advpreds_nocm_useseg.json") as file:
        data = json.load(file)

    models = data['models']
    attacks = data['attacks']
    epsilon = data['epsilon']
    accuracies = data['accuracies']
    accuracies = accuracies_in + accuracies

    bird_found_acc = data['bird_found_acc']
    nobird_found_acc = data['nobird_found_acc']

    plot.plot_mask_stage2_eps_vs_accuracy_multiple(epsilon, accuracies, attacks, models_in + models,
                                                   normalize=normalize, use_seaborn=False, ylabel='All images Accuracy')
    plot.plot_mask_stage2_eps_vs_accuracy_multiple(epsilon, bird_found_acc, attacks, models, normalize=normalize,
                                                   use_seaborn=False, ylabel='Bird found Accuracy')
    plot.plot_mask_stage2_eps_vs_accuracy_multiple(epsilon, nobird_found_acc, attacks, models, normalize=normalize,
                                                   use_seaborn=False, ylabel='No bird found Accuracy')


def show_mask_stage2_travBirds_forward_pass():
    normalize = False
    with open(BASE_PATH + '/outputs/' + 'results_birds_cb.json') as file:
        data_in = json.load(file)
    models_in = data_in['models']
    accuracies_in = data_in['accuracies']

    # no segmentation masks applied
    with open(f"{BASE_PATH}/outputs/results_MaskRCNN_stage2_travBirds_nocm_pred.json", 'rb') as file:
        data = json.load(file)

    models = [models_in[0]] + data['models']
    datasets = data['datasets']
    accuracies = [accuracies_in[0]] + data['accuracies']
    bird_found_acc = data['bird_found_acc']
    nobird_found_acc = data['nobird_found_acc']

    plot.plot_bird_results(models, datasets, accuracies, normalize=normalize, ylabel='All image Accuracy')
    plot.plot_bird_results(data['models'], datasets, bird_found_acc, normalize=normalize, ylabel='Bird found Accuracy')
    plot.plot_bird_results(data['models'], datasets[1:], np.array(nobird_found_acc)[:, :, 1:], normalize=normalize, ylabel='No bird found Accuracy')

    with open(f'{BASE_PATH}/outputs/results_birds_cub_black_inception.json') as file:
        data_inb = json.load(file)
    models_inb = data_inb['models']
    accuracies_inb = data_inb['accuracies']

    # with segmentation masks applied
    with open(f"{BASE_PATH}/outputs/results_MaskRCNN_stage2_travBirds_nocm_pred_useseg.json", 'rb') as file:
        data = json.load(file)

    models = [models_inb[0]] + data['models']
    datasets = data['datasets']
    accuracies = [accuracies_inb[0]] + data['accuracies']
    bird_found_acc = data['bird_found_acc']
    nobird_found_acc = data['nobird_found_acc']

    plot.plot_bird_results(models, datasets, accuracies, normalize=normalize, ylabel='All image Accuracy')
    plot.plot_bird_results(data['models'], datasets, bird_found_acc, normalize=normalize, ylabel='Bird found Accuracy')
    plot.plot_bird_results(data['models'], datasets[1:], np.array(nobird_found_acc)[:, :, 1:], normalize=normalize, ylabel='No bird found Accuracy')


def show_all_birds_results():
    with open(BASE_PATH + '/outputs/' + 'results_birds_cb.json') as file:
        data = json.load(file)
    models = data['models']
    seeds = data['seeds']
    datasets = data['datasets']
    accuracies = data['accuracies']
    plot.plot_birds_acc_and_norm_results(model_names=models, datasets=datasets, accuracies=accuracies)

    with open(BASE_PATH + '/outputs/' + 'results_birds_scouter.json') as file:
        data = json.load(file)
    models = data['models']
    datasets = data['datasets']
    accuracies = data['accuracies']

    with open(f'{BASE_PATH}/outputs/results_birds_inception_25.json') as file:
        data_in = json.load(file)
    models = data_in['models'] + models
    accuracies = data_in['accuracies'] + accuracies
    plot.plot_birds_acc_and_norm_results(model_names=models, datasets=datasets, accuracies=accuracies)

    with open(f'{BASE_PATH}/outputs/results_birds_scouter_variations.json') as file:
        data_var = json.load(file)
    models = data_in['models'] + data_var['models']
    accuracies = data_in['accuracies'] + data_var['accuracies']
    plot.plot_birds_acc_and_norm_results(model_names=models, datasets=datasets, accuracies=accuracies)

    with open(f'{BASE_PATH}/outputs/results_birds_cb.json') as file:
        data_in = json.load(file)
    models_in = data_in['models']
    accuracies_in = data_in['accuracies']
    with open(f'{BASE_PATH}/outputs/results_birds_ViT_new.json') as file:
        data = json.load(file)
    models = data['models']
    datasets = data['datasets']
    accuracies = data['accuracies']
    models = [models_in[0]] + models
    accuracies = [accuracies_in[0]] + accuracies
    plot.plot_birds_acc_and_norm_results(model_names=models, datasets=datasets, accuracies=accuracies)

    with open(f'{BASE_PATH}/outputs/results_birds_cub_black_inception.json') as file:
        data = json.load(file)
    models = data['models']
    datasets = data['datasets']
    accuracies = data['accuracies']
    plot.plot_birds_acc_and_norm_results(model_names=models, datasets=datasets, accuracies=accuracies)


def convert_txtdict_to_json():
    with open(f"{BASE_PATH}/outputs/results_MaskRCNN_stage2_travBirds_pred_useseg.txt", 'rb') as file:
        dic = yaml.safe_load(file)

    with open(f"{BASE_PATH}/outputs/test.json", 'w') as file:
        json.dump(dic, file)


if __name__ == '__main__':
    print('this is the path: ', BASE_PATH)
    # show_vit_results()
    # show_all_normalized_results()

    #show_all_birds_results()

    #show_mask_stage2_XtoC_results()
    # show_mask_stage2_CtoY_results()
    #show_mask_stage2_attack_XtoY_results()

    #show_maskrcnn_stage1_adversarial_results()
    #show_maskrcnn_stage1_travBirds_results()

    show_mask_stage2_adversarial_forward_pass()
    #show_mask_stage2_travBirds_forward_pass()

    # convert_txtdict_to_json()

    #show_cb_cub_black_segmentation_results()

    # show_cb_XtoC_results()
    # show_cb_CtoY_results()
    # show_cb_XtoCtoY_results()
    # show_cb_birds_results()

    # show_scouter_birds_results()
    # show_scouter_results()

    #show_vit_adversarial_results()
    #show_vit_birds_results()

    # show_cub_black_XtoCtoY_results()
    # show_inception25_birds_results()
    # show_inception_black_birds_results()
    # show_scouter_variations_birds_results()
