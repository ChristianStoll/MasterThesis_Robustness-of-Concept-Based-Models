import torch
import torchattacks

import os
import random
import json
import numpy as np
import re

import utils_plot as plot
import utils
from dataset import load_data_cb, load_data_scouter, load_data_vit, load_data_MaskBottleneck, load_travelling_birds
from config import model_dirs, n_attributes
from config import BASE_PATH
from model_templates.utils_models import Normalization, SelectOutput, FullBNModel, FullMaskStage2Model

import adapted_torchattacks

random.seed(123)
torch.set_num_threads(4)


def attack_concepts(model_names, seeds, eps, attacks, use_sigmoid, num_examples, batch_size, rtpt=None):  # CtoY
    model_accuracies = []
    initially_wrong_preds = []
    right_from_initially_wrong_predictions = []

    for m in model_names:
        # setup dataset and dataloader

        if 'cb' in m.lower():
            data_dir = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl'.replace("\\", '/')
            loader = load_data_cb([data_dir], use_attr=True, no_img=False, batch_size=batch_size, image_dir='images',
                                  normalize=True)
        elif 'mask' in m.lower():
            # load args of concept model, since independent one might not contain the required information
            m_name = '_'.join(m.split('_')[1:])
            if 'Independent' in m:
                concept_name = m_name.replace('Independent', 'Concept').replace('mask', '')
            else:  # Sequential
                concept_name = m_name.replace('Sequential', 'Concept').replace('mask', '')
            # use the first seed for the args file, since except for seed related arguments, all others are equal
            args = utils.load_model_args_as_dict(f'{BASE_PATH}/models/MaskBottleneck/{concept_name}/Seed1/args.yaml')
            data_dir = os.path.join(BASE_PATH, args['data_dir'], 'test.pkl').replace("\\", '/')

            loader = load_data_MaskBottleneck([data_dir], use_attr=True, no_img=False, batch_size=batch_size,
                                              normalize=True, num_classes=args['num_classes'],
                                              crop_type=args['crop_type'],
                                              apply_segmentation=args['apply_segmentation'])
        else:
            print(f'no loader assigned for model {m}')
            exit()

        # this is used later to calculate the accuracy
        if num_examples is not None:
            num_imgs = num_examples
        else:
            num_imgs = len(loader.dataset)

        seed_accs = []
        m_iwp = []
        m_rfiwp = []
        for seed in seeds:
            if rtpt is not None:
                rtpt.step()

            if 'cb' in m.lower():
                concept_dict_dir = f"{BASE_PATH}/models/ConceptBottleneck/{model_dirs['cb_concept']}{seed}.pth" \
                    .replace("\\", '/')
                model_dict_dir = f"{BASE_PATH}/models/ConceptBottleneck/{model_dirs[m]}{seed}.pth".replace("\\", '/')

                # initialize concept model
                concept_model = utils.load_model(model_name='cb_concept', path_to_state_dict=concept_dict_dir)
                concept_model.eval()
                concept_model.cuda()

                # initialize model
                model = utils.load_model(model_name=m, path_to_state_dict=model_dict_dir)

            elif 'mask' in m.lower():
                if 'independent' in m.lower():
                    concept_path_name = '_'.join(m.split('_')[1:]).replace('Independent', 'Concept')
                    concept_name = m.replace('Independent', 'Concept')
                    end_name = 'Independent'
                else:  # Sequential
                    concept_path_name = '_'.join(m.split('_')[1:]).replace('Sequential', 'Concept')
                    concept_name = m.replace('Sequential', 'Concept')
                    end_name = '_'.join(m.split('_')[1:])

                concept_dict_dir = \
                    f"{BASE_PATH}/models/MaskBottleneck/{concept_path_name}/Seed{seed}/best_model.pth".replace("\\",
                                                                                                               '/')
                concept_model = utils.load_model(model_name=concept_name, path_to_state_dict=concept_dict_dir)
                concept_model.eval()
                concept_model.cuda()

                model_dict_dir = \
                    f'{BASE_PATH}/models/MaskBottleneck/{end_name}/Seed{seed}/best_model.pth'.replace("\\", '/')
                model = utils.load_model(model_name=m, path_to_state_dict=model_dict_dir)
            else:
                print('no model or concept model loaded!')
                exit()

            model.eval()
            model.cuda()

            accuracies = []
            s_iwp = []
            s_rfiwp = []
            for attack in attacks:
                accs = []
                att_iwp = []
                att_rfiwp = []
                print(f'model: {m}, seed: {seed}, attack: {attack}')
                for epsilon in eps:
                    correct = 0
                    cnt = 0
                    iwp = 0
                    rfiwp = 0

                    cnt_images = 0
                    for _, data in enumerate(loader):
                        cnt += 1
                        # data: [img_as_tensor, class_number_as_tensor, ggf. attribute as list]
                        img, class_labels, attribute_labels = data

                        img_var = torch.autograd.Variable(img).cuda()
                        class_labels_var = torch.autograd.Variable(class_labels).cuda()

                        # the input to the attacking model is the initial prediction of the concept model
                        attributes = concept_model(img_var)

                        if 'cb' in m.lower():
                            attributes = torch.cat(attributes, dim=1)

                        if use_sigmoid:
                            input_adv = torch.nn.Sigmoid()(attributes)
                        else:
                            input_adv = attributes

                        # check how many attributes are predicted wrong
                        init_pred = (model(input_adv).max(1, keepdim=True)[1]).detach().cpu()

                        # Check for initially wrong predictions
                        initially_predicted_labels = torch.eq(init_pred.squeeze(), class_labels).numpy()
                        out, counts = np.unique(initially_predicted_labels, return_counts=True)
                        if np.where(out == False)[0].size > 0:
                            iwp += counts[np.where(out == False)[0][0]]

                        # cast to float to be able to optimize input tensor
                        input_adv.float()

                        if epsilon != 0.0:
                            if attack == 'bim':
                                steps = min(max(int(min(epsilon * 255 + 4, 1.25 * epsilon * 255)), 1), 7)
                                atk = torchattacks.BIM(model=model, eps=epsilon, alpha=1 / 255, steps=steps)
                                perturbed = atk(input_adv, class_labels_var)
                            elif attack == 'pgd':
                                atk = torchattacks.PGD(model=model, eps=epsilon, alpha=1 / 255, steps=2)
                                perturbed = atk(input_adv, class_labels_var)
                            elif attack == 'fgsm':
                                atk = torchattacks.FGSM(model=model, eps=epsilon)
                                perturbed = atk(input_adv, class_labels_var)
                            elif attack == 'sparsefool':
                                if cnt + 1 * batch_size > 100:
                                    if cnt * batch_size >= 100:
                                        remaining = 100 - batch_size * cnt
                                        input_adv = input_adv[:remaining]
                                        class_labels_var = class_labels_var[:remaining]
                                        class_labels = class_labels[:remaining]
                                        initially_predicted_labels = initially_predicted_labels[:remaining]
                                atk = adapted_torchattacks.SparseFool(model, steps=2, steps_deepfool=2,
                                                                      lam=epsilon * 10, overshoot=0.02)
                                perturbed = atk(input_adv, class_labels_var)
                            else:
                                print('no attack was chosen')
                                exit()
                        else:
                            if attack == 'sparsefool' and cnt * batch_size >= 100:
                                remaining = 100 - batch_size * (cnt - 1)
                                input_adv = input_adv[:remaining]
                                class_labels_var = class_labels_var[:remaining]
                                class_labels = class_labels[:remaining]
                                initially_predicted_labels = initially_predicted_labels[:remaining]
                            perturbed = input_adv.clone().detach()
                        cnt_images += perturbed.shape[0]

                        # get the newly perturbed/predicted class
                        adv_pred = model(perturbed)
                        new_pred = (adv_pred.max(1, keepdim=True)[1]).detach().cpu()

                        # Check for successful predictions
                        correctly_predicted_labels = torch.eq(new_pred.squeeze(), class_labels).numpy()
                        out, counts = np.unique(correctly_predicted_labels, return_counts=True)
                        if np.where(out == True)[0].size > 0:
                            correct += counts[np.where(out)[0][0]]

                        # right_from_initially_wrong_predictions
                        # get relevant locations:
                        value, counts = np.unique(
                            correctly_predicted_labels[np.where(initially_predicted_labels==False)],
                            return_counts=True)

                        if np.where(value == True)[0].size > 0:
                            rfiwp += counts[np.where(value)[0][0]]

                        # stop conditions: general case if not the full dataset is used
                        if num_examples is not None and cnt >= num_examples:
                            break
                        # lower number of examples for sparsefool
                        if attack == 'sparsefool' and cnt_images >= 99:
                            break

                    # Calculate final accuracy for this epsilon
                    if attack != 'sparsefool':
                        if num_examples is not None:
                            final_acc = correct / float(num_imgs * batch_size)
                            print("Epsilon: {}\tTest Accuracy = {} / {} = {}"
                                  .format(epsilon, correct, num_imgs * batch_size, final_acc))
                        else:
                            final_acc = correct / float(len(loader.dataset))
                            print("Epsilon: {}\tTest Accuracy = {} / {} = {}"
                                  .format(epsilon, correct, len(loader.dataset), final_acc))
                    else:
                        final_acc = correct / float(100)
                        print("Epsilon: {}\tTest Accuracy = {} / {} = {}"
                              .format(epsilon, correct, 100, final_acc))
                    accs.append(final_acc)
                    att_iwp.append(int(iwp))
                    att_rfiwp.append(int(rfiwp))

                accuracies.append(accs)
                s_iwp.append(att_iwp)
                s_rfiwp.append(att_rfiwp)

            seed_accs.append(accuracies)
            m_iwp.append(s_iwp)
            m_rfiwp.append(s_rfiwp)

        model_accuracies.append(seed_accs)
        initially_wrong_preds.append(m_iwp)
        right_from_initially_wrong_predictions.append(m_rfiwp)

    return model_accuracies, initially_wrong_preds, right_from_initially_wrong_predictions


def attack_image(model_names, seeds, eps, attacks, batch_size, num_examples=None, use_segmentation=False, rtpt=None,
                 save_perturbations=False, use_attr_to_check_concepts=False):
    model_accuracies = []
    if use_attr_to_check_concepts:
        model_num_wrong = []
        model_conf_matrix = []
        model_dataset_stats = []
    for m in model_names:
        # setup dataset & dataloader
        if 'cb' in m or 'inception-v3' in m:
            data_dir = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl'.replace("\\", '/')
            if 'cub_black' in m:
                image_dir = ('AdversarialData/CUB_black/').replace("\\", '/')
                # TODO
                # loader = load_travelling_birds([data_dir], image_dir='CUB_black', batch_size=batch_size, img_size=256)

                loader = load_data_cb([data_dir], use_attr=use_attr_to_check_concepts, no_img=False,
                                      batch_size=batch_size, image_dir=image_dir, n_class_attr=2, normalize=False,
                                      use_segmentation=use_segmentation)
            else:
                if 'inception-v3' in m:
                    num_classes = int(re.findall(r'\d+', m)[-1])
                else:
                    num_classes = 200
                loader = load_data_cb([data_dir], use_attr=use_attr_to_check_concepts, no_img=False,
                                      batch_size=batch_size, num_classes=num_classes, image_dir='images',
                                      normalize=False, n_class_attr=2)

        elif 'scouter' in m:
            num_classes = int(re.findall(r'\d+', m)[0])
            data_dir = (BASE_PATH + '/data/CUB_200_2011/').replace("\\", '/')
            loader = load_data_scouter(data_dir=data_dir, num_classes=num_classes, batch_size=batch_size, train=False,
                                       normalize=False)
        elif 'vit' in m:
            data_dir = f'{BASE_PATH}/data/CUB_200_2011/'.replace("\\", '/')
            img_size = int(m.lower().split('_')[-1])
            loader = load_data_vit(data_dir=data_dir, num_classes=200, batch_size=batch_size, img_size=img_size,
                                   normalize=False)
        elif 'mask' in m.lower():
            concept_path_name = '_'.join(m.split('_')[1:]).replace('Independent', 'Concept')

            # use the first seed for the args file, since except for seed related arguments, all others are equal
            # use info from concept model, since independent version does not contain the required information
            args = \
                utils.load_model_args_as_dict(f'{BASE_PATH}/models/MaskBottleneck/{concept_path_name}/Seed1/args.yaml')
            data_dir = os.path.join(BASE_PATH, args['data_dir'], 'test.pkl').replace("\\", '/')
            loader = load_data_MaskBottleneck([data_dir], use_attr=False, no_img=False,
                                              batch_size=batch_size, normalize=False, num_classes=args['num_classes'],
                                              crop_type=args['crop_type'],
                                              apply_segmentation=args['apply_segmentation'],
                                              load_segmentation=args['apply_segmentation'])
        else:
            print('no dataset has been chosen')
            exit()

        # this is used later to calculate the accuracy
        if num_examples is not None:
            num_imgs = num_examples
        else:
            num_imgs = len(loader.dataset)

        model_seed_accs = []
        if use_attr_to_check_concepts:
            seed_num_wrong = []
            seed_conf_matrix = []
            seed_dataset_stats = []
        for seed in seeds:
            if rtpt is not None:
                rtpt.step()
            # initialize model
            if 'cb' in m:
                model_dict_dir = \
                    (BASE_PATH + '/models/ConceptBottleneck/' + model_dirs[m] + str(seed) + '.pth').replace("\\", '/')
                concept_dict_dir = \
                    (BASE_PATH + '/models/ConceptBottleneck/' + model_dirs['cb_concept'] + str(seed) + '.pth') \
                        .replace("\\", '/')
                if 'independent' in m.lower() or 'sequential' in m.lower():
                    model = torch.nn.Sequential(
                        Normalization(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
                        FullBNModel(
                            concept_model=utils.load_model(model_name='cb_concept',
                                                           path_to_state_dict=concept_dict_dir),
                            end_model=utils.load_model(model_name=m, path_to_state_dict=model_dict_dir),
                            use_sigmoid=True, return_attributes=False)
                    )
                    if use_attr_to_check_concepts:
                        model_concept = utils.load_model(model_name='cb_concept', path_to_state_dict=concept_dict_dir)
                        model_concept.cuda()
                        model_concept.eval()
            elif 'inception-v3' in m.lower():
                model_dict_dir = f'{BASE_PATH}/models/ConceptBottleneck/{model_dirs[m]}{seed}.pth'.replace("\\", '/')
                model = torch.nn.Sequential(
                    Normalization(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
                    utils.load_model(model_name=m, path_to_state_dict=model_dict_dir),
                    SelectOutput()
                )
            elif 'scouter' in m.lower():
                scouter_dict_dir = f'{BASE_PATH}/models/{model_dirs[m]}.pth'.replace("\\", '/').replace('+', str(seed))
                model = torch.nn.Sequential(
                    Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    utils.load_model(m, path_to_state_dict=scouter_dict_dir)
                )
                torch.backends.cudnn.enabled = False
            elif 'vit' in m.lower():
                model_dict_dir = \
                    f'{BASE_PATH}/models/{model_dirs[m]}{seed}/model_best.pth'.replace("\\", '/').replace('+',
                                                                                                          str(seed))
                model = torch.nn.Sequential(
                    Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    utils.load_model(m, model_dict_dir)
                )
            elif 'mask' in m.lower():
                if 'independent' in m.lower():
                    concept_path_name = '_'.join(m.split('_')[1:]).replace('Independent', 'Concept')
                    concept_name = m.replace('Independent', 'Concept')
                    end_name = 'Independent'
                else:  # Sequential
                    concept_path_name = '_'.join(m.split('_')[1:]).replace('Sequential', 'Concept')
                    concept_name = m.replace('Sequential', 'Concept')
                    end_name = '_'.join(m.split('_')[1:])

                concept_dict_dir = \
                    f"{BASE_PATH}/models/MaskBottleneck/{concept_path_name}/Seed{seed}/best_model.pth".replace("\\",
                                                                                                               '/')
                end_dict_dir = \
                    f'{BASE_PATH}/models/MaskBottleneck/{end_name}/Seed{seed}/best_model.pth'.replace("\\", '/')

                model = torch.nn.Sequential(
                    Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    FullMaskStage2Model(
                        concept_model=utils.load_model(model_name=concept_name, path_to_state_dict=concept_dict_dir),
                        end_model=utils.load_model(model_name=m, path_to_state_dict=end_dict_dir), use_sigmoid=True)
                )
            else:
                print('no model has been chosen')
                exit()

            model.cuda()
            model.eval()

            accuracies = []
            if use_attr_to_check_concepts:
                atk_num_wrong = []
                atk_conf_matrix = []
                atk_dataset_stats = []
            for attack in attacks:
                print(f'model: {m}, seed: {seed}, attack: {attack}')
                accs = []
                eps_num_wrong = []
                eps_conf_matrix = []
                stats = []
                for epsilon in eps:
                    correct = 0
                    cnt = 0
                    cnt_images = 0

                    # these are only used in case the standard model is checked
                    if use_attr_to_check_concepts:
                        total_wrong = 0
                        eq_0_wrong, leq_1_wrong, leq_3_wrong, leq_5_wrong, leq_10_wrong, g_10_wrong = 0, 0, 0, 0, 0, 0
                        c_matrix = []  # list of confusion matrices

                        # calculate mean & std for each perturbed dataset
                        mean = 0.
                        std = 0.
                        nb_samples = 0.

                    for idx, data in enumerate(loader):
                        cnt += 1
                        # data: [img_as_tensor, class_number_as_tensor, ggf. attribute as list]

                        if 'mask' in m.lower() and use_segmentation:
                            inputs, labels, _, segmentations = data
                        elif use_segmentation:
                            inputs, labels, segmentations = data
                            segmentations.cuda()
                        else:
                            segmentations = None
                            if use_attr_to_check_concepts:
                                inputs, labels, attribute_labels = data
                                attribute_labels = torch.stack(attribute_labels).t()
                            else:
                                inputs, labels = data

                        input_adv = torch.autograd.Variable(inputs).cuda()
                        labels_var = torch.autograd.Variable(labels).cuda()

                        # cast to float to be able to optimize input tensor
                        input_adv.float()

                        if epsilon != 0:
                            if attack == 'bim':
                                steps = min(max(int(min(epsilon * 255 + 4, 1.25 * epsilon * 255)), 1), 7)
                                atk = adapted_torchattacks.BIM(model=model, eps=epsilon, alpha=1 / 255, steps=steps,
                                                               use_segmentation=use_segmentation)
                                perturbed = atk(input_adv, labels_var, segmentations)
                            elif attack == 'pgd':
                                atk = adapted_torchattacks.PGD(model=model, eps=epsilon, alpha=1 / 255, steps=2,
                                                               use_segmentation=use_segmentation)
                                perturbed = atk(input_adv, labels_var, segmentations)
                            elif attack == 'fgsm':
                                atk = adapted_torchattacks.FGSM(model=model, eps=epsilon,
                                                                use_segmentation=use_segmentation)
                                perturbed = atk(input_adv, labels_var, segmentations)
                            elif attack == 'sparsefool':
                                if cnt * batch_size >= 100:
                                    remaining = 100 - batch_size * cnt
                                    input_adv = input_adv[:remaining]
                                    labels_var = labels_var[:remaining]
                                    labels = labels[:remaining]
                                atk = adapted_torchattacks.SparseFool(model, steps=2, steps_deepfool=2,
                                                                      lam=epsilon * 10,
                                                                      overshoot=0.02, use_segmentation=use_segmentation)
                                perturbed = atk(input_adv, labels_var, segmentations)
                            else:
                                print('no attack was chosen')
                                exit()
                        else:
                            if attack == 'sparsefool' and cnt * batch_size >= 100:
                                remaining = 100 - batch_size * (cnt - 1)
                                input_adv = input_adv[:remaining]
                                labels_var = labels_var[:remaining]
                                labels = labels[:remaining]
                            perturbed = input_adv.clone().detach()
                        cnt_images += perturbed.shape[0]

                        # get the newly perturbed/predicted class
                        adv_pred = model(perturbed)
                        new_pred = (adv_pred.max(1, keepdim=True)[1]).detach().cpu()

                        if save_perturbations and cnt_images < 100:
                            dir_to_folder = f"outputs/examples/{m}_{attack}_{epsilon}_seed{seed}/"
                            os.makedirs(dir_to_folder, exist_ok=True)

                            for i in range(labels.shape[0]):
                                plot.save_adversarial_image(original=inputs[i], perturbed=perturbed[i], label=labels[i],
                                                            prediction=new_pred[i], index=idx,
                                                            dir_to_folder=dir_to_folder)

                        if use_attr_to_check_concepts:
                            if 'independent' in m.lower() or 'sequential' in m.lower():
                                concepts = model_concept(perturbed)
                                adv_sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(concepts, dim=1))

                                # Check for successful/wrong predictions
                                tot, cm, eq0, leq1, leq3, leq5, leq10, g10 = utils.check_attribute_predictions(
                                    adv_sigmoid_outputs, attribute_labels)

                                total_wrong += tot
                                eq_0_wrong += eq0
                                leq_1_wrong += leq1
                                leq_3_wrong += leq3
                                leq_5_wrong += leq5
                                leq_10_wrong += leq10
                                g_10_wrong += g10
                                c_matrix.append(cm)

                            # calculate mean & std over perturbed datasets
                            batch_samples = perturbed.size(0)
                            perturbed_view = perturbed.view(batch_samples, perturbed.size(1), -1)
                            mean += perturbed_view.mean(2).sum(0)
                            std += perturbed_view.std(2).sum(0)
                            nb_samples += batch_samples

                        # Check for success
                        correctly_predicted_labels = torch.eq(new_pred.squeeze(), labels).numpy()
                        out, counts = np.unique(correctly_predicted_labels, return_counts=True)

                        if np.where(out == True)[0].size > 0:
                            correct += counts[np.where(out == True)[0][0]]

                        # stop conditions: general case if not the full dataset is used
                        if num_examples is not None and cnt >= num_examples - 1:
                            break
                        # lower number of examples for sparsefool
                        if attack == 'sparsefool' and cnt_images >= 99:
                            break

                    # Calculate final accuracy for this epsilon
                    if attack != 'sparsefool':
                        if num_examples is not None:
                            final_acc = correct / float(num_imgs * batch_size)
                            print("Epsilon: {}\tTest Accuracy = {} / {} = {}"
                                  .format(epsilon, correct, num_imgs * batch_size, final_acc))
                        else:
                            final_acc = correct / float(len(loader.dataset))
                            print("Epsilon: {}\tTest Accuracy = {} / {} = {}"
                                  .format(epsilon, correct, len(loader.dataset), final_acc))
                    else:
                        final_acc = correct / float(100)
                        print("Epsilon: {}\tTest Accuracy = {} / {} = {}"
                              .format(epsilon, correct, 100, final_acc))
                    accs.append(final_acc)

                    if use_attr_to_check_concepts:
                        if 'independent' in m.lower() or 'sequential' in m.lower():
                            eps_num_wrong.append((eq_0_wrong / len(loader.dataset), leq_1_wrong / len(loader.dataset),
                                                  leq_3_wrong / len(loader.dataset), leq_5_wrong / len(loader.dataset),
                                                  leq_10_wrong / len(loader.dataset), g_10_wrong / len(loader.dataset)))
                            eps_conf_matrix.append(c_matrix)

                        mean /= nb_samples
                        std /= nb_samples
                        stats.append({'mean: ': mean.tolist(), 'std: ': std.tolist()})
                accuracies.append(accs)
                if use_attr_to_check_concepts:
                    atk_num_wrong.append(eps_num_wrong)
                    atk_conf_matrix.append(eps_conf_matrix)
                    atk_dataset_stats.append(stats)
            model_seed_accs.append(accuracies)
            if use_attr_to_check_concepts:
                seed_num_wrong.append(atk_num_wrong)
                seed_conf_matrix.append(atk_conf_matrix)
                seed_dataset_stats.append(atk_dataset_stats)
        model_accuracies.append(model_seed_accs)
        if use_attr_to_check_concepts:
            model_num_wrong.append(seed_num_wrong)
            model_conf_matrix.append(seed_conf_matrix)
            model_dataset_stats.append(seed_dataset_stats)

    if use_attr_to_check_concepts:
        if 'independent' in m.lower() or 'sequential' in m.lower():
            return model_accuracies, model_dataset_stats, model_num_wrong, model_conf_matrix, len(loader.dataset)
        else:
            return model_accuracies, model_dataset_stats
    else:
        return model_accuracies


def attack_image_XtoC(model_names, seeds, eps, attacks, batch_size, num_examples=None, rtpt=None):
    # setup dataset & dataloader
    model_accuracies = []
    model_num_wrong = []
    model_conf_matrix = []
    for m in model_names:
        if 'cb' in m.lower():
            data_dir = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl'.replace("\\", '/')
            loader = load_data_cb([data_dir], use_attr=True, no_img=False, batch_size=batch_size, image_dir='images',
                                  normalize=False)
        elif 'mask' in m.lower():
            m_name = '_'.join(m.split('_')[1:])
            # use the first seed for the args file, since except for seed related arguments, all others are equal
            args = utils.load_model_args_as_dict(f'{BASE_PATH}/models/MaskBottleneck/{m_name}/Seed1/args.yaml')
            data_dir = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/masked_attr_data/test.pkl'.replace("\\", '/')

            loader = load_data_MaskBottleneck([data_dir], use_attr=args['use_attr'], no_img=False,
                                              batch_size=batch_size, normalize=False, num_classes=args['num_classes'],
                                              crop_type=args['crop_type'],
                                              apply_segmentation=args['apply_segmentation'],
                                              load_segmentation=args['apply_segmentation'])

        # this is used later to calculate the accuracy
        if num_examples is not None:
            num_imgs = num_examples
        else:
            num_imgs = len(loader.dataset)

        seed_accuracies = []
        seed_num_wrong = []
        seed_conf_matrix = []
        for seed in seeds:
            if rtpt is not None:
                rtpt.step()

            # setup concept model
            if 'cb' in m.lower():
                concept_dict_dir = f'{BASE_PATH}/models/ConceptBottleneck/{model_dirs[m]}{seed}.pth'.replace("\\", '/')
                model = torch.nn.Sequential(
                    Normalization(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
                    utils.load_model(model_name='cb_concept', path_to_state_dict=concept_dict_dir)
                )
            elif 'mask' in m.lower():
                m_name = '_'.join(m.split('_')[1:])
                concept_dict_dir = \
                    f"{BASE_PATH}/models/MaskBottleneck/{m_name}/Seed{seed}/best_model.pth".replace("\\", '/')
                model = torch.nn.Sequential(
                    Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    utils.load_model(model_name=m, path_to_state_dict=concept_dict_dir)
                )

            model.eval()
            model.cuda()

            accuracies = []
            atk_num_wrong = []
            atk_conf_matrix = []
            for attack in attacks:
                print(f'model: {m}, seed: {seed}, attack: {attack}')
                accs = []
                eps_num_wrong = []
                eps_conf_matrix = []
                for epsilon in eps:
                    total_wrong = 0
                    eq_0_wrong = 0
                    leq_1_wrong = 0
                    leq_3_wrong = 0
                    leq_5_wrong = 0
                    leq_10_wrong = 0
                    g_10_wrong = 0
                    c_matrix = []  # list of confusion matrices

                    cnt = 0
                    for _, data in enumerate(loader):
                        cnt += 1
                        # data: [img_as_tensor, class_number_as_tensor, ggf. attribute as list]
                        if 'mask' in m.lower() and args['apply_segmentation']:
                            use_seg = True
                            img, class_labels, attribute_labels, segmentation = data
                            segmentation = segmentation.cuda()
                        else:
                            img, class_labels, attribute_labels = data
                            use_seg = False
                            segmentation = None

                        if 'cb' in m.lower():
                            attribute_labels = torch.stack(attribute_labels).t()
                        attr_labels_var = attribute_labels.float().clone().cuda()

                        input_adv = torch.autograd.Variable(img).cuda()
                        # cast to float to be able to optimize input tensor
                        input_adv.float()

                        if attack == 'bim':
                            steps = min(max(int(min(epsilon * 255 + 4, 1.25 * epsilon * 255)), 1), 7)
                            atk = adapted_torchattacks.BIM(model=model, eps=epsilon, alpha=2 / 255, steps=steps,
                                                           XtoC=m, use_segmentation=use_seg)
                            perturbed = atk(input_adv, attr_labels_var, segmentations=segmentation)
                        elif attack == 'pgd':
                            atk = adapted_torchattacks.PGD(model=model, eps=epsilon, alpha=2 / 255, steps=2, XtoC=m,
                                                           use_segmentation=use_seg)
                            perturbed = atk(input_adv, attr_labels_var, segmentations=segmentation)
                        elif attack == 'fgsm':
                            atk = adapted_torchattacks.FGSM(model=model, eps=epsilon, XtoC=m, use_segmentation=use_seg)
                            perturbed = atk(input_adv, attr_labels_var, segmentations=segmentation)
                        else:
                            print('no attack was chosen')
                            exit()

                        # get the newly perturbed/predicted class
                        adv_pred = model(perturbed)
                        if 'cb' in m.lower():
                            adv_pred = torch.nn.Sigmoid()(torch.cat(adv_pred, dim=1))
                        elif 'hybrid' in m.lower():
                            # different handling for slots:  sklearn's confusion matrix only handles 2D (BxD) arrays
                            adv_pred = utils.hungarian_matching(attr_labels_var, adv_pred)
                            adv_pred = adv_pred.reshape(adv_pred.shape[0], -1)
                            attribute_labels = attribute_labels.reshape(adv_pred.shape[0], -1)

                        # Check for successful/wrong predictions
                        tot, cm, eq0, leq1, leq3, leq5, leq10, g10 = \
                            utils.check_attribute_predictions(adv_pred, attribute_labels)
                        total_wrong += tot
                        eq_0_wrong += eq0
                        leq_1_wrong += leq1
                        leq_3_wrong += leq3
                        leq_5_wrong += leq5
                        leq_10_wrong += leq10
                        g_10_wrong += g10
                        c_matrix.append(cm)

                        if num_examples is not None and cnt >= num_examples - 1:
                            break

                    # Calculate final accuracy for this epsilon
                    if num_examples is not None:
                        correct = (n_attributes * num_imgs * batch_size) - total_wrong
                        final_acc = correct / float(num_imgs * batch_size * n_attributes)
                        print("Epsilon: {}\tTest Accuracy = {} / {} = {}"
                              .format(epsilon, correct, n_attributes * num_imgs * batch_size, final_acc))
                    else:
                        correct = n_attributes * len(loader.dataset) - total_wrong
                        final_acc = correct / float(len(loader.dataset) * n_attributes)
                        print("Epsilon: {}\tTest Accuracy = {} / {} = {}"
                              .format(epsilon, correct, n_attributes * len(loader.dataset), final_acc))
                    accs.append(final_acc)
                    eps_num_wrong.append((eq_0_wrong / len(loader.dataset), leq_1_wrong / len(loader.dataset),
                                          leq_3_wrong / len(loader.dataset), leq_5_wrong / len(loader.dataset),
                                          leq_10_wrong / len(loader.dataset), g_10_wrong / len(loader.dataset)))
                    eps_conf_matrix.append(c_matrix)

                accuracies.append(accs)
                atk_num_wrong.append(eps_num_wrong)
                atk_conf_matrix.append(eps_conf_matrix)
            seed_accuracies.append(accuracies)
            seed_num_wrong.append(atk_num_wrong)
            seed_conf_matrix.append(atk_conf_matrix)
        model_accuracies.append(seed_accuracies)
        model_num_wrong.append(seed_num_wrong)
        model_conf_matrix.append(seed_conf_matrix)

    return model_accuracies, model_conf_matrix, model_num_wrong, n_attributes * len(loader.dataset)


def travellingBirds_test(model_names, seeds, batch_size,
                         image_dirs=['cub', 'CUB_black', 'CUB_fixed/test', 'CUB_random'], rtpt=None):
    accuracies = []
    for m in model_names:
        if rtpt is not None:
            rtpt.step()

        seed_accs = []
        for seed in seeds:
            if 'cb' in m:
                concept_dict_dir = \
                    (f'{BASE_PATH}/models/ConceptBottleneck/' + model_dirs['cb_concept'] + f'{seed}/best_model.pth') \
                        .replace("\\", '/')
                model_dict_dir = f'{BASE_PATH}/models/ConceptBottleneck/{model_dirs[m]}{seed}.pth'.replace("\\", '/')
                if 'independent' in m.lower() or 'sequential' in m.lower():
                    model = FullBNModel(concept_model=
                                        utils.load_model(model_name='cb_concept', path_to_state_dict=concept_dict_dir),
                                        end_model=utils.load_model(model_name=m, path_to_state_dict=model_dict_dir),
                                        use_sigmoid=True, return_attributes=False)
            elif 'inception-v3' in m.lower():
                model_dict_dir = f'{BASE_PATH}/models/ConceptBottleneck/{model_dirs[m]}{seed}.pth'.replace("\\", '/')
                model = torch.nn.Sequential(
                    utils.load_model(model_name=m, path_to_state_dict=model_dict_dir),
                    SelectOutput()
                )
            elif 'scouter' in m.lower():
                scouter_dict_dir = f'{BASE_PATH}/models/{model_dirs[m]}'.replace('\\', '/').replace('+', str(seed))
                model = utils.load_model(m, path_to_state_dict=scouter_dict_dir)
                torch.backends.cudnn.enabled = False  # this is a workaround for not setting the resnet to train mode
            elif 'vit' in m.lower():
                model_dict_dir = \
                    f'{BASE_PATH}/models/{model_dirs[m]}{seed}/model_best.pth'.replace("\\", '/').replace('+', str(seed))
                model = utils.load_model(m, model_dict_dir)
            elif 'hybrid' in m.lower():
                hybrid_concept_dict_dir = \
                    (f'{BASE_PATH}/models/' + model_dirs['HybridConcept'] + f'{seed}/best_model.pth').replace("\\", '/')
                end_dict_dir = f'{BASE_PATH}/models/{model_dirs[m]}{seed}/best_model.pth'.replace("\\", '/')
                model = FullBNModel(concept_model=utils.load_model(model_name='HybridConcept',
                                                                   path_to_state_dict=hybrid_concept_dict_dir),
                                    end_model=utils.load_model(model_name=m, path_to_state_dict=end_dict_dir),
                                    use_sigmoid=True, return_attributes=False)
            else:
                print('no model has been chosen')
                exit()

            model.cuda()
            model.eval()

            dataset_accs = []
            for im_dir in image_dirs:
                print('model: ' + m + ', seed: ' + str(seed) + ', dataset: ' + im_dir)

                # number of classes is different from 200 for some models, image size varies for different models
                num_classes = 200
                isConceptBottleneck = False
                if 'cb' in m.lower() or 'inception-v3' in m:
                    img_size = 299
                    num_classes = int(re.findall(r'\d+', m)[-1])
                    isConceptBottleneck = True
                elif 'scouter' in m.lower():
                    img_size = 260
                    num_classes = int(re.findall(r'\d+', m)[0])
                elif 'vit' in m.lower():
                    img_size = int(m.lower().split('_')[-1])
                elif 'hybrid' in m.lower():
                    path_to_yaml = '/'.join(hybrid_concept_dict_dir.split('/')[:-1] + ['args.yaml'])
                    img_size = utils.load_model_args_as_dict(path_to_yaml)['img_size']

                # setup dataset & dataloader
                # load TravellingBirds (adversarial) dataset
                if im_dir != 'cub':
                    data_dir = (BASE_PATH + '/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl') \
                        .replace("\\", '/')

                    loader = load_travelling_birds([data_dir], image_dir=im_dir, num_classes=num_classes,
                                                   batch_size=batch_size, img_size=img_size,
                                                   isConceptBottleneck=isConceptBottleneck)
                # load CUB_200_2011 dataset
                else:
                    if 'cb' in m.lower() or 'inception-v3' in m:
                        data_dir = (BASE_PATH + '/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl') \
                            .replace("\\", '/')
                        loader = load_data_cb([data_dir], use_attr=False, no_img=False, batch_size=batch_size,
                                              num_classes=num_classes, image_dir='images', n_class_attr=2,
                                              normalize=True)
                    elif 'scouter' in m.lower():
                        if im_dir == 'cub':
                            data_dir = (BASE_PATH + '/data/CUB_200_2011/').replace("\\", '/')

                        loader = load_data_scouter(data_dir=data_dir, num_classes=num_classes, batch_size=batch_size,
                                                   train=False, normalize=True)
                    elif 'vit' in m.lower():
                        data_dir = (BASE_PATH + '/data/CUB_200_2011/').replace("\\", '/')
                        loader = load_data_vit(data_dir=data_dir, num_classes=200, batch_size=batch_size,
                                               img_size=img_size, normalize=True)
                    elif 'hybrid' in m.lower():
                        data_dir = (BASE_PATH + '/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl') \
                            .replace("\\", '/')
                        """loader = load_data_hybrid([data_dir], use_attr=False, no_img=False, batch_size=batch_size,
                                                  image_dir='images', img_size=img_size, normalize=True)"""

                correct = 0
                with torch.no_grad():
                    for _, data in enumerate(loader):
                        # data: [img_as_tensor, class_number_as_tensor, ggf. attribute as list]
                        inputs, labels = data
                        input_adv = torch.autograd.Variable(inputs).cuda()

                        # run pipeline of prediction of classes
                        pred = model(input_adv)
                        pred = (pred.max(1, keepdim=True)[1]).detach().cpu()

                        # choose top 1 prediciton
                        correctly_predicted = torch.eq(pred.squeeze(), labels).numpy()
                        out, counts = np.unique(correctly_predicted, return_counts=True)

                        # check how many predictions of a batch are correct
                        if np.where(out == True)[0].size > 0:
                            correct += counts[np.where(out == True)[0][0]]

                # Calculate final accuracy
                final_acc = correct / float(len(loader.dataset))
                print("Test Accuracy = {} / {} = {}".format(correct, len(loader.dataset), final_acc))
                dataset_accs.append(final_acc)
            seed_accs.append(dataset_accs)
        accuracies.append(seed_accs)

    return accuracies


def test_travelling_birds():
    seeds = [1, 2, 3]
    model_names = ['standard', 'sequential', 'independent']
    accs = travellingBirds_test(model_names=model_names, seeds=[1, 2, 3], num_examples=None, batch_size=64)

    data = {}
    data['models'] = ['standard', 'sequential', 'independent']
    data['seeds'] = seeds
    data['datasets'] = ['CUB', 'CUB_black', 'CUB_fixed', 'CUB_random']
    data['accuracies'] = accs

    with open('outputs/results_birds.json', 'w') as f:
        json.dump(data, f)


def attack_concepts_CtoY():
    # model_names = ['independent', 'sequential']
    model_names = ['sequential']
    # attacks = ['sparsefool', 'fgsm', 'pgd', 'bim']
    attacks = ['sparsefool']
    eps = [0.0, 0.001, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    seeds = [1, 2, 3]

    for use_sigmoid in [True, False]:
        # model_accuracies (no iwp/rfiwp included), initially_wrong_predictions, right_from_initially_wrong_predictions
        accs, iwp, rfiwp = attack_concepts(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks,
                                           use_sigmoid=use_sigmoid, num_examples=10, set='test', batch_size=8)

        data = {}
        data['models'] = model_names
        data['seeds'] = seeds
        data['attacks'] = attacks
        data['epsilon'] = eps
        data['accuracies'] = accs
        data['initially_wrong_predictions'] = iwp
        data['right_from_initially_wrong_predictions'] = rfiwp

        """with open('outputs/tworesults_attack_CtoY_sigmoid_' + str(use_sigmoid) + '.json', 'w') as f:
            json.dump(data, f)"""


def attack_images_XtoCtoY():
    # eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    model_names = ['cb_standard', 'cb_sequential', 'cb_independent']
    # model_names = ['standard']
    # attacks = ['sparsefool']
    attacks = ['fgsm']
    seeds = [1]

    accs = attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=20,
                        dataset_type='test', batch_size=32)

    """data = {}
    data['models'] = model_names
    data['seeds'] = seeds
    data['attacks'] = attacks
    data['epsilon'] = eps
    data['accuracies'] = accs

    with open('outputs/results_attack_XtoCtoY.json', 'w') as f:
        json.dump(data, f)"""


def attack_img_XtoC():
    attacks = ['fgsm']
    eps = [0.01, 0.025]
    seeds = [1]

    accs, confusion_matrix, num_c_wrong, total_num_concepts = attack_image_XtoC(seeds=seeds, eps=eps,
                                                                                attacks=attacks,
                                                                                num_examples=3, set='test',
                                                                                batch_size=8)

    data = {'models': ['cb_concept'], 'total_num_concepts': total_num_concepts, 'seeds': seeds, 'attacks': attacks,
            'epsilon': eps, 'accuracies': accs, 'num_c_wrong': num_c_wrong, 'confusion_matrix': confusion_matrix}

    with open('outputs/results_attack_XtoC_testing.json', 'w') as f:
        json.dump(data, f)


def attack_scouter():
    # eps = [0.0, 0.001, 0.0025]
    eps = [0.0, 0.01]
    model_names = ['scouter25+']
    # model_names = ['cb_standard']
    # attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    attacks = ['fgsm']
    seeds = [1]
    # seeds = [1, 2]

    accs = attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=10,
                        dataset_type='test', batch_size=16)

    """data = {}
    data['models'] = model_names
    data['seeds'] = seeds
    data['attacks'] = attacks
    data['epsilon'] = eps
    data['accuracies'] = accs

    with open('outputs/results_scouter.json', 'w') as f:
        json.dump(data, f)"""


def attack_scouter_variations():
    model_names = ['scouter25+_s1_lv', 'scouter25-_s1_lv', 'scouter25+_s1_lc', 'scouter25-_s1_lc']
    attacks = ['fgsm']
    eps = [0.0]
    seeds = [1, 2, 3]

    accs = attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                        dataset_type='test', batch_size=32)

    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    with open('outputs/scouter_var_accs.json', 'w') as f:
        json.dump(data, f)


def test_travBirds_scouter():
    seeds = [1, 2, 3]
    model_names = ['scouter25+', 'scouter25-']
    accs = travellingBirds_test(model_names=model_names, seeds=seeds, num_examples=None, batch_size=64, set='test')

    data = {'models': model_names, 'seeds': seeds, 'datasets': ['CUB', 'CUB_black', 'CUB_fixed', 'CUB_random'],
            'accuracies': accs}

    with open('outputs/results_birds_scouter.json', 'w') as f:
        json.dump(data, f)


def test_e2e_cub_black_segmentation():
    model_names = ['cub_black_cb_standard']
    attacks = ['sparsefool']
    eps = [0.0, 0.0025, 0.2]
    seeds = [1]

    accs = attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=2,
                        dataset_type='test', batch_size=2, path_to_dataset_diff_from_cub='CUB_black/',
                        use_segmentation=True)


def find_reason():
    # eps = [0.05, 0.075, 0.1, 0.125, 0.15]
    eps = [0.05, 0.075, 0.1, 0.125, 0.15]
    model_names = ['cb_standard']
    attacks = ['fgsm']
    seeds = [1]
    batch_size = 32

    accs, dataset_stats = attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=5,
                                       dataset_type='test', batch_size=batch_size, save_perturbations=True,
                                       use_attr_to_check_concepts=False)

    """data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs,
            'dataset_stats': dataset_stats}

    with open('outputs/find_reason.json', 'w') as f:
        json.dump(data, f)"""


def attack_inception25():
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    model_names = ['inception-v3_25']
    attacks = ['fgsm', 'bim', 'pgd', 'sparsefool']
    seeds = [1, 2, 3]

    accs = attack_image(model_names=model_names, seeds=seeds, eps=eps, attacks=attacks, num_examples=None,
                        dataset_type='test', batch_size=32)

    data = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': accs}

    with open('outputs/results_attack_Inception25.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    print('this is the path: ', BASE_PATH)
    pkl_file_path = 'CUB_200_2011/CUB_processed/class_attr_data_10/'
    # find_reason()

    concept_dict_dir = f'{BASE_PATH}/models/' + model_dirs['HybridConcept-CNN_Loss'] + f'1/best_model.pth'.replace("\\",
                                                                                                                   '/')
    model = torch.nn.Sequential(
        Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        utils.load_model(model_name='HybridConcept', path_to_state_dict=concept_dict_dir)
    )

    # attack_inception25()
    # show_vit_results()
    # show_all_normalized_results()

    # test_segmentation()
    # attack_scouter()

    # show_cb_cub_black_segmentation_results()
    # attack_scouter()
    # test_e2e_cub_black_segmentation()
    # test_travelling_birds()
    # attack_images_XtoCtoY()
    # attack_concepts_CtoY()
    # attack_img_XtoC()
    # attack_scouter_variations()

    # attack_img_XtoC()
    # test_travBirds_scouter()

    # show_cb_XtoC_results()
    """show_cb_CtoY_results()
    show_cb_XtoCtoY_results()
    show_cb_birds_results()
    show_scouter_results()

    # show_scouter_birds_results()

    # show_cub_black_XtoCtoY_results()
    """
