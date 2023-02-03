import os
import json
import random
import argparse
import torch
from torchvision import transforms
from rtpt import RTPT

import numpy as np
from sklearn.metrics import confusion_matrix

import utils
from dataset import load_data_MaskRCNN, load_data_TravBirds_MaskRCNN, load_data_mask_stage2_adv, \
    load_data_mask_stage2_travBirds
from config import BASE_PATH, BIRD_IDX
from analysis import AverageMeter
from model_templates.utils_models import InverseNormalization, preprocess_mask_outputs, FullMaskStage2Model
from model_templates.Mask_RCNN.tv_maskrcnn import maskrcnn_resnet50_fpn
from adapted_torchattacks import FGSM_MASK, BIM_MASK, PGD_MASK, SparseFool_MASK

random.seed(123)
torch.set_num_threads(4)


def save_preprocessed_mask_outputs(outputs, directory):
    map_id_to_img = {}
    with open(f"{BASE_PATH}/data/CUB_200_2011/images.txt") as f:
        for line in f:
            (key, val) = line.split()
            map_id_to_img[int(key)] = val

    transform_to_save_img = transforms.ToPILImage(mode='RGB')
    transform_to_save_mask = transforms.ToPILImage()

    # save different images
    if "stage2_inputs_crop_box" in outputs.keys():
        for i in range(outputs["stage2_inputs_crop_box"].shape[0]):
            img_id = outputs["stage2_img_id_to_save"][i].item()
            # save bb crops
            save_path = f"{directory}/bbcrops/{map_id_to_img[img_id]}".replace('\\', '/')
            os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
            transform_to_save_img(outputs["stage2_inputs_crop_box"][i]).save(save_path, format='JPEG')

            # save segmented seg crops
            save_path = f"{directory}/bbcrops_seg/{map_id_to_img[img_id]}".replace('\\', '/')
            os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
            transform_to_save_img(outputs["stage2_bbcrop_segmented"][i]).save(save_path, format='JPEG')

            # save segmentation edge crops
            save_path = f"{directory}/segbbcrops/{map_id_to_img[img_id]}".replace('\\', '/')
            os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
            transform_to_save_img(outputs["stage2_inputs_crop_segment"][i]).save(save_path, format='JPEG')

            # save segmentation edge crops with segmentation mask applied
            save_path = f"{directory}/segbbcrops_seg/{map_id_to_img[img_id]}".replace('\\', '/')
            os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
            transform_to_save_img(outputs["stage2_inputs_crop_segment_segmented"][i]).save(save_path, format='JPEG')

            # save cropped segmentation masks (masks are applicable only for segbb_crops)
            save_path = f"{directory}/segmasks/{map_id_to_img[img_id]}".replace('\\', '/')
            os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
            transform_to_save_mask(outputs["segmentation_mask"][i]).save(save_path, format='PNG')

    # save images where bird-class is not found after adversarial attack
    for i in range(outputs["cnt_nobirds"]):
        img_id = outputs["no_birds_img_id_to_save"][i].item()
        # save cropped segmentation masks (masks are applicable only for segbb_crops)
        save_path = f"{directory}/nobird/{map_id_to_img[img_id]}".replace('\\', '/')
        os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
        transform_to_save_img(outputs["img_no_birds"][i]).save(save_path, format='JPEG')


def attack_maskrcnn(attacks, eps, args):
    bird_idx = torch.Tensor([BIRD_IDX]).type(torch.LongTensor).cuda()
    inverse_normalization = InverseNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name="attack Mask R-CNN", max_iterations=3*11)
    # Start the RTPT tracking
    rtpt.start()

    """
    segmentation model: 1st step in pipeline 
    num_classes of ms COCO -> birds included -> leave it like that, since the results on CUB are good
    code copied and adapted from torchvision.models (torchvision 0.9.0+cu111)
    """
    mask_model = maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
    mask_model.cuda()

    # use all images - even those, where a bird label was not found -> those will be sent through without modification
    test_data_path = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl'.replace("\\", '/')
    loader = load_data_MaskRCNN(pkl_paths=[test_data_path], use_attr=True, no_img=False, batch_size=args.batch_size,
                                return_img_id=True, add_other_mask=False)

    atk_no_birds = []
    atk_total_loss, atk_cls_loss, atk_box_reg_loss, atk_mask_loss, atk_objectness_loss, atk_rpn_box_reg_loss = \
        [], [], [], [], [], []
    atk_top1_birds, atk_avg_birds, atk_avg_birds_allimgs = [], [], []
    for attack in attacks:
        eps_no_birds = []
        eps_total_loss, eps_cls_loss, eps_box_reg_loss, eps_mask_loss, eps_objectness_loss, eps_rpn_box_reg_loss = \
            [], [], [], [], [], []
        eps_top1_birds, eps_avg_birds, eps_avg_birds_allimgs = [], [], []

        for epsilon in eps:
            rtpt.step()
            no_birds = 0
            cls_loss_meter = AverageMeter()
            box_reg_loss_meter = AverageMeter()
            mask_loss_meter = AverageMeter()
            objectness_loss_meter = AverageMeter()
            rpn_box_reg_loss_meter = AverageMeter()
            top1_bird_score = AverageMeter()
            avg_bird_score_allimgs = AverageMeter()  # score for all images including the ones, where no bird is found
            avg_bird_score = AverageMeter()  # score for images, where birds are detected

            if attack == 'bim':
                steps = min(max(int(min(epsilon * 255 + 4, 1.25 * epsilon * 255)), 1), 7)
                atk = BIM_MASK(model=mask_model, eps=epsilon, alpha=1 / 255, steps=steps)
            elif attack == 'pgd':
                atk = PGD_MASK(model=mask_model, eps=epsilon, alpha=1 / 255, steps=2)
            elif attack == 'fgsm':
                atk = FGSM_MASK(model=mask_model, eps=epsilon)
            elif attack == 'sparsefool':
                atk = SparseFool_MASK(model=mask_model, steps=2, steps_deepfool=2, lam=epsilon * 10, overshoot=0.02,
                                      descision_threshold=args.threshold)

            for idx, data in enumerate(loader):
                inputs, labels, attr_labels, segmentation_label, boundingbox_label, img_id, targets = data
                inputs = inputs.cuda()

                new_targets = [{'boxes': targets['boxes'][i].cuda(),
                                'labels': bird_idx, 'masks': targets['masks'][i].cuda()}
                               for i in range(inputs.shape[0])]

                # one time predicting the orginal image is required to find the padding borders and resized targets
                mask_model.train()
                _, resized_imgs, resized_targets = mask_model(inputs, new_targets)
                # remove image normalization, since mask_rcnn does it by itself and we get the normalized images here
                resized_imgs = inverse_normalization(resized_imgs.tensors)

                perturbed_imgs = atk(resized_imgs, labels=resized_targets)

                # crop/zero out padded borders
                # find crop point at first transformed images:
                _, _, h, w = resized_imgs.shape
                # initial border positions
                x_left, x_right, y_top, y_bot = 0, w - 1, 0, h - 1
                # find cropping positions for padded borders
                constant_value = torch.tensor([0.485, 0.456, 0.406]).cuda()
                noise_threshold = 0.02
                cropped_imgs = []
                zeroed_adv_imgs = []
                for i in range(resized_imgs.shape[0]):
                    for pos in range(100):
                        if resized_imgs[i, :, h // 2, pos].equal(constant_value) or \
                                (resized_imgs[i, :, h // 2, pos].sum() <= noise_threshold):
                            x_left += 1

                        if resized_imgs[i, :, h // 2, w - 1 - pos].equal(constant_value) or \
                                (resized_imgs[i, :, h // 2, w - 1 - pos].sum() <= noise_threshold):
                            x_right -= 1

                        if resized_imgs[i, :, pos, w // 2].equal(constant_value) or \
                                (resized_imgs[i, :, pos, w // 2].sum() <= noise_threshold):
                            y_top += 1

                        if resized_imgs[i, :, h - 1 - pos, w // 2].equal(constant_value) or \
                                (resized_imgs[i, :, h - 1 - pos, w // 2].sum() <= noise_threshold):
                            y_bot -= 1

                    # crop adversarial images -> only for saving!
                    cropped_imgs.append(perturbed_imgs[i, :, y_top:y_bot + 1, x_left:x_right + 1])

                    # zero out the borders of the adversarial image to predict again -> size matches to resized_targets
                    tmp = torch.zeros(perturbed_imgs[i].shape).cuda()
                    tmp[:, y_top:y_bot + 1, x_left:x_right + 1] = 1
                    zeroed_adv_imgs.append(perturbed_imgs[i] * tmp)

                cropped_imgs = torch.stack(cropped_imgs)
                zeroed_adv_imgs = torch.stack(zeroed_adv_imgs)

                # update loss metrics
                mask_model.train()
                if epsilon > 0.:
                    loss_dict, _, _ = mask_model(zeroed_adv_imgs, resized_targets)
                else:
                    loss_dict, _, _ = mask_model(inputs, resized_targets)

                cls_loss_meter.update(loss_dict["loss_classifier"].item(), zeroed_adv_imgs.size(0))
                box_reg_loss_meter.update(loss_dict["loss_box_reg"].item(), zeroed_adv_imgs.size(0))
                mask_loss_meter.update(loss_dict["loss_mask"].item(), zeroed_adv_imgs.size(0))
                objectness_loss_meter.update(loss_dict["loss_objectness"].item(), zeroed_adv_imgs.size(0))
                rpn_box_reg_loss_meter.update(loss_dict["loss_rpn_box_reg"].item(), zeroed_adv_imgs.size(0))

                mask_model.eval()
                with torch.no_grad():
                    # image outputs already come in size 224x224 here -> ready to save
                    preprocessed_outputs = preprocess_mask_outputs(inputs=cropped_imgs, model=mask_model,
                                                                   threshold=args.threshold, img_id=img_id)
                    no_birds += preprocessed_outputs["cnt_nobirds"]

                    # store certainties for bird detections
                    if preprocessed_outputs["bird_avg_certainty"] > 0.:
                        top1_bird_score.update(preprocessed_outputs["bird_top1_certainty"], inputs.size(0))
                        avg_bird_score.update(preprocessed_outputs["bird_avg_certainty"], inputs.size(0))
                    avg_bird_score_allimgs.update(preprocessed_outputs["bird_avg_certainty"], inputs.size(0))
                if args.generate_new_data:
                    save_preprocessed_mask_outputs(
                        preprocessed_outputs, directory=f"{BASE_PATH}/data/adversarial/{attack}/{int(epsilon * 1000)}/")

            # Print out results of current attack & epsilon
            if attack != 'sparsefool':
                print(
                    f"Attack: {attack}\tEpsilon: {epsilon}\tTotal Loss: {sum(loss_dict.values()):.5f}\t"
                    f"Losses: cls {cls_loss_meter.avg:.5f}, box_reg {box_reg_loss_meter.avg:.5f}, mask {mask_loss_meter.avg:.5f}, "
                    f"objectness {objectness_loss_meter.avg:.5f}, rpn_box_reg {rpn_box_reg_loss_meter.avg:.5f}, "
                    f"no_bird_found: {no_birds}, attacked_imgs: {len(loader.dataset)}")
            else:
                print(f"Attack: {attack}\tEpsilon: {epsilon}\tLosses: cls {cls_loss_meter.avg:.5f}, "
                      f"box_reg {box_reg_loss_meter.avg:.5f}, mask {mask_loss_meter.avg}, objectness "
                      f"{objectness_loss_meter.avg:.5f}, rpn_box_reg {rpn_box_reg_loss_meter.avg:.5f}, "
                      f"no_bird_found: {no_birds}, attacked_imgs: {100}")

            eps_no_birds.append(no_birds)
            eps_top1_birds.append(top1_bird_score.avg)
            eps_avg_birds.append(avg_bird_score.avg.item())
            eps_avg_birds_allimgs.append(avg_bird_score_allimgs.avg.item())
            eps_total_loss.append(sum([cls_loss_meter.avg, box_reg_loss_meter.avg, mask_loss_meter.avg,
                                       objectness_loss_meter.avg, rpn_box_reg_loss_meter.avg]))
            eps_cls_loss.append(cls_loss_meter.avg)
            eps_box_reg_loss.append(box_reg_loss_meter.avg)
            eps_mask_loss.append(mask_loss_meter.avg)
            eps_objectness_loss.append(objectness_loss_meter.avg)
            eps_rpn_box_reg_loss.append(rpn_box_reg_loss_meter.avg)

        atk_top1_birds.append(eps_top1_birds)
        atk_avg_birds.append(eps_avg_birds)
        atk_avg_birds_allimgs.append(eps_avg_birds_allimgs)
        atk_no_birds.append(eps_no_birds)
        atk_total_loss.append(eps_total_loss)
        atk_cls_loss.append(eps_cls_loss)
        atk_box_reg_loss.append(eps_box_reg_loss)
        atk_mask_loss.append(eps_mask_loss)
        atk_objectness_loss.append(eps_objectness_loss)
        atk_rpn_box_reg_loss.append(eps_rpn_box_reg_loss)

    loss_dict = {"loss_classifier": atk_cls_loss, "loss_box_reg": atk_box_reg_loss, "loss_mask": atk_mask_loss,
                 "loss_objectness": atk_objectness_loss, "loss_rpn_box_reg": atk_rpn_box_reg_loss}

    bird_scores = {"top1_birds": atk_top1_birds, "avg_birds_detected": atk_avg_birds,
                   "avg_birds_allimgs": atk_avg_birds_allimgs}

    results = {'models': 'Mask R-CNN', 'attacks': attacks, 'epsilon': eps, 'no_bird_pred': atk_no_birds,
               'total_loss': atk_total_loss, 'indiv_losses': loss_dict, "bird_scores": bird_scores}

    try:
        with open(f'{BASE_PATH}/outputs/results_MaskRCNN_attacks.json', 'w') as f:
            json.dump(results, f)
    except:
        with open(f'{BASE_PATH}/outputs/results_MaskRCNN_attacks.txt', 'w') as f:
            print(results, file=f)


def travBirds_maskrcnn(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name="travBirds Mask R-CNN", max_iterations=3)
    # Start the RTPT tracking
    rtpt.start()

    birds_sets = ['CUB', 'CUB_black', 'CUB_fixed/test', 'CUB_random']
    """
    segmentation model: 1st step in pipeline 
    num_classes of ms COCO -> birds included -> leave it like that, since the results on CUB are good
    code copied and adapted from torchvision.models (torchvision 0.9.0+cu111)
    """
    mask_model = maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
    mask_model.cuda()
    mask_model.eval()

    dataset_no_birds = []
    dataset_top1_birds, dataset_avg_birds, dataset_avg_birds_allimgs = [], [], []

    # use all images - even those, where a bird label was not found -> those will be sent through without modification
    for birds_set in birds_sets:
        rtpt.step()
        test_data_path = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl'.replace("\\", '/')
        if birds_set == 'CUB':
            loader = load_data_MaskRCNN(pkl_paths=[test_data_path], use_attr=True, no_img=False,
                                        batch_size=args.batch_size, return_img_id=True, add_other_mask=False)
        else:
            loader = load_data_TravBirds_MaskRCNN(pkl_paths=[test_data_path], batch_size=args.batch_size,
                                                  birds_dir=f"AdversarialData/{birds_set}")

        no_birds = 0
        top1_bird_score = AverageMeter()
        avg_bird_score_allimgs = AverageMeter()  # score for all images including the ones, where no bird is found
        avg_bird_score = AverageMeter()  # score for images, where birds are detected

        with torch.no_grad():
            for idx, data in enumerate(loader):
                if birds_set == 'CUB':
                    inputs, _, _, _, _, img_id, _ = data
                else:
                    inputs, img_id = data
                inputs = inputs.cuda()

                # image outputs already come in size 224x224 here -> ready to save
                preprocessed_outputs = preprocess_mask_outputs(inputs=inputs, model=mask_model,
                                                               threshold=args.threshold, img_id=img_id)
                no_birds += preprocessed_outputs["cnt_nobirds"]

                # store certainties for bird detections
                if preprocessed_outputs["bird_avg_certainty"] > 0.:
                    top1_bird_score.update(preprocessed_outputs["bird_top1_certainty"], inputs.size(0))
                    avg_bird_score.update(preprocessed_outputs["bird_avg_certainty"], inputs.size(0))
                avg_bird_score_allimgs.update(preprocessed_outputs["bird_avg_certainty"], inputs.size(0))

                # save generated images
                if args.generate_new_data:
                    save_preprocessed_mask_outputs(
                        preprocessed_outputs, directory=f"{BASE_PATH}/data/travBirds/{birds_set}/")

        # Print out results of current attack & epsilon
        print(f"Dataset: {birds_set}\tno_bird_found: {no_birds}")

        dataset_no_birds.append(no_birds)
        dataset_top1_birds.append(top1_bird_score.avg)
        dataset_avg_birds.append(avg_bird_score.avg.item())
        dataset_avg_birds_allimgs.append(avg_bird_score_allimgs.avg.item())

    results = {'models': 'Mask R-CNN', 'birds_sets': birds_sets, 'no_bird_pred': dataset_no_birds,
               "top1_birds": dataset_top1_birds, "avg_birds_detected": dataset_avg_birds,
               "avg_birds_allimgs": dataset_avg_birds_allimgs}

    try:
        with open(f'{BASE_PATH}/outputs/results_MaskRCNN_travBirds_CUB.json', 'w') as f:
            json.dump(results, f)
    except:
        with open(f'{BASE_PATH}/outputs/results_MaskRCNN_travBirds_CUB.txt', 'w') as f:
            print(results, file=f)


def predict_adversarial_stage2(model_names, args, use_seg=''):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=f"adv_stage2{use_seg}", max_iterations=len(model_names)*3*4)
    # Start the RTPT tracking
    rtpt.start()

    seeds = [1, 2, 3]
    attacks = ['fgsm', 'bim', 'pgd']
    eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]

    model_acc, model_bf_acc, model_nobf_acc = [], [], []  # store accuracies
    model_conf_matrix, model_bf_cm, model_nobf_cm = [], [], []  # store confusion matrices
    model_correct, model_correct_bf, model_correct_nobf, model_total_bf = [], [], [], []  # store counts
    for m in model_names:
        seed_acc, seed_bf_acc, seed_nobf_acc = [], [], []  # store accuracies
        seed_conf_matrix, seed_bf_cm, seed_nobf_cm = [], [], []  # store confusion matrices
        seed_correct, seed_correct_bf, seed_correct_nobf, seed_total_bf = [], [], [], []  # store counts
        for seed in seeds:
            # initialize model
            if 'independent' in m.lower():
                concept_path_name = '_'.join(m.split('_')[1:]).replace('Independent', 'Concept')
                concept_name = m.replace('Independent', 'Concept')
                end_name = 'Independent'
            else:  # Sequential
                concept_path_name = '_'.join(m.split('_')[1:]).replace('Sequential', 'Concept')
                concept_name = m.replace('Sequential', 'Concept')
                end_name = '_'.join(m.split('_')[1:])

            concept_dict_dir = \
                f"{BASE_PATH}/models/MaskBottleneck/{concept_path_name}/Seed{seed}/best_model.pth".replace("\\", '/')
            end_dict_dir = \
                f'{BASE_PATH}/models/MaskBottleneck/{end_name}/Seed{seed}/best_model.pth'.replace("\\", '/')

            model = FullMaskStage2Model(
                concept_model=utils.load_model(model_name=concept_name, path_to_state_dict=concept_dict_dir),
                end_model=utils.load_model(model_name=m, path_to_state_dict=end_dict_dir),
                use_sigmoid=True, return_attributes=True
            )
            model.cuda()
            model.eval()

            atk_acc, atk_bf_acc, atk_nobf_acc = [], [], []
            atk_conf_matrix, atk_bf_cm, atk_nobf_cm = [], [], []  # store confusion matrices
            atk_correct, atk_correct_bf, atk_correct_nobf, atk_total_bf = [], [], [], []  # store counts
            for attack in attacks:
                rtpt.step()
                print(f'model: {m}, seed: {seed}, attack: {attack}')
                eps_acc, eps_bf_acc, eps_nobf_acc = [], [], []  # store accuracies
                eps_conf_matrix, eps_bf_cm, eps_nobf_cm = [], [], []  # store confusion matrices
                eps_correct, eps_correct_bf, eps_correct_nobf, eps_total_bf = [], [], [], []  # store counts
                for epsilon in eps:
                    correct = 0
                    correct_bird_found = 0
                    correct_nobird_found = 0
                    total_birds_found = 0
                    c_matrix = []  # list of confusion matrices
                    birds_found_cm = []
                    nobirds_found_cm = []

                    # use the first seed for the args file: except for seed related arguments, all are equal
                    # use info from concept model: independent version does not contain the required information
                    concept_path_name = '_'.join(m.split('_')[1:]).replace('Independent', 'Concept')
                    args_dict = \
                        utils.load_model_args_as_dict(
                            f'{BASE_PATH}/models/MaskBottleneck/{concept_path_name}/Seed1/args.yaml')
                    data_dir = \
                        f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl'.replace("\\", '/')

                    loader = load_data_mask_stage2_adv([data_dir], attack=attack, epsilon=str(int(epsilon * 1000)),
                                                       adv_dataset_dir='adversarial', batch_size=args.batch_size,
                                                       crop_type=args_dict['crop_type'],
                                                       apply_segmentation=args_dict['apply_segmentation'])

                    with torch.no_grad():
                        for idx, data in enumerate(loader):
                            inputs, class_labels, attr_labels, birds_found = data
                            total_birds_found += torch.sum(birds_found).item()
                            current_birds_found = torch.sum(birds_found).item()

                            inputs = inputs.cuda()

                            # get the newly perturbed/predicted class
                            adv_pred, pred_concepts = model(inputs)
                            new_pred = (adv_pred.max(1, keepdim=True)[1]).detach().cpu()

                            if args.store_confusion_matrices:
                                # determine confusion matrix of attribute predictions and attribute labels
                                out_binary = torch.where(pred_concepts.cpu() >= 0.5, torch.tensor(1), torch.tensor(0))
                                attr_birds_found = birds_found.unsqueeze(1).expand(attr_labels.shape)
                                cm = []
                                for i in range(pred_concepts.shape[0]):
                                    tn, fp, fn, tp = confusion_matrix(attr_labels.detach().numpy()[i],
                                                                      out_binary.detach().numpy()[i]).ravel()
                                    cm.append([int(tn), int(fp), int(fn), int(tp)])
                                c_matrix.append(cm)

                                # tn are completely false, but fp, fn, tp should be correct (calculate tn from them)
                                bf_cm = []
                                nobf_cm = []
                                bf_attr_labels = (attr_labels * attr_birds_found).detach().numpy()
                                bf_out_binary = (out_binary * attr_birds_found).detach().numpy()
                                nobf_attr_labels = (attr_labels * ~attr_birds_found).detach().numpy()
                                nobf_out_binary = (out_binary * ~attr_birds_found).detach().numpy()
                                for i in range(pred_concepts.shape[0]):
                                    if birds_found[i].item():
                                        tn, fp, fn, tp = confusion_matrix(bf_attr_labels[i], bf_out_binary[i]).ravel()
                                        bf_cm.append([int(tn), int(fp), int(fn), int(tp)])
                                    else:
                                        tn, fp, fn, tp = \
                                            confusion_matrix(nobf_attr_labels[i], nobf_out_binary[i]).ravel()
                                        nobf_cm.append([int(tn), int(fp), int(fn), int(tp)])
                                birds_found_cm.append(bf_cm)
                                nobirds_found_cm.append(nobf_cm)

                            # Check for successful class predictions
                            correctly_predicted_labels = torch.eq(new_pred.squeeze(), class_labels).numpy()
                            out, counts = np.unique(correctly_predicted_labels, return_counts=True)

                            # correct for all images (birds found & not found)
                            if np.where(out == True)[0].size > 0:
                                correct += counts[np.where(out == True)[0][0]]

                            if current_birds_found > 0:
                                # number of correct prediction for birds found from mask rcnn:
                                out, counts = np.unique(correctly_predicted_labels * birds_found.numpy(),
                                                        return_counts=True)
                                if np.where(out == True)[0].size > 0:
                                    correct_bird_found += counts[np.where(out == True)[0][0]]

                            if pred_concepts.shape[0] - current_birds_found > 0:
                                # number of correct prediction for nobird found from mask rcnn:
                                out, counts = np.unique(correctly_predicted_labels * ~birds_found.numpy(),
                                                        return_counts=True)
                                if np.where(out == True)[0].size > 0:
                                    correct_nobird_found += counts[np.where(out == True)[0][0]]

                    # Calculate final accuracy for this epsilon
                    all_img_acc = correct / float(len(loader.dataset))
                    birds_found_acc = correct_bird_found / float(total_birds_found)
                    if len(loader.dataset) != total_birds_found:
                        nobirds_found_acc = correct_nobird_found / float(len(loader.dataset) - total_birds_found)
                    else:
                        nobirds_found_acc = 1.

                    # Print out results of current attack & epsilon
                    print(f"Attack: {attack}\tEpsilon: {epsilon}\tall imgs acc: {all_img_acc:.5f}, "
                          f"birds found acc {birds_found_acc:.5f}, nobirds found acc: {nobirds_found_acc} "
                          f"total birds found: {total_birds_found}, num_imgs total: {len(loader.dataset)}")

                    eps_acc.append(all_img_acc)
                    eps_bf_acc.append(birds_found_acc)
                    eps_nobf_acc.append(nobirds_found_acc)

                    eps_correct.append(correct)
                    eps_correct_bf.append(correct_bird_found)
                    eps_correct_nobf.append(correct_nobird_found)
                    eps_total_bf.append(total_birds_found)

                    eps_conf_matrix.append(c_matrix)
                    eps_bf_cm.append(birds_found_cm)
                    eps_nobf_cm.append(nobirds_found_cm)

                atk_acc.append(eps_acc)
                atk_bf_acc.append(eps_bf_acc)
                atk_nobf_acc.append(eps_nobf_acc)

                atk_correct.append(eps_correct)
                atk_correct_bf.append(eps_correct_bf)
                atk_correct_nobf.append(eps_correct_nobf)
                atk_total_bf.append(eps_total_bf)

                atk_conf_matrix.append(eps_conf_matrix)
                atk_bf_cm.append(eps_bf_cm)
                atk_nobf_cm.append(eps_nobf_cm)

            seed_acc.append(atk_acc)
            seed_bf_acc.append(atk_bf_acc)
            seed_nobf_acc.append(atk_nobf_acc)

            seed_correct.append(atk_correct)
            seed_correct_bf.append(atk_correct_bf)
            seed_correct_nobf.append(atk_correct_nobf)
            seed_total_bf.append(atk_total_bf)

            seed_conf_matrix.append(atk_conf_matrix)
            seed_bf_cm.append(atk_bf_cm)
            seed_nobf_cm.append(atk_nobf_cm)

        model_acc.append(seed_acc)
        model_bf_acc.append(seed_bf_acc)
        model_nobf_acc.append(seed_nobf_acc)

        model_correct.append(seed_correct)
        model_correct_bf.append(seed_correct_bf)
        model_correct_nobf.append(seed_correct_nobf)
        model_total_bf.append(seed_total_bf)

        model_conf_matrix.append(seed_conf_matrix)
        model_bf_cm.append(seed_bf_cm)
        model_nobf_cm.append(seed_nobf_cm)

    results = {'models': model_names, 'seeds': seeds, 'attacks': attacks, 'epsilon': eps, 'accuracies': model_acc,
               'bird_found_acc': model_bf_acc, 'nobird_found_acc': model_nobf_acc, 'num_pred_correct': model_correct,
               'bird_found_num_pred_correct': model_correct_bf, 'nobird_found_num_pred_correct': model_correct_nobf,
               'total_birds_found': model_total_bf, 'concept_conf_matrix': model_conf_matrix,
               'bf_concept_conf_matrix': model_bf_cm, 'nobf_concept_conf_matrix': model_nobf_cm}

    os.makedirs(f'{BASE_PATH}/outputs/', exist_ok=True)
    try:
        with open(f'{BASE_PATH}/outputs/results_MaskRCNN_stage2_advpreds{use_seg}.json', 'w') as f:
            json.dump(results, f)
    except Exception as e:
        print(e)
        with open(f'{BASE_PATH}/outputs/results_MaskRCNN_stage2_advpreds{use_seg}.txt', 'w') as f:
            print(results, file=f)


def predict_birds_stage2(model_names, args, use_seg=''):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name=f"birds_stage2{use_seg}", max_iterations=len(model_names) * 3 * 4)
    # Start the RTPT tracking
    rtpt.start()

    seeds = [1, 2, 3]

    datasets = ['CUB_Cropped', 'travBirds/CUB_black', 'travBirds/CUB_fixed/test', 'travBirds/CUB_random']

    model_acc, model_bf_acc, model_nobf_acc = [], [], []  # store accuracies
    model_conf_matrix, model_bf_cm, model_nobf_cm = [], [], []  # store confusion matrices
    model_correct, model_correct_bf, model_correct_nobf, model_total_bf = [], [], [], []  # store counts
    for m in model_names:
        seed_acc, seed_bf_acc, seed_nobf_acc = [], [], []  # store accuracies
        seed_conf_matrix, seed_bf_cm, seed_nobf_cm = [], [], []  # store confusion matrices
        seed_correct, seed_correct_bf, seed_correct_nobf, seed_total_bf = [], [], [], []  # store counts
        for seed in seeds:
            # initialize model
            if 'independent' in m.lower():
                concept_path_name = '_'.join(m.split('_')[1:]).replace('Independent', 'Concept')
                concept_name = m.replace('Independent', 'Concept')
                end_name = 'Independent'
            else:  # Sequential
                concept_path_name = '_'.join(m.split('_')[1:]).replace('Sequential', 'Concept')
                concept_name = m.replace('Sequential', 'Concept')
                end_name = '_'.join(m.split('_')[1:])

            concept_dict_dir = \
                f"{BASE_PATH}/models/MaskBottleneck/{concept_path_name}/Seed{seed}/best_model.pth".replace("\\", '/')
            end_dict_dir = \
                f'{BASE_PATH}/models/MaskBottleneck/{end_name}/Seed{seed}/best_model.pth'.replace("\\", '/')

            model = FullMaskStage2Model(
                concept_model=utils.load_model(model_name=concept_name, path_to_state_dict=concept_dict_dir),
                end_model=utils.load_model(model_name=m, path_to_state_dict=end_dict_dir),
                use_sigmoid=True, return_attributes=True
            )
            model.cuda()
            model.eval()

            dataset_acc, dataset_bf_acc, dataset_nobf_acc = [], [], []  # store accuracies
            dataset_conf_matrix, dataset_bf_cm, dataset_nobf_cm = [], [], []  # store confusion matrices
            dataset_correct, dataset_correct_bf, dataset_correct_nobf, dataset_total_bf = [], [], [], []  # store counts
            for dataset in datasets:
                rtpt.step()
                print(f'model: {m}, seed: {seed}, dataset: {dataset}')
                correct = 0
                correct_bird_found = 0
                correct_nobird_found = 0
                total_birds_found = 0
                c_matrix = []  # list of confusion matrices
                birds_found_cm = []
                nobirds_found_cm = []

                # use the first seed for the args file: except for seed related arguments, all are equal
                # use info from concept model: independent version does not contain the required information
                concept_path_name = '_'.join(m.split('_')[1:]).replace('Independent', 'Concept')
                args_dict = \
                    utils.load_model_args_as_dict(
                        f'{BASE_PATH}/models/MaskBottleneck/{concept_path_name}/Seed1/args.yaml')
                if 'CUB_Cropped' in dataset:
                    data_dir = \
                        f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/masked_attr_data/test.pkl'.replace("\\", '/')
                else:
                    data_dir = \
                        f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl'.replace("\\", '/')
                loader = load_data_mask_stage2_travBirds([data_dir], adv_dataset_dir=dataset,
                                                         batch_size=args.batch_size, crop_type=args_dict['crop_type'],
                                                         apply_segmentation=args_dict['apply_segmentation'])

                with torch.no_grad():
                    for idx, data in enumerate(loader):
                        inputs, class_labels, attr_labels, birds_found = data
                        total_birds_found += torch.sum(birds_found).item()
                        current_birds_found = torch.sum(birds_found).item()
                        inputs = inputs.cuda()

                        # get the newly perturbed/predicted class
                        adv_pred, pred_concepts = model(inputs)
                        new_pred = (adv_pred.max(1, keepdim=True)[1]).detach().cpu()

                        if args.store_confusion_matrices:
                            # determine confusion matrix of attribute predictions and attribute labels
                            out_binary = torch.where(pred_concepts.cpu() >= 0.5, torch.tensor(1), torch.tensor(0))
                            attr_birds_found = birds_found.unsqueeze(1).expand(attr_labels.shape)
                            cm = []
                            for i in range(pred_concepts.shape[0]):
                                tn, fp, fn, tp = confusion_matrix(attr_labels.detach().numpy()[i],
                                                                  out_binary.detach().numpy()[i]).ravel()
                                cm.append([int(tn), int(fp), int(fn), int(tp)])
                            c_matrix.append(cm)

                            # tn are completely false, but fp, fn, tp should be correct (calculate tn from them)
                            bf_cm = []
                            nobf_cm = []
                            bf_attr_labels = (attr_labels * attr_birds_found).detach().numpy()
                            bf_out_binary = (out_binary * attr_birds_found).detach().numpy()
                            nobf_attr_labels = (attr_labels * ~attr_birds_found).detach().numpy()
                            nobf_out_binary = (out_binary * ~attr_birds_found).detach().numpy()
                            for i in range(pred_concepts.shape[0]):
                                if birds_found[i].item():
                                    tn, fp, fn, tp = confusion_matrix(bf_attr_labels[i], bf_out_binary[i]).ravel()
                                    bf_cm.append([int(tn), int(fp), int(fn), int(tp)])
                                else:
                                    tn, fp, fn, tp = confusion_matrix(nobf_attr_labels[i], nobf_out_binary[i]).ravel()
                                    nobf_cm.append([int(tn), int(fp), int(fn), int(tp)])
                            birds_found_cm.append(bf_cm)
                            nobirds_found_cm.append(nobf_cm)

                        # Check for successful class predictions
                        correctly_predicted_labels = torch.eq(new_pred.squeeze(), class_labels).numpy()
                        out, counts = np.unique(correctly_predicted_labels, return_counts=True)

                        # correct for all images (birds found & not found)
                        if np.where(out == True)[0].size > 0:
                            correct += counts[np.where(out == True)[0][0]]

                        if current_birds_found > 0:
                            # number of correct prediction for birds found from mask rcnn:
                            out, counts = np.unique(correctly_predicted_labels * birds_found.numpy(),
                                                    return_counts=True)
                            if np.where(out == True)[0].size > 0:
                                correct_bird_found += counts[np.where(out == True)[0][0]]

                        if pred_concepts.shape[0] - current_birds_found > 0:
                            # number of correct prediction for nobird found from mask rcnn:
                            out, counts = np.unique(correctly_predicted_labels * ~birds_found.numpy(),
                                                    return_counts=True)
                            if np.where(out == True)[0].size > 0:
                                correct_nobird_found += counts[np.where(out == True)[0][0]]

                # Calculate final accuracy for this epsilon
                all_img_acc = correct / float(len(loader.dataset))
                birds_found_acc = correct_bird_found / float(total_birds_found)
                if len(loader.dataset) != total_birds_found:
                    nobirds_found_acc = correct_nobird_found / float(len(loader.dataset) - total_birds_found)
                else:
                    nobirds_found_acc = 1.

                # Print out results of current attack & epsilon
                print(f"Dataset: {dataset}\tall imgs acc: {all_img_acc:.5f}, "
                      f"birds found acc {birds_found_acc:.5f}, nobirds found acc: {nobirds_found_acc} "
                      f"total birds found: {total_birds_found}, num_imgs total: {len(loader.dataset)}")

                dataset_acc.append(all_img_acc)
                dataset_bf_acc.append(birds_found_acc)
                dataset_nobf_acc.append(nobirds_found_acc)

                dataset_correct.append(correct)
                dataset_correct_bf.append(correct_bird_found)
                dataset_correct_nobf.append(correct_nobird_found)
                dataset_total_bf.append(total_birds_found)

                dataset_conf_matrix.append(c_matrix)
                dataset_bf_cm.append(birds_found_cm)
                dataset_nobf_cm.append(nobirds_found_cm)

            seed_acc.append(dataset_acc)
            seed_bf_acc.append(dataset_bf_acc)
            seed_nobf_acc.append(dataset_nobf_acc)

            seed_correct.append(dataset_correct)
            seed_correct_bf.append(dataset_correct_bf)
            seed_correct_nobf.append(dataset_correct_nobf)
            seed_total_bf.append(dataset_total_bf)

            seed_conf_matrix.append(dataset_conf_matrix)
            seed_bf_cm.append(dataset_bf_cm)
            seed_nobf_cm.append(dataset_nobf_cm)

        model_acc.append(seed_acc)
        model_bf_acc.append(seed_bf_acc)
        model_nobf_acc.append(seed_nobf_acc)

        model_correct.append(seed_correct)
        model_correct_bf.append(seed_correct_bf)
        model_correct_nobf.append(seed_correct_nobf)
        model_total_bf.append(seed_total_bf)

        model_conf_matrix.append(seed_conf_matrix)
        model_bf_cm.append(seed_bf_cm)
        model_nobf_cm.append(seed_nobf_cm)

    results = {'models': model_names, 'seeds': seeds, 'datasets': datasets, 'accuracies': model_acc,
               'bird_found_acc': model_bf_acc, 'nobird_found_acc': model_nobf_acc, 'num_pred_correct': model_correct,
               'bird_found_num_pred_correct': model_correct_bf, 'nobird_found_num_pred_correct': model_correct_nobf,
               'total_birds_found': model_total_bf, 'concept_conf_matrix': model_conf_matrix,
               'bf_concept_conf_matrix': model_bf_cm, 'nobf_concept_conf_matrix': model_nobf_cm}

    os.makedirs(f'{BASE_PATH}/outputs/', exist_ok=True)
    try:
        with open(f'{BASE_PATH}/outputs/results_MaskRCNN_stage2_travBirds_nocm_pred{use_seg}.json', 'w') as f:
            json.dump(results, f)
    except Exception as e:
        print(e)
        with open(f'{BASE_PATH}/outputs/results_MaskRCNN_stage2_travBirds_nocm_pred{use_seg}.txt', 'w') as f:
            print(results, file=f)


if __name__ == '__main__':
    # https://github.com/pytorch/vision/blob/main/references/detection/engine.py
    # https://bjornkhansen95.medium.com/mask-r-cnn-for-segmentation-using-pytorch-8bbfa8511883

    parser = argparse.ArgumentParser('Workstation Attacks', add_help=False)
    # self defined
    parser.add_argument('--task', default='', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('-threshold', type=float, default=0.3,
                        help='threshold for score/certainty about segmentation class')
    parser.add_argument('-crop_type', type=str, default='bb',
                        help='type of image cropping applied after pass trough Mask-RCNN (if not labelbb chosen, '
                             'labelbb are annotations from cub), Choose: [labelbb, segbb, bb] '
                             '- for Concept & Sequential Models')
    parser.add_argument('-apply_segmentation', action='store_true',
                        help='Wheter to apply a segmentation mask to the cropped image, only works for crop_types: '
                             '[labelbb, cropbb, segbb]')
    parser.add_argument('-generate_new_data', action='store_true', help='Use, when being interested in generating a new'
                                                                        'dataset for the second stage of the pipeline')
    parser.add_argument('-store_confusion_matrices', action='store_true', help='Use, when being interested in storing '
                                                                               'the confusion matrices of the concepts '
                                                                               'predicted. This will inflate the size '
                                                                               'of the files by a lot')
    args = parser.parse_args()

    if args.task == 'attack_maskrcnn':
        attacks = ['fgsm', 'bim', 'pgd']
        eps = [0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
        attack_maskrcnn(attacks, eps, args)
    elif args.task == 'travBirds_maskrcnn':
        travBirds_maskrcnn(args)
    elif args.task == 'predict_adversarial_stage2_noseg':
        model_names = ['Mask_Independent_cropbb', 'Mask_Independent_segbb', 'Mask_Sequential_cropbb',
                       'Mask_Sequential_segbb']
        predict_adversarial_stage2(model_names, args, use_seg='')
    elif args.task == 'predict_adversarial_stage2_useseg':
        model_names_black = ['Mask_Independent_cropbb_useseg', 'Mask_Independent_segbb_useseg',
                             'Mask_Sequential_cropbb_useseg', 'Mask_Sequential_segbb_useseg']
        predict_adversarial_stage2(model_names_black, args, use_seg='_useseg')
    elif args.task == 'predict_birds_stage2_noseg':
        model_names = ['Mask_Independent_cropbb', 'Mask_Independent_segbb', 'Mask_Sequential_cropbb',
                       'Mask_Sequential_segbb']
        predict_birds_stage2(model_names, args, use_seg='')
    elif args.task == 'predict_birds_stage2_useseg':
        model_names_black = ['Mask_Independent_cropbb_useseg', 'Mask_Independent_segbb_useseg',
                             'Mask_Sequential_cropbb_useseg', 'Mask_Sequential_segbb_useseg']
        predict_birds_stage2(model_names_black, args, use_seg='_useseg')





