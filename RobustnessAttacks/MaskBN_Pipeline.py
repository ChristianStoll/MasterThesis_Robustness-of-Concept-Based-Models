import argparse
import os
import csv
from config import BASE_PATH
from config import COCO_INSTANCE_CATEGORY_NAMES as coco_names
import cv2
import torch
import random
import numpy as np
import torchvision.models as tvmodels
from torchvision.transforms import transforms
from torchvision.ops import box_iou
from dataset import load_data_MaskRCNN
import matplotlib.pyplot as plt
from torchmetrics import IoU
import utils
from analysis import AverageMeter
from rtpt import RTPT

from model_templates.utils_models import preprocess_mask_outputs
from model_templates.Mask_RCNN.tv_maskrcnn import maskrcnn_resnet50_fpn

torch.set_num_threads(4)

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def save_list_of_lists(list_of_lists, filename):
    os.makedirs('/'.join(filename.split('/')[:-1]), exist_ok=True)
    with open(filename, "w") as f:
        wr = csv.writer(f)
        wr.writerows(list_of_lists)


def show_segmentation_example():
    # num_classes of ms COCO -> birds included
    model = tvmodels.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
    model.cuda()
    model.eval()

    train_data_path = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/train.pkl'.replace("\\", '/')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')

    print(train_data_path)
    print(val_data_path)

    loader = load_data_MaskRCNN(pkl_paths=[val_data_path], use_attr=True, no_img=False, batch_size=1)

    for idx, data in enumerate(loader):
        inputs, labels, attr_labels, segmentation, boundingbox = data
        inputs = inputs.float().cuda()
        segmentation = segmentation.float()

        # outputs = model(inputs)

        masks, boxes, labels = get_outputs(inputs, model, threshold=0.1)
        print(masks.shape)
        print('labels: ', labels)
        print('seg', segmentation.shape)
        print('mask', masks.shape)
        print('in', inputs.shape)
        print()

        alpha = 1
        beta = 0.6  # transparency for the segmentation map
        gamma = 0  # scalar added to each sum
        # cv2.addWeighted(inputs[0].detach().cpu().numpy().transpose((1, 2, 0), alpha, masks.transpose((1, 2, 0)), beta, gamma, image)

        m = masks.sum(axis=0)
        m = m / np.max(m).astype(float)
        ms = np.stack((m, np.zeros(m.shape), np.zeros(m.shape)), axis=-1)
        print(f'ms: {ms.shape}')

        plt.imshow(inputs[0].detach().cpu().numpy().transpose((1, 2, 0)) * 0.5 + ms * 0.5)
        plt.show()

        # keys: masks, boxes, labels, scores

        """print(type(outputs))
        print(f'shape: {len(outputs)}')
        print([type(x) for x in outputs])
        print(outputs[0].keys())
        print(outputs[0]['masks'].shape)
        print()
        #print(outputs[1].keys())

        print(f"boxes: {outputs[0]['boxes']}")
        print(f"labels: {outputs[0]['labels']}")
        label_names = [coco_names[x] for x in outputs[0]['labels']]
        print(label_names)

        print(f"scores: {outputs[0]['scores']}")
        #print(f"masks: {outputs[0]['masks']}")"""
        if idx > 10:
            break


"""
    The following Code is from:
    https://debuggercafe.com/instance-segmentation-with-pytorch-and-mask-r-cnn/
    reused functions:
    - get_outputs()
    - draw_segmentation_map()
"""
def get_outputs(image, model, threshold=0.5): # threshold for discarding marked areas
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)[0]

    print(outputs.keys())
    # get all the scores
    scores = list(outputs['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs['masks'] > 0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]

    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]

    # get the classes labels
    labels = [coco_names[i] for i in outputs['labels']] # matching of segmentation with coco classes # TODO change?
    return masks, boxes, labels


def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1
    beta = 0.6  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a random color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        print(f'seg_map: {segmentation_map.shape}')
        # convert the original PIL image into NumPy format
        #image = np.transpose(image, (1, 2, 0))
        print('img.shape', image.shape)
        print('masks.shape: ', masks.shape)
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(src1=image, alpha=alpha, src2=segmentation_map, beta=beta, gamma=gamma)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color,
                      thickness=2)
        # put the label text above the objects
        cv2.putText(image, labels[i], (boxes[i][0][0], boxes[i][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    thickness=2, lineType=cv2.LINE_AA)

    return image


def test_pretrained_maskrcnn(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name='Test Mask-RCNN', max_iterations=5*3)
    # Start the RTPT tracking
    rtpt.start()

    # num_classes of ms COCO -> birds included
    model = tvmodels.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
    model.cuda()
    model.eval()

    # output ['boxes', 'labels', 'scores', 'masks']
    results = []
    for threshold in [0.5, 0.3, 0.2, 0.1, 0.0]:
        train_data_path = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/train.pkl'.replace("\\", '/')
        val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
        test_data_path = train_data_path.replace('train.pkl', 'test.pkl')

        for dataset_path in [train_data_path, val_data_path, test_data_path]:
            rtpt.step()
            loader = load_data_MaskRCNN(pkl_paths=[dataset_path], use_attr=True, no_img=False,
                                        batch_size=args.batch_size, resize_imgs=args.resize_imgs)

            iou_mask_meter = AverageMeter()
            iou_bb_meter = AverageMeter()
            tv_box_iou_bb_meter = AverageMeter()
            cnt_nobirds = 0
            for _, data in enumerate(loader):
                inputs, labels, attr_labels, segmentation, boundingbox_label = data
                inputs = inputs.float().cuda()
                segmentation = segmentation.type(torch.IntTensor)
                boundingbox_label = boundingbox_label.type(torch.IntTensor)

                outputs = model(inputs)
                bird_idx = coco_names.index('bird')

                # handle each element in batch
                masks = []
                boxes = []
                for i in range(len(outputs)):
                    """threshold scores to get certainty gt 'threshold', get number of remaining scores 
                    (outputs are sorted by descending scores)"""
                    scores = outputs[i]['scores']
                    threshold_idx = len([utils.index(scores, x) for x in scores if x > threshold])

                    # apply threshold on outputs
                    labels = outputs[i]['labels'][:threshold_idx]
                    bboxes = outputs[i]['boxes'][:threshold_idx]
                    seg_masks = outputs[i]['masks'][:threshold_idx]

                    if bird_idx in labels:
                        # extract bird predictions for masks and bounding_boxes
                        bird_masks = torch.stack([m > 0.5 for label, m in zip(labels, seg_masks) if label == bird_idx])

                        bird_boxes = torch.stack([bb for label, bb in zip(labels, bboxes) if label == bird_idx])
                        # extract most outer coordinates of bounding boxes
                        box = torch.tensor(
                            [bird_boxes[:, 0].min(), bird_boxes[:, 1].min(), bird_boxes[:, 2].max(), bird_boxes[:, 3].max()])
                    else:
                        cnt_nobirds += 1
                        box = torch.zeros_like(outputs[i]['boxes'][0]).cpu()
                        bird_masks = torch.zeros_like(outputs[i]['masks'][0].unsqueeze(0))
                    boxes.append(box)

                    bm = torch.zeros((2, 256, 256))
                    bm[0] = bird_masks.sum(axis=0)
                    masks.append(bm)
                segmentation_masks = torch.stack(masks).cpu()
                bounding_boxes = torch.stack(boxes).cpu()

                # https://torchmetrics.readthedocs.io/en/latest/references/modules.html#iou
                # the IoU loss needs to be sent to cuda manually
                iou = IoU(num_classes=2)
                iou_mask_meter.update(iou(segmentation_masks, segmentation).item(), segmentation_masks.shape[0])

                # calculation of IoU for bounding boxes having 4 coordinates
                iou_bb_meter.update(utils.jaccard(boundingbox_label, bounding_boxes).mean().item(), bounding_boxes.shape[0])
            print('Mask IOU: ', iou_mask_meter.avg)
            print('BB IOU: ', iou_bb_meter.avg)

            results.append(f"Threshold: {threshold}\tDataset-Type: {dataset_path.split('/')[-1].split('.')[0]}\tMask "
                           f"IoU: {iou_mask_meter.avg}\tBoundingBox IoU: {iou_bb_meter.avg}\tno birds in image predicted: {cnt_nobirds}")

    with open(f"{BASE_PATH}/outputs/Pretrained_MaskRCNN_results_no_resize.txt", 'w') as file:
        file.write('\n'.join(results))
        file.close()


def save_segmented_imgs(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name='Test Mask-RCNN', max_iterations=3)
    # Start the RTPT tracking
    rtpt.start()

    transform_to_save_img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToPILImage(mode='RGB')
        ])
    transform_to_save_mask = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToPILImage()
        ])

    map_id_to_img = {}
    with open(f"{BASE_PATH}/data/CUB_200_2011/images.txt") as f:
        for line in f:
            (key, val) = line.split()
            map_id_to_img[int(key)] = val

    # num_classes of ms COCO -> birds included, outputs: dict['boxes', 'labels', 'scores', 'masks']
    model = tvmodels.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
    model.cuda()
    model.eval()

    train_data_path = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/train.pkl'.replace("\\", '/')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    test_data_path = train_data_path.replace('train.pkl', 'test.pkl')

    results = []
    full_dataset_bb_list = []
    full_dataset_segbb_list = []
    for dataset_path in [train_data_path, val_data_path, test_data_path]:
        bird_idx = coco_names.index('bird')
        rtpt.step()
        loader = load_data_MaskRCNN(pkl_paths=[dataset_path], use_attr=True, no_img=False, batch_size=args.batch_size,
                                    return_img_id=True, add_other_mask=True)

        iou_mask_meter = AverageMeter()
        iou_bb_meter = AverageMeter()
        tv_box_iou_bb_meter = AverageMeter()
        cnt_nobirds = 0
        for _, data in enumerate(loader):
            # inputs, labels, attr_labels, segmentation, bounding_box = data
            inputs, labels, attr_labels, segmentation_label, boundingbox_label, img_id = data
            inputs = inputs.float().cuda()

            processed_outputs = preprocess_mask_outputs(inputs, model, args.threshold, img_id)

            model.eval()
            outputs = model(inputs)  # output is list of #b dicts

            # handle each element in batch
            masks = []
            boxes = []
            segmentation_masks = []
            attr_labels_remaining = []

            stage2_inputs_crop_box = []
            stage2_inputs_crop_segment = []
            stage2_segmentation_mask = []
            stage2_bb = []
            stage2_seg_bb = []
            stage2_img_id_to_save = []
            for i in range(len(outputs)):
                """threshold scores to get certainty gt 'threshold', get number of remaining scores 
                (outputs are sorted by descending scores)"""
                scores = outputs[i]['scores']
                threshold_idx = len([utils.index(scores, x) for x in scores if x > args.threshold])

                # apply threshold on outputs
                labels = outputs[i]['labels'][:threshold_idx]
                bboxes = outputs[i]['boxes'][:threshold_idx]
                seg_masks = outputs[i]['masks'][:threshold_idx]

                if bird_idx in labels:
                    # extract bird predictions for masks and bounding_boxes
                    bird_masks = torch.stack([m > 0.5 for label, m in zip(labels, seg_masks) if label == bird_idx])
                    bird_boxes = torch.stack([bb for label, bb in zip(labels, bboxes) if label == bird_idx])
                    # extract most outer coordinates of bounding boxes - required for calculating the IoU
                    bounding_box = torch.tensor(
                        [bird_boxes[:, 0].min(), bird_boxes[:, 1].min(), bird_boxes[:, 2].max(), bird_boxes[:, 3].max()]
                    ).type(torch.IntTensor)

                    # cut out bounding box area of image and segmentation mask
                    stage2_inputs_crop_box.append(inputs[i, :, bounding_box[0].item():bounding_box[2].item() + 1,
                                         bounding_box[1].item():bounding_box[3].item() + 1])

                    # calculate bird mask of img
                    segmentation_masks.append(
                        torch.stack([m > 0.5 for label, m in zip(labels, seg_masks) if label == bird_idx]).sum(axis=0))

                    # crop image around edges of segmentation:
                    # outer coordinates of segmentation
                    seg_idxs = torch.where(segmentation_masks[-1][0] > 0)
                    seg_box = torch.tensor([seg_idxs[0].min(), seg_idxs[1].min(), seg_idxs[0].max(), seg_idxs[1].max()])

                    stage2_inputs_crop_segment.append(inputs[i, :, seg_box[0].item():seg_box[2].item() + 1,
                                         seg_box[1].item():seg_box[3].item() + 1])

                    # crop segmentation mask around edges of segmented bird
                    stage2_segmentation_mask.append(segmentation_masks[-1][0][seg_box[0].item():seg_box[2].item() + 1,
                                                    seg_box[1].item():seg_box[3].item() + 1])

                    # also save bounding boxes and current image id
                    stage2_bb.append(bounding_box)
                    stage2_seg_bb.append(seg_box)
                    stage2_img_id_to_save.append(img_id[i])

                    attr_labels_remaining.append(attr_labels[i])
                else:
                    # nothing to do here
                    cnt_nobirds += 1
                    pass

                # add background mask
                bm = torch.zeros((2, 256, 256))
                bm[0] = bird_masks.sum(axis=0)
                masks.append(bm)

            segmentation_masks = torch.stack(masks).cpu()
            bounding_boxes = torch.stack(stage2_bb).cpu()

            # the IoU loss needs to be sent to cuda manually
            iou = IoU(num_classes=2)
            iou_mask_meter.update(iou(segmentation_masks, segmentation_label.type(torch.IntTensor)).item(),
                                  segmentation_masks.shape[0])

            # calculation of IoU for bounding boxes having 4 coordinates
            iou_bb_meter.update(utils.jaccard(boundingbox_label, bounding_boxes).mean().item(),
                                bounding_boxes.shape[0])
            tv_box_iou_bb_meter.update(box_iou(boundingbox_label, bounding_boxes).mean().item(),
                                       bounding_boxes.shape[0])

            # fit each tensor to the same size (224x224 for resnet34)
            stage2_inputs_crop_box = [transform_to_save_img(x) for x in stage2_inputs_crop_box]
            stage2_inputs_crop_segment = [transform_to_save_img(x) for x in stage2_inputs_crop_segment]
            stage2_segmentation_mask = \
                [transform_to_save_mask(torch.where(x.unsqueeze(0) > 0, 1., 0.)) for x in stage2_segmentation_mask]

            # save segmented, cropped and resized images
            for i in range(len(stage2_img_id_to_save)):
                # save bb crops
                save_path = f"{BASE_PATH}/data/CUB_Cropped/bbcrops/{map_id_to_img[img_id[i].item()]}".replace('\\', '/')
                os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
                stage2_inputs_crop_box[i].save(save_path, format='JPEG')
                # save segmentation edge crops
                save_path = f"{BASE_PATH}/data/CUB_Cropped/segbbcrops/{map_id_to_img[img_id[i].item()]}".replace('\\', '/')
                os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
                stage2_inputs_crop_segment[i].save(save_path, format='JPEG')
                # save cropped segmentation masks
                save_path = f"{BASE_PATH}/data/CUB_Cropped/segmasks/{map_id_to_img[img_id[i].item()]}".replace('\\', '/')
                os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
                stage2_segmentation_mask[i].save(save_path, format='PNG')

                # save bbs and segmentation bbs
                full_dataset_bb_list.append([stage2_img_id_to_save[i].item()] + stage2_bb[i].cpu().tolist())
                full_dataset_segbb_list.append([stage2_img_id_to_save[i].item()] + stage2_seg_bb[i].cpu().tolist())

        print('Mask IOU: ', iou_mask_meter.avg)
        print('BB IOU: ', iou_bb_meter.avg)
        print('TV BB IOU: ', tv_box_iou_bb_meter.avg)

        results.append(f"Threshold: {args.threshold}\tDataset-Type: {dataset_path.split('/')[-1].split('.')[0]}\tMask "
                       f"IoU: {iou_mask_meter.avg}\tBoundingBox IoU: {iou_bb_meter.avg}\t"
                       f"TV Box_IoU: {tv_box_iou_bb_meter.avg}\tno birds in image predicted: {cnt_nobirds}")

    with open(f"{BASE_PATH}/outputs/Pretrained_MaskRCNN_results_crop.txt", 'w') as file:
        file.write('\n'.join(results))
        file.close()

    # save bounding boxes of segmentations and direct bbs from mask-rcnn
    save_list_of_lists(full_dataset_segbb_list, f"{BASE_PATH}/data/CUB_Cropped/segmentation_bbs.txt".replace('\\', '/'))
    save_list_of_lists(full_dataset_bb_list, f"{BASE_PATH}/data/CUB_Cropped/maskrcnn_bbs.txt".replace('\\', '/'))


def save_segmented_imgs_new(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name='Test Mask-RCNN', max_iterations=3)
    # Start the RTPT tracking
    rtpt.start()

    transform_to_save_img = transforms.ToPILImage(mode='RGB')
    transform_to_save_mask = transforms.ToPILImage()

    map_id_to_img = {}
    with open(f"{BASE_PATH}/data/CUB_200_2011/images.txt") as f:
        for line in f:
            (key, val) = line.split()
            map_id_to_img[int(key)] = val

    # num_classes of ms COCO -> birds included, outputs: dict['boxes', 'labels', 'scores', 'masks']
    model = maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
    model.cuda()
    model.eval()

    train_data_path = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/train.pkl'.replace("\\", '/')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    test_data_path = train_data_path.replace('train.pkl', 'test.pkl')

    results = []
    full_dataset_bb_list = []
    full_dataset_segbb_list = []
    for dataset_path in [train_data_path, val_data_path, test_data_path]:
        loader = load_data_MaskRCNN(pkl_paths=[dataset_path], use_attr=True, no_img=False, batch_size=args.batch_size,
                                    return_img_id=True, add_other_mask=True)

        iou_mask_meter = AverageMeter()
        iou_bb_meter = AverageMeter()
        tv_box_iou_bb_meter = AverageMeter()
        cnt_nobirds = 0
        for _, data in enumerate(loader):
            # inputs, labels, attr_labels, segmentation, bounding_box = data
            inputs, labels, attr_labels, segmentation_label, boundingbox_label, img_id, _ = data
            inputs = inputs.float().cuda()

            outputs = preprocess_mask_outputs(inputs, model, args.threshold, img_id)

            # save segmented, cropped and resized images
            if "stage2_img_id_to_save" in outputs.keys():
                for i in range(len(outputs["stage2_img_id_to_save"])):
                    img_id = outputs["stage2_img_id_to_save"][i].item()
                    # save bb crops
                    save_path = f"{BASE_PATH}/data/CUB_Cropped/bbcrops/{map_id_to_img[img_id]}".replace('\\', '/')
                    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
                    transform_to_save_img(outputs["stage2_inputs_crop_box"][i]).save(save_path, format='JPEG')

                    # save segmented seg crops
                    save_path = f"{BASE_PATH}/data/CUB_Cropped/bbcrops_seg/{map_id_to_img[img_id]}".replace('\\', '/')
                    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
                    transform_to_save_img(outputs["stage2_bbcrop_segmented"][i]).save(save_path, format='JPEG')

                    # save segmentation edge crops
                    save_path = f"{BASE_PATH}/data/CUB_Cropped/segbbcrops/{map_id_to_img[img_id]}".replace('\\', '/')
                    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
                    transform_to_save_img(outputs["stage2_inputs_crop_segment"][i]).save(save_path, format='JPEG')

                    # save segmentation edge crops with segmentation mask applied
                    save_path = f"{BASE_PATH}/data/CUB_Cropped/segbbcrops_seg/{map_id_to_img[img_id]}".replace('\\', '/')
                    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
                    transform_to_save_img(outputs["stage2_inputs_crop_segment_segmented"][i]).save(save_path, format='JPEG')

                    # save cropped segmentation masks (masks are applicable only for segbb_crops)
                    save_path = f"{BASE_PATH}/data/CUB_Cropped/segmasks/{map_id_to_img[img_id]}".replace('\\', '/')
                    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
                    transform_to_save_mask(outputs["segmentation_mask"][i]).save(save_path, format='PNG')

                    # save bbs and segmentation bbs
                    full_dataset_bb_list.append(
                        [outputs["stage2_img_id_to_save"][i].item()] + outputs["stage2_bb"][i].cpu().tolist())
                    full_dataset_segbb_list.append(
                        [outputs["stage2_img_id_to_save"][i].item()] + outputs["stage2_segmentation_bb"][i].cpu().tolist())

        print('Mask IOU: ', iou_mask_meter.avg)
        print('BB IOU: ', iou_bb_meter.avg)
        print('TV BB IOU: ', tv_box_iou_bb_meter.avg)

        results.append(f"Threshold: {args.threshold}\tDataset-Type: {dataset_path.split('/')[-1].split('.')[0]}\tMask "
                       f"IoU: {iou_mask_meter.avg}\tBoundingBox IoU: {iou_bb_meter.avg}\t"
                       f"TV Box_IoU: {tv_box_iou_bb_meter.avg}\tno birds in image predicted: {cnt_nobirds}")

    with open(f"{BASE_PATH}/outputs/Pretrained_MaskRCNN_results_crop_new.txt", 'w') as file:
        file.write('\n'.join(results))
        file.close()

    # save bounding boxes of segmentations and direct bbs from mask-rcnn
    save_list_of_lists(full_dataset_segbb_list, f"{BASE_PATH}/data/CUB_Cropped/segmentation_bbs.txt".replace('\\', '/'))
    save_list_of_lists(full_dataset_bb_list, f"{BASE_PATH}/data/CUB_Cropped/maskrcnn_bbs.txt".replace('\\', '/'))


if __name__ == '__main__':
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='Hybrid Model Training')
    parser.add_argument('-batch_size', '-b', type=int, default=4, help='mini-batch size')
    parser.add_argument('-threshold', type=float, default=0.3,
                        help='threshold for score/certainty about segmentation class')
    parser.add_argument('-exp', type=str, help='Name of experiment to run.', default='save_segmented_imgs')
    parser.add_argument('-resize_imgs', action='store_true', help='Whether to resize the images before feeding them '
                                                                  'into the MaskRCNN, only allowed with batch_size==1')

    args = parser.parse_args()

    if args.exp == 'test_pretrained_maskrcnn':
        test_pretrained_maskrcnn(args)
    elif args.exp == 'save_segmented_imgs_new':
        save_segmented_imgs_new(args)
    # faulty implementation, kept as reference for now
    """elif args.exp == 'save_segmented_imgs':
        save_segmented_imgs(args)"""
    # CUDA_VISIBLE_DEVICES=0 python3 MaskBN_Pipeline.py -exp test_pretrained_maskrcnn -batch_size 1 -resize_imgs
    # CUDA_VISIBLE_DEVICES=2 python3 MaskBN_Pipeline.py -exp save_segmented_imgs_new -batch_size 1 -resize_imgs




