import os
import argparse
from rtpt import RTPT
import yaml
from analysis import Logger, AverageMeter, binary_accuracy
import utils
from config import BASE_PATH, n_attributes
import torch
import numpy as np
import torchvision.models as tvmodels
from torchvision.transforms import transforms as transforms
from dataset import load_data_MaskRCNN, find_class_imbalance
from sklearn.metrics import f1_score, confusion_matrix
from config import COCO_INSTANCE_CATEGORY_NAMES as coco_names


def get_2nd_stage_inputs(segmentation_model, data, transform_stage2, args):
    # inputs, labels, attr_labels, segmentation, bounding_box = data
    inputs, labels, attr_labels, _, _ = data
    inputs = inputs.float().cuda()

    segmentation_model.eval()
    segmentation_outputs = segmentation_model(inputs)  # output is list of #b dicts

    bird_idx = coco_names.index('bird')

    # handle each element in batch
    # segmentation_masks = []
    attr_labels_remaining = []
    stage2_inputs = []
    for i in range(args.batch_size):
        """threshold scores to get certainty gt 'threshold', get number of remaining scores 
        (outputs are sorted by descending scores)"""
        scores = segmentation_outputs[i]['scores']
        threshold_idx = len([utils.index(scores, x) for x in scores if x > args.threshold])

        # apply threshold on outputs
        labels = segmentation_outputs[i]['labels'][:threshold_idx]
        bboxes = segmentation_outputs[i]['boxes'][:threshold_idx]
        seg_masks = segmentation_outputs[i]['masks'][:threshold_idx]

        if bird_idx in labels:
            # extract bird predictions for masks and bounding_boxes
            bird_boxes = torch.stack([bb for label, bb in zip(labels, bboxes) if label == bird_idx])
            # extract most outer coordinates of bounding boxes
            bounding_box = torch.tensor(
                [bird_boxes[:, 0].min(), bird_boxes[:, 1].min(), bird_boxes[:, 2].max(), bird_boxes[:, 3].max()])

            # cut out bounding box area of image and segmentation mask
            stage2_inputs.append(inputs[i, :, bounding_box[0].item():bounding_box[2].item() + 1,
                                 bounding_box[1].item():bounding_box[3].item() + 1])

            # segmentation masks are currently not required
            """segmentation_masks.append(
                torch.stack([m > 0.5 for label, m in zip(labels, seg_masks) if label == bird_idx])
                    .sum(axis=0)[i, :, bounding_box[0].item():bounding_box[2].item() + 1, bounding_box[1].item():bounding_box[3].item() + 1])"""

            attr_labels_remaining.append(attr_labels[i])
        else:
            # nothing to do here
            pass
    # segmentation_masks = torch.stack([transforms.RandomResizedCrop((224, 224))(m) for m in segmentation_masks])

    # fit each tensor to the same size (224x224 for resnet34) and make a batch out of it again
    stage2_inputs = torch.stack([transform_stage2(x) for x in stage2_inputs])
    attr_labels_remaining = torch.stack(attr_labels_remaining)

    return stage2_inputs.cuda(), attr_labels_remaining.cuda()


def run_epoch_concept(model, segmentation_model, optimizer, loader, loss_meter, acc_meter, f1_meter, criterion, args,
                      is_training):
    if is_training:
        model.train()
        torch.set_grad_enabled(True)
        transform_stage2 = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        model.eval()
        torch.set_grad_enabled(False)
        transform_stage2 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    for idx, data in enumerate(loader):
        stage2_inputs, attr_labels = get_2nd_stage_inputs(segmentation_model=segmentation_model, data=data,
                                                          transform_stage2=transform_stage2, args=args)

        concept_outputs = model(stage2_inputs)
        loss = criterion(concept_outputs, attr_labels.float())

        tn, fp, fn, tp = confusion_matrix(attr_labels.cpu().detach().flatten(start_dim=0).numpy(),
                                          concept_outputs.cpu().detach().flatten(start_dim=0).numpy() > 0.5,
                                          normalize='true').flatten()

        # 1. Durchgang (Baseline) loss = loss
        # 2. Durchgang (LossCM_bn_hc) loss = (loss + fp + fn) * 0.5
        # 3. Durchgang (LossCMFN_bh_hc) loss = (loss + fp + 4*fn) / 6.
        # 4. Durchgang (Loss_CM2FN) loss = (loss + fp + 2*fn) / 2.
        loss = (loss + fp + 2 * fn) / 2.

        # add f1 score instead of top5 accuracy
        f1 = f1_score(attr_labels.cpu().detach().flatten(start_dim=0).numpy(),
                      concept_outputs.cpu().detach().flatten(start_dim=0).numpy() > 0.5)

        acc_meter.update(binary_accuracy(torch.nn.Sigmoid()(concept_outputs), attr_labels), concept_outputs.shape[0])
        f1_meter.update(f1, 1)
        loss_meter.update(loss, concept_outputs.shape[0])

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_meter, acc_meter, f1_meter


def train(args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name='Train Hybrid Model', max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    if os.path.exists(args.log_dir):  # replace old folder by a new one
        for f in os.listdir(args.log_dir):
            os.remove(os.path.join(args.log_dir, f))
    else:
        os.makedirs(args.log_dir)

    # Save args as text
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    with open(os.path.join(args.log_dir, 'args.yaml').replace("\\", '/'), 'w') as f:
        f.write(args_text)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'))
    logger.write(str(args) + '\n')
    logger.flush()

    # num_classes of ms COCO -> birds included
    model = tvmodels.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=2)
    model.cuda()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                        weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)

    if args.scheduler == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00005)

    train_data_path = os.path.join(BASE_PATH, args.data_dir, 'train.pkl').replace("\\", '/')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write('train data path: %s\n' % train_data_path)

    imbalance = find_class_imbalance(train_data_path, True).cuda()
    criterion = torch.nn.BCEWithLogitsLoss(weight=imbalance)

    train_loader = load_data_MaskRCNN([train_data_path], args.use_attr, args.no_img, args.batch_size, normalize=False)
    val_loader = load_data_MaskRCNN([val_data_path], args.use_attr, args.no_img, args.batch_size, normalize=False)

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        rtpt.step()
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        train_f1_meter = AverageMeter()

        train_loss_meter, train_acc_meter, train_f1_meter = \
            run_epoch_concept(model=model, optimizer=optimizer, loader=train_loader, loss_meter=train_loss_meter,
                              acc_meter=train_acc_meter, f1_meter=train_f1_meter, criterion=criterion, args=args,
                              is_training=True)

        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        val_f1_meter = AverageMeter()

        with torch.no_grad():
            val_loss_meter, val_acc_meter, val_f1_meter = \
                run_epoch_concept(model=model, optimizer=optimizer, loader=val_loader, loss_meter=val_loss_meter,
                                  acc_meter=val_acc_meter, f1_meter=val_f1_meter, criterion=criterion, args=args)

        if args.metric == 'loss':
            if best_val_loss > val_loss_meter.avg:
                best_val_epoch = epoch
                best_val_loss = val_loss_meter.avg

                save_state = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                    'lr_scheduler': scheduler.state_dict(),
                }
                logger.write('New model best model at epoch %d\n' % epoch)
                torch.save(save_state, os.path.join(args.log_dir, 'best_model.pth').replace("\\", '/'))
        else:
            if best_val_acc < val_acc_meter.avg:
                best_val_epoch = epoch
                best_val_acc = val_acc_meter.avg

                save_state = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                    'lr_scheduler': scheduler.state_dict(),
                }
                logger.write('New model best model at epoch %d\n' % epoch)
                torch.save(save_state, os.path.join(args.log_dir, 'best_model.pth').replace("\\", '/'))

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg

        logger.write('Epoch [%d]:\tTrain loss: %.6f\tTrain accuracy: %.4f\tTrain F1-Score: %.4f\t'
                     'Val loss: %.6f\tVal acc: %.4f\tVal F1-Score: %.4f\t'
                     'Best val epoch: %d\n'
                     % (
                         epoch, train_loss_avg, train_acc_meter.avg, train_f1_meter.avg, val_loss_avg,
                         val_acc_meter.avg, val_f1_meter.avg, best_val_epoch))
        logger.flush()

        scheduler.step()  # scheduler step to update lr at the end of epoch
        # inspect lr
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_last_lr())

        """# early stopping
        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break"""
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break


def parse_arguments():
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='Hybrid Model Training')
    parser.add_argument('--seed', required=True, type=int, help='Numpy and torch seed.')
    parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
    parser.add_argument('-batch_size', '-b', type=int, default=2, help='mini-batch size')
    parser.add_argument('-epochs', '-e', type=int, default=20, help='epochs for training process')
    parser.add_argument('-lr', default=0.005, type=float, help="learning rate")
    parser.add_argument('-weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('-pretrained', '-p', action='store_true',
                        help='whether to load pretrained model & just fine-tune')
    parser.add_argument('-data_dir', default='data/CUB_200_2011/CUB_processed/class_attr_data_10',
                        help='directory to the training data')
    parser.add_argument('-optimizer', default='SGD',
                        help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
    parser.add_argument('-scheduler', default='Step',
                        help='Type of scheduler to use, options incl Step, CosineAnnealing')
    parser.add_argument('-metric', type=str, default='loss',
                        help='The metric to optimize the model to. Choose one of: [loss, accuracy],  default = loss')
    parser.add_argument('-step_size', type=int, default=3,
                        help='Number of steps before decaying current learning rate by half')
    parser.add_argument('-img_size', type=int, default=256,
                        help='The size of the Images (has to be divisible by 4, default: 256')
    parser.add_argument('-num_classes', type=int, default=2, help='Number of classes to use')
    parser.add_argument('-threshold', type=float, default=0.5,
                        help='threshold for score/certainty about segmentation class')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    #train(args)
