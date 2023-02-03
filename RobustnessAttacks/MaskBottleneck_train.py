import os
import argparse
from rtpt import RTPT
import yaml
import torch
import torchvision.models as tvmodels
from sklearn.metrics import f1_score, confusion_matrix

from dataset import load_data_MaskBottleneck, find_class_imbalance
from analysis import Logger, AverageMeter, binary_accuracy, accuracy
from config import BASE_PATH, n_attributes, model_dirs
from utils import load_model
from model_templates.utils_models import MLP


torch.set_num_threads(4)


def run_epoch_sequential(model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training, concept_model):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    concept_model.eval()
    if is_training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    for idx, data in enumerate(loader):
        inputs, class_labels = data
        if isinstance(inputs, list):
            # inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t()
            inputs = torch.flatten(inputs, start_dim=1).float()
        inputs = inputs.cuda()
        class_labels = class_labels.cuda()

        # use concept model to predict concepts -> feed these into the end model after applying sigmoid function
        stage2_inputs = torch.nn.Sigmoid()(concept_model(inputs))
        if args.binarize_attr:
            stage2_inputs = torch.where(stage2_inputs > 0.5)
        outputs = model(stage2_inputs)

        loss = criterion(outputs, class_labels)
        acc = accuracy(outputs, class_labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss_meter, acc_meter


def run_epoch_independent(model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            # inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t()
            inputs = torch.flatten(inputs, start_dim=1).float()
        inputs = inputs.float().cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss_meter, acc_meter


def run_epoch_concept(model, optimizer, loader, loss_meter, acc_meter, f1_meter, criterion, args, is_training):
    if is_training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    for _, data in enumerate(loader):
        inputs, _, attr_labels = data
        inputs = inputs.cuda()
        attr_labels = attr_labels.cuda()

        concept_outputs = model(inputs)
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

        acc_meter.update(binary_accuracy(torch.nn.Sigmoid()(concept_outputs), attr_labels).item(),
                         concept_outputs.shape[0])
        f1_meter.update(f1.item(), concept_outputs.shape[0])
        loss_meter.update(loss.item(), concept_outputs.shape[0])

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_meter, acc_meter, f1_meter


def train(model, args):
    # Create RTPT object
    rtpt = RTPT(name_initials='cstoll', experiment_name='Train MaskConcept', max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    model.cuda()

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

    if args.exp == 'Concept_XtoC':
        criterion = torch.nn.BCEWithLogitsLoss(weight=imbalance)
    elif args.exp == 'Independent_CtoY' or args.exp == 'Sequential_CtoY':
        criterion = torch.nn.CrossEntropyLoss()

    train_loader = load_data_MaskBottleneck([train_data_path], args.use_attr, args.no_img, args.batch_size,
                                            crop_type=args.crop_type, apply_segmentation=args.apply_segmentation,
                                            is_train=True)
    val_loader = load_data_MaskBottleneck([val_data_path], args.use_attr, args.no_img, args.batch_size,
                                          crop_type=args.crop_type, apply_segmentation=args.apply_segmentation)

    if args.exp == 'Sequential_CtoY':
        print(args.crop_type)
        if args.apply_segmentation:
            crop_type = f"{args.crop_type}_useseg"
        else:
            crop_type = args.crop_type
        concept_dict_dir = f"{BASE_PATH}/models/{model_dirs['MaskBottleneck_Concept']}{args.seed}/best_model.pth"\
            .replace("\\", '/').replace('+', crop_type)
        concept_model = load_model('MaskBottleneck_Concept', path_to_state_dict=concept_dict_dir)
        concept_model.cuda()
        concept_model.eval()

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        rtpt.step()
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        train_f1_meter = AverageMeter()

        if args.exp == 'Independent_CtoY':
            train_loss_meter, train_acc_meter = \
                run_epoch_independent(model=model, optimizer=optimizer, loader=train_loader, loss_meter=train_loss_meter,
                                      acc_meter=train_acc_meter, criterion=criterion, args=args, is_training=True)
        elif args.exp == 'Sequential_CtoY':
            train_loss_meter, train_acc_meter = \
                run_epoch_sequential(model=model, optimizer=optimizer, loader=train_loader, loss_meter=train_loss_meter,
                                     acc_meter=train_acc_meter, criterion=criterion, args=args, is_training=True,
                                     concept_model=concept_model)
        elif args.exp == 'Concept_XtoC':
            train_loss_meter, train_acc_meter, train_f1_meter = \
                run_epoch_concept(model=model, optimizer=optimizer, loader=train_loader, loss_meter=train_loss_meter,
                                  acc_meter=train_acc_meter, f1_meter=train_f1_meter, criterion=criterion, args=args,
                                  is_training=True)

        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        val_f1_meter = AverageMeter()

        with torch.no_grad():
            if args.exp == 'Independent_CtoY':
                val_loss_meter, val_acc_meter = \
                    run_epoch_independent(model=model, optimizer=optimizer, loader=val_loader,
                                          loss_meter=val_loss_meter,
                                          acc_meter=val_acc_meter, criterion=criterion, args=args, is_training=False)
            elif args.exp == 'Sequential_CtoY':
                train_loss_meter, train_acc_meter = \
                    run_epoch_sequential(model=model, optimizer=optimizer, loader=val_loader, loss_meter=val_loss_meter,
                                         acc_meter=val_acc_meter, criterion=criterion, args=args,
                                         is_training=False, concept_model=concept_model)
            elif args.exp == 'Concept_XtoC':
                val_loss_meter, val_acc_meter, val_f1_meter = \
                    run_epoch_concept(model=model, optimizer=optimizer, loader=val_loader, loss_meter=val_loss_meter,
                                      acc_meter=val_acc_meter, f1_meter=val_f1_meter, criterion=criterion, args=args,
                                      is_training=False)

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

        # early stopping
        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break


def train_XtoC(args):  # concept
    model = tvmodels.resnet34(pretrained=True, progress=True)
    model.fc = torch.nn.Linear(model.fc.in_features, n_attributes)  # set output of resnet to n_attributes
    train(model, args)


def train_CtoY(args):  # sequential
    model = MLP(input_dim=args.n_attributes, num_classes=args.num_classes, expand_dim=args.expand_dim)
    train(model, args)


def train_AtoY(args):  # independent
    model = MLP(input_dim=args.n_attributes, num_classes=args.num_classes, expand_dim=args.expand_dim)
    train(model, args)


def train_XtoCtoY(args):
    pass


def parse_arguments():
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='Hybrid Model Training')
    parser.add_argument('-exp', type=str,
                        choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY', 'Standard', 'Joint'],
                        help='Name of experiment to run.')  # HybridConceptModel_Large
    parser.add_argument('--seed', required=True, type=int, help='Numpy and torch seed.')
    parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
    parser.add_argument('-batch_size', '-b', type=int, default=2, help='mini-batch size')
    parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
    parser.add_argument('-lr', default=0.01, type=float, help="learning rate")
    parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('-pretrained', '-p', action='store_true',
                        help='whether to load pretrained model & just fine-tune')
    parser.add_argument('-use_attr', action='store_true',
                        help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
    parser.add_argument('-no_img', action='store_true',
                        help='if included, only use attributes (and not raw imgs) for class prediction')
    parser.add_argument('-n_attributes', type=int, default=112,
                        help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-data_dir', default='data/CUB_200_2011/CUB_processed/class_attr_data_10',
                        help='directory to the training data')
    parser.add_argument('-optimizer', default='SGD',
                        help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
    parser.add_argument('-scheduler', default='Step',
                        help='Type of scheduler to use, options incl Step, CosineAnnealing')
    parser.add_argument('-metric', type=str, default='loss',
                        help='The metric to optimize the model to. Choose one of: [loss, accuracy],  default = loss')
    parser.add_argument('-step_size', type=int, default=1000,
                        help='Number of steps before decaying current learning rate by half')
    parser.add_argument('-num_classes', type=int, default=200, help='Number of classes to use')
    parser.add_argument('-expand_dim', default=0, type=int,
                        help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
    parser.add_argument('-imbalance', action='store_true',
                        help='whether to use the class imbalance sampler and weighted loss propsed in the '
                             'concept bottleneck paper')
    parser.add_argument('-in_channels', type=int, default=3,
                        help='number of input channels (for RGB images 3, concept model), default 3')
    parser.add_argument('-crop_type', type=str, default='labelbb',
                        help='type of image cropping applied after pass trough Mask-RCNN (if not labelbb chosen, '
                             'labelbb are annotations from cub), Choose: [labelbb, segbb, cropbb] '
                             '- for Concept & Sequential Models')
    parser.add_argument('-apply_segmentation', action='store_true',
                        help='Wheter to apply a segmentation mask to the cropped image, only works for crop_types: '
                             '[labelbb, segbb, cropbb]')
    parser.add_argument('-binarize_attr', action='store_true', help='Apply binarization on predicted concepts')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if args.exp == 'Concept_XtoC':
        train_XtoC(args)
    elif args.exp == 'Independent_CtoY':
        train_AtoY(args)
    elif args.exp == 'Sequential_CtoY':
        train_CtoY(args)
    elif args.exp == 'Joint':
        train_XtoCtoY(args)
    else:
        print('wrong experiment!')


