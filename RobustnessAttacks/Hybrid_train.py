"""
Train Hybrid Network using the CUB-200-2011 dataset
"""
import os
import argparse
import torch
import torch.multiprocessing as mp
from rtpt import RTPT
import yaml
from analysis import Logger, AverageMeter, accuracy
import utils

from dataset import load_data_hybrid, find_class_imbalance_slots
from config import BASE_PATH, model_dirs
from model_templates.HybridModel.models import HybridConceptModel, HybridResNet, JointHybrid_Model, SetTransformer, \
    HybridConceptModel_Large
from sklearn.metrics import f1_score, confusion_matrix


def run_epoch_sequential(model, optimizer, loader, loss_meter, acc_meter, criterion, is_training, concept_model=None):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()
    concept_model.eval()
    for idx, data in enumerate(loader):
        inputs, class_labels = data
        inputs_var = torch.autograd.Variable(inputs).float().cuda()
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(class_labels).cuda()
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        # use concept model to predict concepts -> feed these into the end model
        stage2_inputs = concept_model(inputs_var)

        if args.binarize_attr:
            ones = torch.ones(stage2_inputs.shape).float().cuda()
            zeros = torch.zeros(stage2_inputs.shape).float().cuda()
            stage2_inputs = torch.where(stage2_inputs > 0.2, ones, zeros)
        outputs = model(stage2_inputs)

        loss = criterion(outputs, labels_var)
        acc = accuracy(outputs, class_labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0].item(), inputs.size(0))

        if is_training:
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()
            optimizer.step()  # optimizer step to update parameters
    return loss_meter, acc_meter


def run_epoch_independent(model, optimizer, loader, loss_meter, acc_meter, criterion, is_training):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            # inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t()
            inputs = torch.flatten(inputs, start_dim=1).float()
        inputs_var = torch.autograd.Variable(inputs).float().cuda()
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels).cuda()
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        outputs = model(inputs_var)
        loss = criterion(outputs, labels_var)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()
            optimizer.step()  # optimizer step to update parameters
    return loss_meter, acc_meter


def run_epoch_concept(model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training, thread_pool=None):
    """
    For the Concept Part (X -> A, cotraining, simple finetune)
    """
    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):
        if not args.bottleneck:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels = data
            attr_labels_var = torch.autograd.Variable(attr_labels).float()
            attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var
            attr_labels_var = torch.squeeze(attr_labels_var)

        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        # calculate loss & accuracy depending on the task
        if args.exp == 'Joint':
            # X->C->Y
            class_pred, attr_pred = model(inputs_var)
            attr_loss = utils.hungarian_loss(attr_pred, attr_labels_var, thread_pool=thread_pool)
            class_loss = criterion(class_pred, labels_var)

            loss = class_loss + args.attr_loss_weight * attr_loss

            acc = accuracy(class_pred, labels, topk=(1,))
            acc_meter.update(acc[0], inputs.size(0))
        else:
            # X->C
            outputs = model(inputs_var)
            # calculate concept accuracy

            matched_preds = utils.hungarian_matching(attr_labels_var, preds_attrs=outputs)
            loss = criterion(matched_preds, attr_labels_var)

            tn, fp, fn, tp = confusion_matrix(attr_labels.cpu().detach().flatten(start_dim=0).numpy(),
                                              matched_preds.cpu().detach().flatten(start_dim=0).numpy() > 0.5,
                                              normalize='true').flatten()

            # 1. Durchgang (Baseline) loss = loss
            # 2. Durchgang (LossCM_bn_hc) loss = (loss + fp + fn) * 0.5
            # 3. Durchgang (LossCMFN_bh_hc) loss = (loss + fp + 4*fn) / 6.
            # 4. Durchgang (Loss_CM2FN) loss = (loss + fp + 2*fn) / 2.

            loss = (loss + fp + 2*fn) / 2.

            # add f1 score instead of top5 accuracy
            f1 = f1_score(attr_labels_var.cpu().detach().flatten(start_dim=0).numpy(),
                          matched_preds.cpu().detach().flatten(start_dim=0).numpy() > 0.5)
            acc_meter.update(f1, 1)

            """
            loss = utils.hungarian_loss(outputs, attr_labels_var, thread_pool=thread_pool)
            
            matched = utils.hungarian_matching(attr_labels_var, outputs)
            out, counts = torch.unique((matched > 0.5) == (attr_labels_var > 0.5), return_counts=True)
            if torch.where(out)[0].shape[0] > 0:
                correct = counts[torch.where(out == True)[0][0]]
            else:
                correct = 0

            acc = correct * 100.0 / (args.batch_size * (args.n_attributes + 1) * args.n_slots)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))"""
        loss_meter.update(loss, outputs.shape[0])
        # loss_meter.update(loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_meter, acc_meter


def train(model, args):
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

    model = model.cuda()

    # load concept model in case of sequential training
    if args.exp == 'Sequential_CtoY':
        concept_name = '_'.join('Independent_HybridConcept-CNN_Loss'.split('_')[1:])
        concept_dict_dir = f'{BASE_PATH}/models/{model_dirs[concept_name]}{args.seed}/best_model.pth'.replace("\\", '/')
        concept_model = utils.load_model(args.concept_model, concept_dict_dir)
        concept_model.cuda()

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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00005)

    train_data_path = os.path.join(BASE_PATH, args.data_dir, 'train.pkl').replace("\\", '/')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write('train data path: %s\n' % train_data_path)

    """# Determine attribute imbalance
    imbalance = None
    if args.use_attr and not args.no_img and args.imbalance:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        if args.weighted_loss == 'multiple':
            imbalance = utils.find_class_imbalance(train_data_path, multiple_attr=True)
        else:
            imbalance = utils.find_class_imbalance(train_data_path, multiple_attr=False)
    logger.write(str(imbalance) + '\n')"""

    imbalance = find_class_imbalance_slots(train_data_path, True, args.n_slots).cuda()

    criterion = torch.nn.BCELoss(weight=imbalance)
    # additional loss for attributes
    """if args.use_attr and not args.no_img:
        attr_criterion = []  # separate criterion (loss function) for each attribute
        if args.imbalance:
            assert (imbalance is not None)
            for ratio in imbalance:
                attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda()))
        else:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None"""

    train_loader = load_data_hybrid([train_data_path], args.use_attr, args.no_img, args.batch_size,
                                    image_dir=args.image_dir, img_size=args.img_size, is_train=True,
                                    num_slots=args.n_slots)
    val_loader = load_data_hybrid([val_data_path], args.use_attr, args.no_img, args.batch_size, num_slots=args.n_slots,
                                  image_dir=args.image_dir, img_size=args.img_size)

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        with mp.Pool(10) as pool:
            rtpt.step()
            train_loss_meter = AverageMeter()
            train_acc_meter = AverageMeter()
            if args.exp == 'Independent_CtoY':
                train_loss_meter, train_acc_meter = run_epoch_independent(model, optimizer, train_loader,
                                                                          train_loss_meter, train_acc_meter, criterion,
                                                                          is_training=True)
            elif args.exp == 'Sequential_CtoY':
                train_loss_meter, train_acc_meter = run_epoch_sequential(model, optimizer, train_loader,
                                                                         train_loss_meter, train_acc_meter, criterion,
                                                                         is_training=True, concept_model=concept_model)
            else:
                train_loss_meter, train_acc_meter = run_epoch_concept(model, optimizer, train_loader, train_loss_meter,
                                                                      train_acc_meter, criterion, args,
                                                                      is_training=True, thread_pool=pool)

            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()

            with torch.no_grad():
                if args.exp == 'Independent_CtoY':
                    val_loss_meter, val_acc_meter = run_epoch_independent(model, optimizer, val_loader, val_loss_meter,
                                                                          val_acc_meter, criterion, args,
                                                                          is_training=False, thread_pool=pool)
                elif args.exp == 'Sequential_CtoY':
                    val_loss_meter, val_acc_meter = run_epoch_sequential(model, optimizer, val_loader, val_loss_meter,
                                                                         val_acc_meter, criterion, is_training=False,
                                                                         concept_model=concept_model)
                else:
                    val_loss_meter, val_acc_meter = run_epoch_concept(model, optimizer, val_loader, val_loss_meter,
                                                                      val_acc_meter, criterion, args, is_training=False,
                                                                      thread_pool=pool)

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

            """logger.write('Epoch [%d]:\tTrain loss: %.6f\tTrain accuracy: %.4f\t'
                         'Val loss: %.6f\tVal acc: %.4f\t'
                         'Best val epoch: %d\n'
                         % (
                             epoch, train_loss_avg, train_acc_meter.avg, val_loss_avg, val_acc_meter.avg,
                             best_val_epoch))"""
            logger.write('Epoch [%d]:\tTrain loss: %.6f\tTrain F1-Score: %.4f\t'
                         'Val loss: %.6f\tVal F1-Score: %.4f\t'
                         'Best val epoch: %d\n'
                         % (
                             epoch, train_loss_avg, train_acc_meter.avg, val_loss_avg, val_acc_meter.avg,
                             best_val_epoch))
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


def train_XtoC(args):  # concept
    model = HybridConceptModel(n_slots=args.n_slots, n_iters=args.n_iters, n_attr=args.n_attributes,
                               img_size=args.img_size, in_channels=args.in_channels,
                               encoder_hidden_channels=args.encoder_hidden_channels,
                               attention_hidden_channels=args.attention_hidden_channels,
                               bn_hidden_channels=args.bn_hidden_channels)
    train(model, args)


def train_XtoC_Resnet(args):  # concept ResNet
    model = HybridResNet(n_slots=args.n_slots, n_iters=args.n_iters, n_attr=args.n_attributes,
                         img_size=args.img_size, in_channels=args.in_channels,
                         encoder_hidden_channels=args.encoder_hidden_channels,
                         attention_hidden_channels=args.attention_hidden_channels,
                         bn_hidden_channels=args.bn_hidden_channels)
    train(model, args)


def train_XtoC_Large(args):  # concept large
    model = HybridConceptModel_Large(n_slots=args.n_slots, n_iters=args.n_iters, n_attr=args.n_attributes,
                                     img_size=args.img_size, in_channels=args.in_channels,
                                     encoder_hidden_channels=args.encoder_hidden_channels,
                                     attention_hidden_channels=args.attention_hidden_channels,
                                     bn_hidden_channels=args.bn_hidden_channels)
    train(model, args)


def train_CtoY(args):  # sequential
    model = SetTransformer(dim_input=args.n_attributes + 1, dim_output=args.num_classes, dim_hidden=args.set_dim_hidden,
                           num_heads=args.set_num_heads)
    train(model, args)


def train_AtoY(args):  # independent
    model = SetTransformer(dim_input=args.n_attributes + 1, dim_output=args.num_classes, dim_hidden=args.set_dim_hidden,
                           num_heads=args.set_num_heads)
    train(model, args)


def train_XtoCtoY(args):  # joint
    model = JointHybrid_Model(n_attr=args.n_attributes, n_classes=args.num_classes, img_size=args.img_size,
                              n_slots=args.n_slots, n_iters=args.n_iters, return_concepts=True,
                              in_channels=args.in_channels, encoder_hidden_channels=args.encoder_hidden_channels,
                              attention_hidden_channels=args.attention_hidden_channels,
                              bn_hidden_channels=args.bn_hidden_channels)
    train(model, args)


def parse_arguments():
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='Hybrid Model Training')
    parser.add_argument('-exp', type=str,
                        choices=['Concept_XtoC', 'Concept_XtoC_ResNet', 'Concept_XtoC_Large', 'Independent_CtoY',
                                 'Sequential_CtoY',
                                 'Standard', 'Joint'],
                        help='Name of experiment to run.')  # HybridConceptModel_Large
    parser.add_argument('--seed', required=True, type=int, help='Numpy and torch seed.')

    parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
    parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
    parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
    parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
    parser.add_argument('-lr', default=0.01, type=float, help="learning rate")
    parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('-pretrained', '-p', action='store_true',
                        help='whether to load pretrained model & just fine-tune')
    parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
    parser.add_argument('-use_attr', action='store_true',
                        help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
    parser.add_argument('-attr_loss_weight', default=1.0, type=float,
                        help='weight for loss by predicting attributes, used for Joint model')
    parser.add_argument('-no_img', action='store_true',
                        help='if included, only use attributes (and not raw imgs) for class prediction')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels',
                        action='store_true')
    parser.add_argument('-n_attributes', type=int, default=112,
                        help='whether to apply bottlenecks to only a few attributes')
    parser.add_argument('-data_dir', default='official_datasets', help='directory to the training data')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-end2end', action='store_true',
                        help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg')
    parser.add_argument('-optimizer', default='SGD',
                        help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
    parser.add_argument('-scheduler', default='CosineAnnealing',
                        help='Type of scheduler to use, options incl Step, CosineAnnealing')
    parser.add_argument('-metric', type=str, default='loss',
                        help='The metric to optimize the model to. Choose one of: [loss, accuracy],  default = loss')
    parser.add_argument('-scheduler_step', type=int, default=1000,
                        help='Number of steps before decaying current learning rate by half')
    parser.add_argument('-n_slots', type=int, default=1, help='Number of slots')
    parser.add_argument('-n_iters', type=int, default=1, help='Number of Iterations of the Slot Attention Module')
    parser.add_argument('-img_size', type=int, default=256,
                        help='The size of the Images (has to be divisible by 4, default: 256')
    parser.add_argument('-num_classes', type=int, default=200, help='Number of classes to use')
    parser.add_argument('-imbalance', action='store_true',
                        help='whether to use the class imbalance sampler and weighted loss propsed in the '
                             'concept bottleneck paper')
    parser.add_argument('-in_channels', type=int, default=3,
                        help='number of input channels (for RGB images 3, concept model), default 3')
    parser.add_argument('-encoder_hidden_channels', type=int, default=128,
                        help='number of hidden channels of the position encoder (concept model), default: 256')
    parser.add_argument('-bn_hidden_channels', type=int, default=256,
                        help='number of hidden channels of the bottleneck layer (concept model), default: 256')
    parser.add_argument('-attention_hidden_channels', type=int, default=256,
                        help='number of hidden channels of the SlotAttention Module (concept model), default: 256')
    parser.add_argument('-set_dim_hidden', type=int, default=128,
                        help='number of hidden channels in the set transformer (end module), default: 128')
    parser.add_argument('-set_num_heads', type=int, default=4,
                        help='number of attention heads of the set transformer (end module), default: 4')
    parser.add_argument('-concept_model', type=str, default='',
                        help='the concept model in case the experiment is SequentialCtoY, '
                             'Choose one of: [HybridConcept-CNN_Acc, HybridConcept-CNN_Loss], default: ''')
    parser.add_argument('-binarize_attr', action='store_true',
                        help='Wheter to binarize the Concept Output, default: False')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    if args.exp == 'Concept_XtoC':
        train_XtoC(args)
    elif args.exp == 'Concept_XtoC_ResNet':
        train_XtoC_Resnet(args)
    elif args.exp == 'Concept_XtoC_Large':
        train_XtoC_Large(args)
    elif args.exp == 'Independent_CtoY':
        train_AtoY(args)
    elif args.exp == 'Sequential_CtoY':
        train_CtoY(args)
    elif args.exp == 'Joint':
        train_XtoCtoY(args)
