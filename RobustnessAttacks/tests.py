import torch

import random
import numpy as np
import re
from PIL import Image
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

import analysis
import utils
import glob
from dataset import load_data_cb, load_data_scouter, load_data_vit, load_data_hybrid, CUB_Dataset_Hybrid
from config import model_dirs, n_attributes, BASE_PATH
from sklearn.metrics import f1_score, confusion_matrix, multilabel_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(123)


def test_scouter(model_names, batch_size, rtpt=None):
    print('this is the path: ', BASE_PATH)
    seeds = [1, 2, 3]
    num_examples = None
    set = 'test'

    for seed in seeds:
        ckpt = 9
        for m in model_names:
            if rtpt is not None:
                rtpt.step()

            # init data loader
            num_classes = int(re.findall(r'\d+', m)[0])
            data_dir = (BASE_PATH + '/data/CUB_200_2011/').replace("\\", '/')
            if set == 'train':
                train = True
            else:
                train = False
            loader = load_data_scouter(data_dir=data_dir, num_classes=num_classes, batch_size=batch_size, train=train,
                                       normalize=True)

            # this is used later to calculate the accuracy
            if num_examples is not None:
                num_imgs = num_examples
            else:
                num_imgs = len(loader.dataset)

            accs = []
            accs.append('')
            accs.append('')
            while ckpt <= 149:
                s_chkpt = f"{ckpt:04d}"

                scouter_dict_dir = (BASE_PATH + '/models/' + model_dirs[m]).replace('\\', '/').replace('+', str(seed))
                scouter_dict_dir = scouter_dict_dir.replace('.', s_chkpt + '.')
                print(scouter_dict_dir)

                model = utils.load_model(m, path_to_state_dict=scouter_dict_dir)
                torch.backends.cudnn.enabled = False  # TODO ist this the problem?

                model.cuda()
                model.eval()

                correct = 0
                cnt = 0
                cnt_images = 0
                for _, data in enumerate(loader):
                    cnt += 1
                    # data: [img_as_tensor, class_number_as_tensor, ggf. attribute as list]
                    inputs, labels = data
                    input_adv = torch.autograd.Variable(inputs).cuda()

                    # cast to float to be able to optimize input tensor
                    input_adv.float()

                    perturbed = input_adv.clone().detach()
                    cnt_images += perturbed.shape[0]

                    adv_pred = model(perturbed)
                    new_pred = (adv_pred.max(1, keepdim=True)[1]).detach().cpu()

                    # Check for success
                    correctly_predicted_labels = torch.eq(new_pred.squeeze(), labels).numpy()
                    out, counts = np.unique(correctly_predicted_labels, return_counts=True)

                    if np.where(out == True)[0].size > 0:
                        correct += counts[np.where(out == True)[0][0]]

                    # stop conditions: general case if not the full dataset is used
                    if num_examples is not None and cnt >= num_examples - 1:
                        break

                if num_examples is not None:
                    final_acc = correct / float(num_imgs * batch_size)
                    res = "Model: {}\t Chkpt: {}\t Test Accuracy = {} / {} = {}".format(m, s_chkpt, correct,
                                                                                        num_imgs * batch_size,
                                                                                        final_acc)
                    # print(res)
                else:
                    final_acc = correct / float(len(loader.dataset))
                    res = "Model: {}\t Chkpt: {}\t Test Accuracy = {} / {} = {}".format(m, s_chkpt, correct,
                                                                                        len(loader.dataset), final_acc)
                    print(res)
                accs.append(res)
                ckpt += 10

            with open(BASE_PATH + '/outputs/test_scouter_chkpts.txt', 'a') as f:
                f.write('\n'.join(accs))
                f.close()


def test_vit(model_names, batch_size, seeds, rtpt=None, filename=''):
    num_examples = None

    results = []
    for m in model_names:

        # init data loader
        data_dir = (BASE_PATH + '/data/CUB_200_2011/').replace("\\", '/')
        loader = load_data_vit(data_dir=data_dir, num_classes=200, batch_size=batch_size, dataset_type='test',
                               img_size=224, normalize=True)

        # this is used later to calculate the accuracy
        if num_examples is not None:
            num_imgs = num_examples
        else:
            num_imgs = len(loader.dataset)

        for seed in seeds:
            rtpt.step()

            model_dict_dir = \
                (BASE_PATH + '/models/' + model_dirs[m]).replace("\\", '/').replace('+', str(seed))
            model = utils.load_model('vits_p16_224', model_dict_dir)

            model.cuda()
            model.eval()

            correct = 0
            cnt = 0
            cnt_images = 0
            for _, data in enumerate(loader):
                cnt += 1
                # data: [img_as_tensor, class_number_as_tensor, ggf. attribute as list]
                inputs, labels = data
                input_adv = torch.autograd.Variable(inputs).cuda()

                # cast to float to be able to optimize input tensor
                input_adv.float()

                perturbed = input_adv.clone().detach()
                cnt_images += perturbed.shape[0]

                adv_pred = model(perturbed)
                new_pred = (adv_pred.max(1, keepdim=True)[1]).detach().cpu()

                # Check for success
                correctly_predicted_labels = torch.eq(new_pred.squeeze(), labels).numpy()
                out, counts = np.unique(correctly_predicted_labels, return_counts=True)

                if np.where(out == True)[0].size > 0:
                    correct += counts[np.where(out == True)[0][0]]

                # stop conditions: general case if not the full dataset is used
                if num_examples is not None and cnt >= num_examples - 1:
                    break

            if num_examples is not None:
                final_acc = correct / float(num_imgs * batch_size)
                res = "Model: {}\t Seed: {}\t Test Accuracy = {} / {} = {}".format(m, seed, correct,
                                                                                   num_imgs * batch_size, final_acc)
                print(res)
            else:
                final_acc = correct / float(len(loader.dataset))
                res = "Model: {}\t Seed: {}\t Test Accuracy = {} / {} = {}".format(m, seed, correct,
                                                                                   len(loader.dataset), final_acc)
                print(res)
            results.append(res)

    with open(BASE_PATH + f'/outputs/test_vits{filename}.txt', 'a') as f:
        f.write('\n'.join(results))
        f.close()


def test_segmentation():
    model_names = ['cub_black_cb_standard']
    m = model_names[0]
    path_to_dataset_diff_from_cub = 'CUB_black/'
    batch_size = 2
    if 'cb' in m:
        data_dir = (BASE_PATH + '/data/CUB_200_2011/CUB_processed/class_attr_data_10/' + 'test' + '.pkl').replace("\\",
                                                                                                                  '/')

        if 'cub_black' in m:
            image_dir = ('AdversarialData/' + path_to_dataset_diff_from_cub).replace("\\", '/')
            loader = load_data_cb([data_dir], use_attr=False, no_img=False, batch_size=batch_size,
                                  image_dir=image_dir, n_class_attr=2, normalize=False, use_segmentation=True)

        for _, data in enumerate(loader):
            # data: [img_as_tensor, class_number_as_tensor, ggf. attribute as list]
            inputs, labels, segmentations = data
            input_adv = torch.autograd.Variable(inputs).cuda()
            labels_var = torch.autograd.Variable(labels).cuda()

            break


def hybrid_vis():
    m = 'HybridConcept-CNN_Loss'
    seed = 1
    concept_dict_dir = f'{BASE_PATH}/models/{model_dirs[m]}{seed}/best_model.pth'.replace("\\", '/')
    model = utils.load_model(model_name=m, path_to_state_dict=concept_dict_dir,
                             return_hybrid_attn=True)

    model.cuda()
    model.eval()

    # use the first seed for the args file, since except for seed related arguments, all others are equal
    args = utils.load_model_args_as_dict(f'{BASE_PATH}/models/{model_dirs[m]}1/args.yaml')

    data_dir = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl'.replace("\\", '/')
    loader = load_data_hybrid([data_dir], use_attr=args['use_attr'], no_img=False, batch_size=2, normalize=True,
                              num_classes=args['num_classes'])

    correct = 0
    for idx, data in enumerate(loader):
        # data: [img_as_tensor, class_number_as_tensor, ggf. attribute as list]

        if args['use_attr']:
            inputs, labels, attribute_labels = data
            attr_labels_var = torch.autograd.Variable(attribute_labels).float()
            attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var
        else:
            inputs, labels = data

        # cast to float to be able to optimize input tensor
        input_adv = torch.autograd.Variable(inputs.float()).cuda()
        labels_var = torch.autograd.Variable(labels).cuda()

        # set return attn to True
        outputs, attn, updates = model(input_adv)
        slots_vis = attn.clone()

        print(attn.shape)
        print(updates.shape)
        # TODO from utils_plot import visualize_slots (testen)
        # TODO
        if self.slots_per_class > 1:
            new_slots_vis = torch.zeros((slots_vis.size(0), args['n_slots'], slots_vis.size(-1)))
            for slot_class in range(args['num_classes']):
                new_slots_vis[:, slot_class] = \
                    torch.sum(torch.cat(
                        [slots_vis[:, self.slots_per_class * slot_class: self.slots_per_class * (slot_class + 1)]],
                        dim=1),
                        dim=1, keepdim=False)
            slots_vis = new_slots_vis.to(attn.device)

        slots_vis = slots_vis[0]
        slots_vis = ((slots_vis - slots_vis.min()) / (slots_vis.max() - slots_vis.min()) * 255.).reshape(
            slots_vis.shape[:1] + (int(slots_vis.size(1) ** 0.5), int(slots_vis.size(1) ** 0.5)))
        slots_vis = (slots_vis.cpu().detach().numpy()).astype(np.uint8)
        for id, image in enumerate(slots_vis):
            image = Image.fromarray(image, mode='L')
            image.save(f'{BASE_PATH}/outputs/examples/slot_vis/{id}.png')
        print('Sum attn: ', torch.sum(attn.clone(), dim=2, keepdim=False))
        print('Sum updates: ', torch.sum(updates.clone(), dim=2, keepdim=False))

        matched = utils.hungarian_matching(attr_labels_var, outputs)
        out, counts = torch.unique((matched > 0.5) == (attr_labels_var > 0.5), return_counts=True)
        if torch.where(out)[0].shape[0] > 0:
            correct += counts[torch.where(out)[0][0]]

        print(f'correct{correct}')
        print(correct / (attr_labels_var.shape[0] * 2. * 113))

        # Check for successful/wrong predictions
        # result = utils.check_attribute_predictions(outputs, attribute_labels)
        # print(result)
        break


def test_hybrid_concept():
    cdir = 'VisualConceptReasoning/HybridConcept/'
    concept_variations = ['2c', '25c', 'Resnet', 'ResnetLR', 'Seed1', 'Seed2', 'sigacc', 'sigacc128', 'sigmoid']
    seed = 1

    for cv in concept_variations:
        concept_dict_dir = f'{BASE_PATH}/models/{cdir}{cv}/Seed{seed}/best_model.pth'.replace("\\", '/')

        model = utils.load_model(model_name='HybridConcept', path_to_state_dict=concept_dict_dir)
        model.cuda()
        model.eval()

        path_to_yaml = '/'.join(concept_dict_dir.split('/')[:-1] + ['args.yaml'])
        args = utils.load_model_args_as_dict(path_to_yaml)

        data_dir = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl'.replace("\\", '/')
        loader = load_data_hybrid([data_dir], use_attr=args['use_attr'], no_img=False, batch_size=2, normalize=True,
                                  num_classes=args['num_classes'])

        correct = 0
        for idx, data in enumerate(loader):
            # data: [img_as_tensor, class_number_as_tensor, ggf. attribute as list]

            if args['use_attr']:
                inputs, labels, attribute_labels = data
                attr_labels_var = torch.autograd.Variable(attribute_labels).float()
                attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var
            else:
                inputs, labels = data

            # cast to float to be able to optimize input tensor
            input_adv = torch.autograd.Variable(inputs.float()).cuda()
            labels_var = torch.autograd.Variable(labels).cuda()

            outputs = model(input_adv)

            matched = utils.hungarian_matching(attr_labels_var, outputs)
            out, counts = torch.unique((matched > 0.5) == (attr_labels_var > 0.5), return_counts=True)
            if torch.where(out)[0].shape[0] > 0:
                correct += counts[torch.where(out)[0][0]]

            print(f'correct{correct}')
            print(correct / (attr_labels_var.shape[0] * 2. * 113))

            # Check for successful/wrong predictions
            # result = utils.check_attribute_predictions(outputs, attribute_labels)
            # print(result)
            break


def generate_concept_predictions(model_name='HybridConcept-CNN_Loss'):
    hybrid_model_dirs = {
        'HybridConcept-CNN_Acc': 'VisualConceptReasoning/HybridConcept/CNN_Acc/Seed',
        'HybridConcept-CNN_Loss': 'VisualConceptReasoning/HybridConcept/CNN_Loss/Seed',
        'HybridIndependent': 'VisualConceptReasoning/HybridIndependent/Seed',
        'HybridConcept-CNN_Large_LossCM_2048': 'VisualConceptReasoning/HybridConcept/CNNLarge_LossCM/Seed1',  # bn_hc = 2048
        'HybridConcept-CNN_Large_LossCM_256': 'VisualConceptReasoning/HybridConcept/CNNLarge_LossCM/Seed2',  # bn_hc = 256
        'HybridConcept-CNN_Large_LossCM_1024': 'VisualConceptReasoning/HybridConcept/CNNLarge_LossCM/Seed3',  # bn_hc = 1024 (+halfed loss?)
        'HybridConcept-CNN_Large_LossCMFM_1024': 'VisualConceptReasoning/HybridConcept/LossCMFN/CNNLarge_LossCMFN/Seed1', # loss 3
        'HybridConcept-CNN_LossCMFM_1024': 'VisualConceptReasoning/HybridConcept/LossCMFN/CNN_LossCMFN/Seed1', # loss 3
        'HybridConcept-CNN_LossCM2FM_2048': 'VisualConceptReasoning/HybridConcept/LossCM2FN/CNN_LossCMFN/Seed2', # loss 4
        'HybridConcept-CNN_LossCM2FM_1024': 'VisualConceptReasoning/HybridConcept/LossCM2FN/CNN_LossCMFN/Seed1'  # loss 4
    }

    #concept_dict_dir = f'{BASE_PATH}/models/{model_dirs[model_name]}{seed}/best_model.pth'.replace("\\", '/')
    concept_dict_dir = f'{BASE_PATH}/models/{hybrid_model_dirs[model_name]}/best_model.pth'.replace("\\", '/')
    model = utils.load_model(model_name=model_name, path_to_state_dict=concept_dict_dir)
    model.eval()
    model.cuda()

    for file in ['train', 'val', 'test']:
        # use the first seed for the args file, since except for seed related arguments, all others are equal
        args = utils.load_model_args_as_dict(f'{BASE_PATH}/models/{hybrid_model_dirs[model_name]}/args.yaml')
        #args = utils.load_model_args_as_dict(f'{BASE_PATH}/models/{model_dirs[model_name]}1/args.yaml')

        data_dir = f'{BASE_PATH}/data/CUB_200_2011/CUB_processed/class_attr_data_10/{file}.pkl'.replace("\\", '/')

        is_train = any(["train" in path for path in [data_dir]])
        img_size = 256

        if is_train:
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                transforms.RandomAffine(degrees=85, translate=(0.2, 0.2), shear=0.3),
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        dataset = CUB_Dataset_Hybrid(pkl_file_paths=[data_dir], use_attr=args['use_attr'], no_img=False,
                                     image_dir='images',
                                     transform=transform, num_classes=args['num_classes'], num_slots=args['n_slots'])
        loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

        list_out = []
        list_attr_labels = []
        list_classes = []
        for idx, data in enumerate(loader):
            # data: [img_as_tensor, class_number_as_tensor, ggf. attribute as list]

            if args['use_attr']:
                inputs, labels, attribute_labels = data
            else:
                inputs, labels = data

            # cast to float to be able to optimize input tensor
            input_adv = torch.autograd.Variable(inputs.float()).cuda()

            # set return attn to True
            outputs = model(input_adv)
            res = outputs.detach().cpu()

            list_out.append(res)
            list_attr_labels.append(attribute_labels.float())
            list_classes.append(labels)

        torch.save(torch.stack(list_out, dim=0),
                   f'{BASE_PATH}/outputs/ConceptModelOutput/{model_name}_{file}_concept_predictions.pt')
        torch.save(torch.stack(list_attr_labels, dim=0),
                   f'{BASE_PATH}/outputs/ConceptModelOutput/{model_name}_{file}_attr_labels.pt')


def test_losses():
    dataset = 'train'
    outputs = torch.reshape(torch.load(f'{BASE_PATH}/outputs/ConceptModelOutput/{dataset}_concept_predictions.pt'),
                            (-1, 2, 113))
    attr_labels = torch.reshape(torch.load(f'{BASE_PATH}/outputs/ConceptModelOutput/{dataset}_attr_labels.pt'),
                                (-1, 2, 113))

    # attempt 1
    """matched = utils.hungarian_matching(attr_labels, preds_attrs=preds)
    bool_labels, cnts_labels = torch.unique(attr_labels > 0.5, return_counts=True)
    bool_preds, cnts_preds = torch.unique(preds > 0.5, return_counts=True)

    # out, counts = torch.unique((matched > 0.5) == (attr_labels > 0.5), return_counts=True)
    false_acc = float(cnts_preds[torch.where(bool_preds == False)].item()) / cnts_labels[torch.where(bool_labels == False)].item()
    true_acc = float(cnts_preds[torch.where(bool_preds)].item()) / cnts_labels[torch.where(bool_labels)].item()

    combined_accs = (false_acc + true_acc) / 2."""

    # attempt 2 / 3
    matched = utils.hungarian_matching(attr_labels, preds_attrs=outputs)
    bool_labels, cnts_labels = torch.unique(attr_labels > 0.5, return_counts=True)
    bool_preds, cnts_preds = torch.unique(matched > 0.5, return_counts=True)

    print(f'pred: {torch.unique(attr_labels > 0.5, return_counts=True)}')
    print(f'matched: {torch.unique(matched > 0.5, return_counts=True)}')

    # attempt 2 acc: number predicted / number all falses/trues
    """false_acc = float(cnts_preds[torch.where(bool_preds == False)].item()) / cnts_labels[
        torch.where(bool_labels == False)].item()
    true_acc = float(cnts_preds[torch.where(bool_preds==True)].item()) / cnts_labels[torch.where(bool_labels==True)].item()"""

    # attempt 3: normalize num_preds by proportion in batch:
    false_acc = float(cnts_preds[torch.where(bool_preds == False)].item()) * cnts_labels[
        torch.where(bool_labels == False)].item() / torch.sum(cnts_labels)
    true_acc = float(cnts_preds[torch.where(bool_preds == True)].item()) * cnts_labels[
        torch.where(bool_labels == True)].item() / torch.sum(cnts_labels)

    """print(f'True: {float(cnts_preds[torch.where(bool_preds==True)].item())}')
    print(f'Denom: {cnts_labels[torch.where(bool_labels==True)].item()}')"""

    combined_accs = (false_acc + true_acc) / 2.

    with mp.Pool(10) as thread_pool:
        loss = utils.hungarian_loss(predictions=outputs, targets=attr_labels, thread_pool=thread_pool)
    """print(f'Loss: {loss}')
    print(f'1-F_Acc: {1-false_acc}')
    print(f'1-T_Acc: {1-true_acc}')"""
    loss += false_acc + true_acc


if __name__ == '__main__':
    model_names = ['HybridConcept-CNN_Large_LossCM_2048', 'HybridConcept-CNN_Large_LossCM_256',
                   'HybridConcept-CNN_Large_LossCM_1024', 'HybridConcept-CNN_Large_LossCMFM_1024',
                   'HybridConcept-CNN_LossCMFM_1024', 'HybridConcept-CNN_LossCM2FM_1024',
                   'HybridConcept-CNN_LossCM2FM_2048']

    """for name in model_names:
        generate_concept_predictions(name)
    exit()"""

    # generate_concept_predictions()
    # hybrid_vis()

    # files = glob.glob(f'{BASE_PATH}/outputs/ConceptModelOutput/*.pt')

    # with mp.Pool(10) as pool:
    #    print(f'Loss : {utils.hungarian_loss(predictions=torch.ones(2000, 2, 113), targets=torch.zeros(2000, 2, 113), thread_pool=pool)*2}')
    print('CM: (tn, fp, fn, tp)')
    print('baseline:')
    for dataset in ['train', 'val', 'test']:
        pred = torch.reshape(
            torch.load(f'{BASE_PATH}/outputs/ConceptModelOutput/{dataset}_concept_predictions.pt'), (-1, 2, 113))
        attr_labels = torch.reshape(
            torch.load(f'{BASE_PATH}/outputs/ConceptModelOutput/{dataset}_attr_labels.pt'), (-1, 2, 113))

        # attempt 1
        matched = utils.hungarian_matching(attr_labels, preds_attrs=pred)

        tn, fp, fn, tp = confusion_matrix(attr_labels.cpu().detach().flatten(start_dim=0).numpy(),
                                          matched.cpu().detach().flatten(start_dim=0).numpy() > 0.5,
                                          normalize='true').flatten()

        print(f'Model: Baseline \t dataset: {dataset}\t CM: {(tn, fp, fn, tp)}')


    print()

    for dataset in ['train', 'val', 'test']:
        for model in model_names:
            pred = torch.reshape(
                torch.load(f'{BASE_PATH}/outputs/ConceptModelOutput/{model}_{dataset}_concept_predictions.pt'),
                (-1, 2, 113))
            attr_labels = torch.reshape(
                torch.load(f'{BASE_PATH}/outputs/ConceptModelOutput/{model}_{dataset}_attr_labels.pt'), (-1, 2, 113))

            # attempt 1
            matched = utils.hungarian_matching(attr_labels, preds_attrs=pred)

            tn, fp, fn, tp = confusion_matrix(attr_labels.cpu().detach().flatten(start_dim=0).numpy(),
                                              matched.cpu().detach().flatten(start_dim=0).numpy() > 0.5,
                                              normalize='true').flatten()

            print(f'Model: {model}\t dataset: {dataset}\t CM: {(tn, fp, fn, tp)}')
        print()

    # determine class imbalance
    # give weights to bceloss
    # calculate this loss
    # add f1 score instead of top5 accuracy
    loss = torch.nn.BCELoss()
    print(loss(pred, attr_labels))
