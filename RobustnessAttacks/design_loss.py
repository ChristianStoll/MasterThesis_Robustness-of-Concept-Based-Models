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


def design_conf_matrix_based_loss():
    dataset = 'train'
    outputs = torch.reshape(torch.load(f'{BASE_PATH}/outputs/ConceptModelOutput/{dataset}_concept_predictions.pt'),
                            (-1, 2, 113))
    attr_labels = torch.reshape(torch.load(f'{BASE_PATH}/outputs/ConceptModelOutput/{dataset}_attr_labels.pt'),
                                (-1, 2, 113))

    matched_preds = utils.hungarian_matching(attr_labels, outputs)

    # add f1 score instead of top5 accuracy
    f1 = f1_score(attr_labels.cpu().detach().flatten(start_dim=0).numpy(),
                  matched_preds.cpu().detach().flatten(start_dim=0).numpy() > 0.5)

    cm = confusion_matrix(attr_labels.cpu().detach().flatten(start_dim=0).numpy(),
                          matched_preds.cpu().detach().flatten(start_dim=0).numpy() > 0.5,
                          normalize='true').flatten()

    tn, fp, fn, tp = cm

    print(f'f1: {f1}')
    print(f'cm: {cm}')
    print(cm.round(3)) # tn, fp, fn, tp
    print('tn, fp, fn, tp')
    print((tn, fp, fn, tp))

    print()
    print(f'loss idea:')
    loss = fp + fn
    print(loss)



if __name__ == '__main__':
    design_conf_matrix_based_loss()
