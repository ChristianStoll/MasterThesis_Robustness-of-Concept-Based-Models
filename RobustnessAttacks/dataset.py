"""
General utils for training, evaluation and data loading
"""
import os
import pickle
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
import torch
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader
import random
import glob
import pickle
from config import N_ATTRIBUTES, n_attributes, BASE_PATH


class CUBDataset_Bottleneck(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr=2,
                 transform=None, use_segmentation=False):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr
        self.use_segmentation = use_segmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        try:
            """if self.use_cub:
                idx = img_path.split('/').index('CUB_200_2011')
                if self.image_dir != 'images':
                    img_path = '/'.join([self.image_dir] + img_path.split('/')[idx + 1:])
                    img_path = img_path.replace('images/', '')
                else:
                    img_path = '/'.join(img_path.split('/')[idx:])
            else:
                idx = img_path.split('/').index('CUB_200_2011')
                img_path = '/'.join([self.image_dir] + img_path.split('/')[idx + 2:])"""
            idx = img_path.split('/').index('CUB_200_2011')
            if self.image_dir != 'images':
                img_path = '/'.join([self.image_dir] + img_path.split('/')[idx + 1:])
                img_path = img_path.replace('images/', '')
            else:
                img_path = '/'.join(img_path.split('/')[idx:])
            img = Image.open('data/' + img_path).convert('RGB')

        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            # img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img_path = '/'.join(img_path_split[:2] + img_path_split[2:])
            img = Image.open(BASE_PATH + '/data/' + img_path).convert('RGB')

        if self.use_segmentation:
            idx = img_data['img_path'].split('/').index('CUB_200_2011')
            segmentation_path = '/'.join(['data'] + img_data['img_path'].split('/')[idx:]) \
                .replace('images', 'segmentations').replace('.jpg', '.png')
            segmentation = Image.open(segmentation_path).convert('L')
            transform_seg = transforms.Compose([
                transforms.CenterCrop(299),
                transforms.ToTensor(),  # implicitly divides by 255
            ])
            segmentation = transform_seg(segmentation)

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']

            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((N_ATTRIBUTES, self.n_class_attr))
                    one_hot_attr_label[np.arange(N_ATTRIBUTES), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label
        elif self.use_segmentation:
            return img, class_label, segmentation
        else:
            return img, class_label


class TravellingBirds_Dataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the TravellingBirds dataset
    """

    def __init__(self, pkl_file_paths, image_dir, num_classes=200, transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        image_dir: default = 'images'. Will be append to the parent dir
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.image_dir = image_dir
        self.num_classes = num_classes

        if self.num_classes < 200:
            self._update_data()

    def _update_data(self):
        tmp = []
        for el in self.data:
            if el['class_label'] <= self.num_classes:
                tmp.append(el)
        self.data = tmp.copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        idx = img_path.split('/').index('CUB_200_2011')
        img_path = '/'.join([self.image_dir] + img_path.split('/')[idx + 2:])
        img = Image.open(f'{BASE_PATH}/data/AdversarialData/{img_path}').convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        return img, class_label


class CUB_Dataset_Scouter(Dataset):
    def __init__(self, dataset_dir, num_classes, train=True, transform=None):
        super(CUB_Dataset_Scouter, self).__init__()
        self.root = dataset_dir
        self.num = num_classes
        self.train = train
        self.transform_ = transform
        self.classes_file = os.path.join(self.root, 'classes.txt')  # <class_id> <class_name>
        self.image_class_labels_file = os.path.join(self.root, 'image_class_labels.txt')  # <image_id> <class_id>
        self.images_file = os.path.join(self.root, 'images.txt')  # <image_id> <image_name>
        self.train_test_split_file = os.path.join(self.root, 'train_test_split.txt')  # <image_id> <is_training_image>
        self.bounding_boxes_file = os.path.join(self.root, 'bounding_boxes.txt')  # <image_id> <x> <y> <width> <height>

        self._train_ids = []
        self._test_ids = []
        self._image_id_label = {}
        self._train_path_label = []
        self._test_path_label = []

        self._train_test_split()
        self._get_id_to_label()
        self._get_path_label()

    def _train_test_split(self):
        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if label == '1':
                self._train_ids.append(image_id)
            elif label == '0':
                self._test_ids.append(image_id)
            else:
                raise Exception('label error')

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            self._image_id_label[image_id] = class_id

    def _get_path_label(self):
        for line in open(self.images_file):
            image_id, image_name = line.strip('\n').split()
            if int(image_name[:3]) > self.num:
                if image_id in self._train_ids:
                    self._train_ids.pop(self._train_ids.index(image_id))
                else:
                    self._test_ids.pop(self._test_ids.index(image_id))
                continue
            label = self._image_id_label[image_id]
            if image_id in self._train_ids:
                self._train_path_label.append((image_name, label))
            else:
                self._test_path_label.append((image_name, label))

    def __getitem__(self, index):
        if self.train:
            image_name, label = self._train_path_label[index]
        else:
            image_name, label = self._test_path_label[index]
        image_path = os.path.join(self.root, 'images', image_name)

        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        label = int(label) - 1
        label = torch.from_numpy(np.array(label)).type(torch.LongTensor)
        if self.transform_ is not None:
            img = self.transform_(img)
        return img, label

    def __len__(self):
        if self.train:
            return len(self._train_ids)
        else:
            return len(self._test_ids)


class CUB_Dataset(Dataset):
    def __init__(self, dataset_dir, num_classes, dataset_type='test', transform=None):
        super(CUB_Dataset, self).__init__()
        self.root = dataset_dir
        self.num_classes = num_classes
        self.dataset_type = dataset_type
        self.transform_ = transform
        self.classes_file = os.path.join(self.root, 'classes.txt').replace("\\", '/')  # <class_id> <class_name>
        self.image_class_labels_file = os.path.join(self.root, 'image_class_labels.txt').replace("\\",
                                                                                                 '/')  # <image_id> <class_id>
        self.images_file = os.path.join(self.root, 'images.txt').replace("\\", '/')  # <image_id> <image_name>
        self.train_test_split_file = os.path.join(self.root, 'train_test_split.txt').replace("\\",
                                                                                             '/')  # <image_id> <is_training_image>
        self.bounding_boxes_file = os.path.join(self.root, 'bounding_boxes.txt').replace("\\",
                                                                                         '/')  # <image_id> <x> <y> <width> <height>

        self._train_ids = []
        self._val_ids = []
        self._test_ids = []
        self._image_id_label = {}
        self._train_path_label = []
        self._val_path_label = []
        self._test_path_label = []

        self.val_ratio = 0.2
        # create train, val, test files and train_val split if necessary
        files = '\t'.join(glob.glob(os.path.join(self.root, '*.txt').replace("\\", '/')))
        if not ('train.txt' in files and 'val.txt' in files and 'test.txt' in files):
            self._train_val_test_split()  # create files containing train, val, test image ids

        self._read_image_ids()
        self._get_id_to_label()
        self._get_path_label()

    def _train_val_test_split(self):
        train_val_ids = []
        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if label == '1':
                train_val_ids.append(image_id)
            elif label == '0':
                self._test_ids.append(image_id)
            else:
                raise Exception('label error')

        random.shuffle(train_val_ids)
        split = int(self.val_ratio * len(train_val_ids))
        self._train_ids = train_val_ids[split:]
        self._val_ids = train_val_ids[: split]

        for ids, filename in zip([self._train_ids, self._val_ids, self._test_ids], ['train', 'val', 'test']):
            with open(os.path.join(self.root, f'{filename}.txt').replace("\\", '/'), 'w') as f:
                f.write('\n'.join(ids))
                f.close()

    def _read_image_ids(self):
        # reset list entries
        self._train_ids, self._val_ids, self._test_ids = [], [], []
        # read image ids
        for filename in ['train', 'val', 'test']:
            if filename == 'train':
                for line in open(os.path.join(self.root, f'{filename}.txt').replace("\\", '/')):
                    self._train_ids.append(line.strip('\n'))
            elif filename == 'val':
                for line in open(os.path.join(self.root, f'{filename}.txt').replace("\\", '/')):
                    self._val_ids.append(line.strip('\n'))
            else:
                for line in open(os.path.join(self.root, f'{filename}.txt').replace("\\", '/')):
                    self._test_ids.append(line.strip('\n'))

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            self._image_id_label[image_id] = class_id

    def _get_path_label(self):
        for line in open(self.images_file):
            image_id, image_name = line.strip('\n').split()
            if int(image_name[:3]) > self.num_classes:
                if image_id in self._train_ids:
                    self._train_ids.pop(self._train_ids.index(image_id))
                elif image_id in self._val_ids:
                    self._val_ids.pop(self._val_ids.index(image_id))
                else:
                    self._test_ids.pop(self._test_ids.index(image_id))
                continue
            label = self._image_id_label[image_id]
            if image_id in self._train_ids:
                self._train_path_label.append((image_name, label))
            elif image_id in self._val_ids:
                self._val_path_label.append((image_name, label))
            else:
                self._test_path_label.append((image_name, label))

    def __getitem__(self, index):
        if self.dataset_type == 'train':
            image_name, label = self._train_path_label[index]
        elif self.dataset_type == 'val':
            image_name, label = self._val_path_label[index]
        else:
            image_name, label = self._test_path_label[index]
        image_path = os.path.join(self.root, 'images', image_name).replace("\\", '/')

        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        label = int(label) - 1
        label = torch.from_numpy(np.array(label)).type(torch.LongTensor)
        if self.transform_ is not None:
            img = self.transform_(img)
        return img, label

    def __len__(self):
        if self.dataset_type == 'train':
            return len(self._train_ids)
        elif self.dataset_type == 'val':
            return len(self._val_ids)
        else:
            return len(self._test_ids)


class CUB_Dataset_Hybrid(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, image_dir, dataset_dir='data', num_slots=0, num_classes=200,
                 transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        image_dir: default = 'images'. Will be append to the parent dir
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.image_dir = image_dir
        self.dataset_dir = dataset_dir
        self.num_slots = num_slots
        self.num_classes = num_classes

        if self.num_classes < 200:
            self._update_data()

    def _update_data(self):
        tmp = []
        for el in self.data:
            if el['class_label'] <= self.num_classes:
                tmp.append(el)
        self.data = tmp.copy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            if self.image_dir != 'images':
                img_path = '/'.join([BASE_PATH, self.image_dir] + img_path.split('/')[idx + 1:])
                img_path = img_path.replace('images/', '')
            else:
                img_path = '/'.join(img_path.split('/')[idx:])
            img = Image.open('/'.join([BASE_PATH, self.dataset_dir, img_path])).convert('RGB')
        except:
            img_path_split = img_path.split('/')

            img_path = '/'.join(img_path_split[:2] + img_path_split[2:])
            img = Image.open('/'.join([self.dataset_dir, img_path])).convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            attr_label = img_data['attribute_label']
            # add 1 as object label
            slot_attr_label = attr_label.copy()
            slot_attr_label.append(1)
            # num_attr 0's and an additional 0 as background label -> 113
            # creates a tensor with the attribute labels and num_slots lists of background labels
            """if self.num_slots > 1:
                attr_label = np.array([slot_attr_label] + [[0] * (len(attr_label)+1) for _ in range(self.num_slots-1)])
            else:
                print('At least 2 slots are required!')
                exit()"""

            attr_label = np.array([slot_attr_label, [0] * (len(attr_label) + 1)])
            # randomly shuffles the rows of the array -> especially relevant for independent training
            if self.is_train:
                np.random.shuffle(attr_label)
            # attr_label = torch.from_numpy(attr_label)

        if self.use_attr and self.no_img:
            return attr_label, class_label
        elif self.use_attr and not self.no_img:
            return img, class_label, attr_label
        else:
            return img, class_label


class CUB_Dataset_MaskRCNN(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, dataset_dir='data', num_classes=200,
                 transform=None, return_img_id=False, add_other_mask=False):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        image_dir: default = 'images'. Will be append to the parent dir
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.dataset_dir = dataset_dir
        self.num_classes = num_classes
        self.return_img_id = return_img_id
        self._map_id_to_img_name()
        self.add_other_mask = add_other_mask

        self._load_bounding_boxes()
        self.coco_classes = 91

        if self.num_classes < 200:
            self._update_data()

    def _update_data(self):
        tmp = []
        for el in self.data:
            if el['class_label'] <= self.num_classes:
                tmp.append(el)
        self.data = tmp.copy()

    def _map_id_to_img_name(self):
        self.img_to_id = {}
        with open('/'.join([BASE_PATH, self.dataset_dir, '/CUB_200_2011/images.txt'])) as f:
            for line in f:
                (key, val) = line.split()
                self.img_to_id[val] = int(key)

    def _load_bounding_boxes(self):
        bb_path = '/'.join([BASE_PATH, self.dataset_dir, "/CUB_200_2011/bounding_boxes.txt"]).replace("\\", '/')
        self.bounding_boxes = [line.split()[1:] for line in open(bb_path, 'r')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        id = img_path.split('/').index('CUB_200_2011')
        img_path = '/'.join(img_path.split('/')[id:])
        img_path = '/'.join([BASE_PATH, self.dataset_dir, img_path])

        # load image
        img = Image.open(img_path).convert('RGB')
        # load segmentation
        segmentation_path = img_path.replace('images', 'segmentations').replace('.jpg', '.png')
        seg = Image.open(segmentation_path).convert('L')
        if self.transform:
            img = self.transform(img)
            seg = self.transform(seg)

        if self.add_other_mask:
            segmentation = np.zeros((2, seg.shape[1], seg.shape[2]))
            segmentation[0] = seg
        else:
            segmentation = seg
        segmentation = torch.tensor(segmentation > 0.5, dtype=torch.float)

        class_label = img_data['class_label']
        if self.use_attr:
            attr_label = torch.tensor(img_data['attribute_label'])

        # -1 since bb ids are 1..n instead of 0..n in list
        img_id = self.img_to_id['/'.join(img_path.split('/')[-2:])]
        x, y, w, h = np.array(self.bounding_boxes[img_id - 1], dtype=float)
        bounding_box = np.array([x, y, x + w, y + h])

        target = {"boxes": torch.tensor([bounding_box], dtype=torch.int), "labels": torch.ones(2, dtype=torch.int),
                  "masks": torch.as_tensor(segmentation, dtype=torch.uint8)}

        if self.return_img_id:
            return img, class_label, attr_label, segmentation, bounding_box, img_id, target

        if self.use_attr and self.no_img:
            return attr_label, class_label
        elif self.use_attr and not self.no_img:
            return img, class_label, attr_label, segmentation, bounding_box
        else:
            return img, class_label, segmentation, bounding_box


class TravBirds_Dataset_MaskRCNN(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, birds_dir, dataset_dir='data', transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        image_dir: default = 'images'. Will be append to the parent dir
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.birds_dir = birds_dir
        self._map_id_to_img_name()

    def _map_id_to_img_name(self):
        self.img_to_id = {}
        with open('/'.join([BASE_PATH, self.dataset_dir, '/CUB_200_2011/images.txt'])) as f:
            for line in f:
                (key, val) = line.split()
                self.img_to_id[val] = int(key)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        id = img_path.split('/').index('CUB_200_2011')
        img_path = '/'.join(img_path.split('/')[id+1:])
        img_path = '/'.join([BASE_PATH, self.dataset_dir, self.birds_dir, img_path]).replace('images/', '')

        # load image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        img_id = self.img_to_id['/'.join(img_path.split('/')[-2:])]
        return img, img_id


class CUB_Dataset_adversarial_MaskBottleneck(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, attack, epsilon, adv_dataset_dir, crop_type, data_dir='data', transform=None,
                 apply_segmentation=False):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        image_dir: default = 'images'. Will be append to the parent dir
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.data_dir = data_dir
        self.adv_dataset_dir = adv_dataset_dir
        self.crop_type = crop_type
        self.apply_segmentation = apply_segmentation
        self.attack = attack
        self.epsilon = epsilon

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        path = img_data['img_path']
        # Trim unnecessary paths
        id = path.split('/').index('CUB_200_2011')
        path = '/'.join(path.split('/')[id+1:])
        path = '/'.join([BASE_PATH, self.data_dir, self.adv_dataset_dir, self.attack, self.epsilon, path])

        if self.apply_segmentation:
            if self.crop_type == 'cropbb':
                img_path = path.replace('images', 'bbcrops_seg')
            elif self.crop_type == 'segbb':
                img_path = path.replace('images', 'segbbcrops_seg')
            else:
                print('wrong crop type, choose from [cropbb, segbb]')
                exit()
        else:
            if self.crop_type == 'cropbb':
                img_path = path.replace('images', 'bbcrops')
            elif self.crop_type == 'segbb':
                img_path = path.replace('images', 'segbbcrops')
            else:
                print('wrong crop type, choose from [cropbb, segbb]')
                exit()
        # load image
        try:
            img = Image.open(img_path).convert('RGB')
            bird_found = True
        except:
            path_no_birds = path.replace('images', 'nobird')
            img = Image.open(path_no_birds).convert('RGB')
            bird_found = False

        if self.transform:
            img = self.transform(img)

        class_label = img_data['class_label']
        attr_label = torch.tensor(img_data['attribute_label'])

        return img, class_label, attr_label, torch.tensor(bird_found)


class CUB_Dataset_travBirds_MaskBottleneck(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, adv_dataset_dir, crop_type, data_dir='data', transform=None,
                 apply_segmentation=False):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        image_dir: default = 'images'. Will be append to the parent dir
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.data_dir = data_dir
        self.adv_dataset_dir = adv_dataset_dir
        self.crop_type = crop_type
        self.apply_segmentation = apply_segmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        path = img_data['img_path']
        # Trim unnecessary paths
        id = path.split('/').index('CUB_200_2011')
        path = '/'.join(path.split('/')[id+1:])
        path = '/'.join([BASE_PATH, self.data_dir, self.adv_dataset_dir, path])

        if self.apply_segmentation:
            if self.crop_type == 'cropbb':
                img_path = path.replace('images', 'bbcrops_seg')
            elif self.crop_type == 'segbb':
                img_path = path.replace('images', 'segbbcrops_seg')
            else:
                print('wrong crop type, choose from [cropbb, segbb]')
                exit()
        else:
            if self.crop_type == 'cropbb':
                img_path = path.replace('images', 'bbcrops')
            elif self.crop_type == 'segbb':
                img_path = path.replace('images', 'segbbcrops')
            else:
                print('wrong crop type, choose from [cropbb, segbb]')
                exit()
        # load image
        try:
            img = Image.open(img_path).convert('RGB')
            bird_found = True
        except:
            path_no_birds = path.replace('images', 'nobird')
            img = Image.open(path_no_birds).convert('RGB')
            bird_found = False

        if self.transform:
            img = self.transform(img)

        class_label = img_data['class_label']
        attr_label = torch.tensor(img_data['attribute_label'])

        return img, class_label, attr_label, torch.tensor(bird_found)


class CUB_Dataset_MaskBottleneck(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, data_dir='data', num_classes=200, transform=None,
                 crop_type='', apply_segmentation=False, load_segmentation=False):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        image_dir: default = 'images'. Will be append to the parent dir
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.crop_type = crop_type
        self.apply_segmentation = apply_segmentation
        self.load_segmentation = load_segmentation
        self._map_id_to_img_name()

        if self.crop_type == 'labelbb':
            self._load_bounding_boxes()

        if self.num_classes < 200:
            self._update_data()

    def _update_data(self):
        tmp = []
        for el in self.data:
            if el['class_label'] <= self.num_classes:
                tmp.append(el)
        self.data = tmp.copy()

    def _load_bounding_boxes(self):
        bb_path = '/'.join([BASE_PATH, self.data_dir, "/CUB_200_2011/bounding_boxes.txt"]).replace("\\", '/')
        self.bounding_boxes = [line.split()[1:] for line in open(bb_path, 'r')]

    def __len__(self):
        return len(self.data)

    def _map_id_to_img_name(self):
        self.img_to_id = {}
        with open('/'.join([BASE_PATH, self.data_dir, '/CUB_200_2011/images.txt'])) as f:
            for line in f:
                (key, val) = line.split()
                self.img_to_id[val] = int(key)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        path = img_data['img_path']
        # Trim unnecessary paths
        id = path.split('/').index('CUB_200_2011')
        path = '/'.join(path.split('/')[id:])
        path = '/'.join([BASE_PATH, self.data_dir, path])

        """# try to load img if it exists
        if self.crop_type == 'cropbb':
            img_path = path.replace('images', 'bbcrops').replace('CUB_200_2011', 'CUB_Cropped')
        elif self.crop_type == 'segbb':
            img_path = path.replace('images', 'segbbcrops').replace('CUB_200_2011', 'CUB_Cropped')
        else:
            img_path = path
        # load image
        img = Image.open(img_path).convert('RGB')

        if self.apply_segmentation:
            if self.crop_type == 'segbb':
                segmentation_path = path.replace(f'images', 'segmasks').replace('CUB_200_2011', 'CUB_Cropped')
                seg = Image.open(segmentation_path).convert('L')
            elif self.crop_type == 'labelbb'
                segmentation_path = path.replace('images', 'segmentations').replace('.jpg', '.png')\
                    .replace('CUB_200_2011', 'CUB_Cropped')
                seg = Image.open(segmentation_path).convert('L')
            img = transforms.ToTensor()(img) * transforms.ToTensor()(seg)
        else:
            img = transforms.ToTensor()(img)"""

        if self.apply_segmentation:
            if self.crop_type == 'cropbb':
                img_path = path.replace('images', 'bbcrops_seg').replace('CUB_200_2011', 'CUB_Cropped')
            elif self.crop_type == 'segbb':
                img_path = path.replace('images', 'segbbcrops_seg').replace('CUB_200_2011', 'CUB_Cropped')
            elif self.crop_type == 'labelbb':
                img_path = path.replace('images', 'CUB_black').replace('CUB_200_2011', 'AdversarialData')
            else:
                print('wrong crop type, choose from [cropbb, segbb, labelbb]')
                exit()
        else:
            if self.crop_type == 'cropbb':
                img_path = path.replace('images', 'bbcrops').replace('CUB_200_2011', 'CUB_Cropped')
            elif self.crop_type == 'segbb':
                img_path = path.replace('images', 'segbbcrops').replace('CUB_200_2011', 'CUB_Cropped')
            elif self.crop_type == 'labelbb':
                img_path = path

            else:
                print('wrong crop type, choose from [cropbb, segbb, labelbb]')
                exit()
        # load image
        img = Image.open(img_path).convert('RGB')

        """if self.crop_type == 'labelbb':  # crop original image with annotated bounding box
            img_id = self.img_to_id['/'.join(img_path.split('/')[-2:])]
            x, y, w, h = np.array(self.bounding_boxes[img_id - 1], dtype=float)
            # bounding_box = np.array([x, y, x + w, y + h])
            img = img[:, int(x):int(x + w + 1), int(y):int(y + h + 1)]"""

        if self.transform:
            img = self.transform(img)

        if self.load_segmentation:
            if self.crop_type == 'segbb':
                seg_path = img_path.replace('segbbcrops_seg', 'segmasks')
                segmentation = transforms.ToTensor()(Image.open(seg_path).convert('L'))
            elif self.crop_type == 'cropbb':
                segmentation = torch.where(img == 0., torch.zeros(img.shape), torch.ones(img.shape))
            else:
                print('This crop type does not exist')
                exit()

        class_label = img_data['class_label']
        attr_label = torch.tensor(img_data['attribute_label'])

        if self.load_segmentation:
            return img, class_label, attr_label, segmentation

        if self.use_attr and self.no_img:
            return attr_label, class_label
        elif self.use_attr and not self.no_img:
            return img, class_label, attr_label
        else:
            return img, class_label


def load_data_cb(pkl_paths, use_attr, no_img, batch_size, uncertain_label=False, n_class_attr=2, image_dir='images',
                 normalize=True, use_segmentation=False, num_classes=200, resol=299):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
    """
    if normalize:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(),  # implicitly divides by 255
        ])

    if num_classes == 200:
        dataset = \
            CUBDataset_Bottleneck(pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform,
                                  use_segmentation=use_segmentation)
    else:
        # extract path from pkl_path
        idx = pkl_paths[0].split('/').index('CUB_200_2011')
        data_dir = '/'.join(pkl_paths[0].split('/')[:idx + 1])
        dataset = CUB_Dataset(dataset_dir=data_dir, num_classes=num_classes, dataset_type='test', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)


def load_data_scouter(data_dir, num_classes, batch_size, train=False, img_size=260, normalize=False):
    if normalize:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    dataset = CUB_Dataset_Scouter(dataset_dir=data_dir, num_classes=num_classes, train=train, transform=transform)

    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)


def load_data_vit(data_dir, num_classes, batch_size, dataset_type='test', img_size=224, normalize=False):
    if dataset_type == 'train':
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=(0.4, 0.8), contrast=0.5, saturation=(0.5, 1.5)),
            transforms.RandomAffine(degrees=85, translate=(0.2, 0.2), shear=0.3),
            transforms.RandomResizedCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif dataset_type == 'val':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        if normalize:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
    dataset = CUB_Dataset(dataset_dir=data_dir, num_classes=num_classes, dataset_type=dataset_type, transform=transform)
    return DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)


def load_data_hybrid(pkl_paths, use_attr, no_img, batch_size, image_dir='images', normalize=True, is_train=False,
                     num_slots=2, num_classes=200, img_size=256):
    if is_train:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            # transforms.RandomAffine(degrees=85, translate=(0.2, 0.2), shear=0.3),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        if normalize:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # implicitly divides by 255
            ])

    dataset = CUB_Dataset_Hybrid(pkl_file_paths=pkl_paths, use_attr=use_attr, no_img=no_img, image_dir=image_dir,
                                 transform=transform, num_classes=num_classes, num_slots=num_slots)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=is_train)


def load_travelling_birds(pkl_paths, batch_size, img_size, image_dir='', num_classes=200, isConceptBottleneck=False):
    if not isConceptBottleneck:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # ConceptBottleneck models center crop instead of resizing and use different mean/std during the transformation
        transform = transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
        ])

    dataset = TravellingBirds_Dataset(pkl_paths, image_dir=image_dir, transform=transform, num_classes=num_classes)
    return DataLoader(dataset, batch_size=batch_size)


def load_data_MaskRCNN(pkl_paths, use_attr, no_img, batch_size, num_classes=200, return_img_id=False,
                       add_other_mask=False):
    transform = transforms.ToTensor()

    dataset = CUB_Dataset_MaskRCNN(pkl_file_paths=pkl_paths, use_attr=use_attr, no_img=no_img,
                                   transform=transform, num_classes=num_classes, return_img_id=return_img_id,
                                   add_other_mask=add_other_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def load_data_TravBirds_MaskRCNN(pkl_paths, batch_size, birds_dir):
    transform = transforms.ToTensor()

    dataset = TravBirds_Dataset_MaskRCNN(pkl_file_paths=pkl_paths, transform=transform, birds_dir=birds_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def load_data_MaskBottleneck(pkl_paths, use_attr, no_img, batch_size, normalize=True, is_train=False, num_classes=200,
                             img_size=224, crop_type='labelbb', apply_segmentation=False, load_segmentation=False):
    if is_train:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        if normalize:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])

    dataset = CUB_Dataset_MaskBottleneck(pkl_file_paths=pkl_paths, use_attr=use_attr, no_img=no_img,
                                         transform=transform, num_classes=num_classes, crop_type=crop_type,
                                         apply_segmentation=apply_segmentation, load_segmentation=load_segmentation)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=is_train)


def load_data_mask_stage2_adv(pkl_paths, batch_size, crop_type, attack, epsilon, adv_dataset_dir,
                              apply_segmentation=False, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CUB_Dataset_adversarial_MaskBottleneck(pkl_file_paths=pkl_paths, transform=transform, crop_type=crop_type,
                                                     apply_segmentation=apply_segmentation, attack=attack,
                                                     epsilon=epsilon, adv_dataset_dir=adv_dataset_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def load_data_mask_stage2_travBirds(pkl_paths, batch_size, crop_type, adv_dataset_dir, apply_segmentation=False,
                                    img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CUB_Dataset_travBirds_MaskBottleneck(pkl_file_paths=pkl_paths, transform=transform, crop_type=crop_type,
                                                   apply_segmentation=apply_segmentation,
                                                   adv_dataset_dir=adv_dataset_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def find_class_imbalance_slots(pkl_file, multiple_attr=False, n_slots=2, attr_idx=-1):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """
    imbalance_ratio = []
    data = pickle.load(open(pkl_file, 'rb'))
    n = len(data)
    n_attr = len(data[0]['attribute_label'])
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in data:
        labels = d['attribute_label']
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)

    # imbalance normalization/calculation
    n_ones.append(n)
    total.append(n * n_slots)
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j] / n_ones[j] - 1)
    if not multiple_attr:  # e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return torch.Tensor(imbalance_ratio)


def find_class_imbalance(pkl_file, multiple_attr=False, attr_idx=-1):
    """
    Code from ConceptBottleneck paper
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """
    imbalance_ratio = []
    data = pickle.load(open(pkl_file, 'rb'))
    n = len(data)
    n_attr = len(data[0]['attribute_label'])
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in data:
        labels = d['attribute_label']
        if multiple_attr:  # -> True during training
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)

    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j] / n_ones[j] - 1)
    if not multiple_attr:  # e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return torch.Tensor(imbalance_ratio)


if __name__ == "__main__":
    os.chdir('..')
    BASE_PATH = os.getcwd().replace("\\", '/')

    data_dir = (BASE_PATH + '/data/CUB_200_2011/CUB_processed/class_attr_data_10/test.pkl').replace("\\", '/')
    loader = load_data_MaskBottleneck([data_dir], use_attr=True, no_img=False, batch_size=5, num_classes=200)

    for i, data in enumerate(loader):
        img, class_label, attr_label = data
        print(f'{i}, shape: {img.shape}')
        print(f'attribute labels: {len(attr_label)}')
        print(f'1 att {attr_label.shape}')
        print(attr_label)
        print(img)
        break
