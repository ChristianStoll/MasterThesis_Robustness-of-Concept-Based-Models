import torch
import torchvision.transforms as transforms
from config import COCO_INSTANCE_CATEGORY_NAMES as coco_names
from config import BIRD_IDX

import matplotlib.pyplot as plt
import numpy as np


def index(tensor: torch.Tensor, value, ith_match: int = 0) -> torch.Tensor:
    """
    Returns generalized index (i.e. location/coordinate) of the first occurence of value
    in Tensor. For flat tensors (i.e. arrays/lists) it returns the indices of the occurrences
    of the value you are looking for. Otherwise, it returns the "index" as a coordinate.
    If there are multiple occurences then you need to choose which one you want with ith_index.
    e.g. ith_index=0 gives first occurence.

    Reference: https://stackoverflow.com/a/67175757/1601580
    :return:
    """
    # bool tensor of where value occurred
    places_where_value_occurs = (tensor == value)
    # get matches as a "coordinate list" where occurence happened
    matches = (tensor == value).nonzero()  # [number_of_matches, tensor_dimension]
    if matches.size(0) == 0:  # no matches
        return -1
    else:
        # get index/coordinate of the occurence you want (e.g. 1st occurence ith_match=0)
        index = matches[ith_match]
        return index


class FullBNModel(torch.nn.Module):
    """
        self defined full model combining X->C and C->Y model_templates
    """

    def __init__(self, concept_model, end_model, use_sigmoid, return_attributes):
        super(FullBNModel, self).__init__()
        self.concept_model = concept_model
        self.end_model = end_model
        self.use_sigmoid = use_sigmoid
        self.return_attributes = return_attributes

    def forward(self, x):
        attributes = torch.cat(self.concept_model(x), dim=1)
        if self.use_sigmoid:
            stage2_inputs = torch.nn.Sigmoid()(attributes)
        else:
            stage2_inputs = attributes
        class_outputs = self.end_model(stage2_inputs)

        if self.return_attributes:
            return class_outputs, stage2_inputs
        else:
            return class_outputs


class FullHybridModel(torch.nn.Module):
    """
            self defined full model combining X->C and C->Y model_templates
        """

    def __init__(self, concept_model, end_model, binarize_attr=False, return_attributes=False):
        super(FullHybridModel, self).__init__()
        self.concept_model = concept_model
        self.end_model = end_model
        self.binarize_attr = binarize_attr
        self.return_attributes = return_attributes

    def forward(self, x):
        attr_outputs = self.concept_model(x)

        if self.binarize_attr:
            stage2_inputs = attr_outputs.where(x > 0.5, 1., 0.)
        else:
            stage2_inputs = attr_outputs
        class_outputs = self.end_model(stage2_inputs)

        if self.return_attributes:
            return class_outputs, attr_outputs
        else:
            return class_outputs


class FullMaskStage2Model(torch.nn.Module):
    def __init__(self, concept_model, end_model, use_sigmoid=True, return_attributes=False):
        super(FullMaskStage2Model, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.concept_model = concept_model
        self.end_model = end_model
        self.sigmoid = torch.nn.Sigmoid()
        self.return_attributes = return_attributes

    def forward(self, x):
        x = self.concept_model(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        attr = x.clone()
        x = self.end_model(x)

        if self.return_attributes:
            return x, attr
        return x


class SelectOutput(torch.nn.Module):
    """
        self defined output layer to return only one vector/image (no list/tuple or anything like that)
    """

    def __init__(self):
        super(SelectOutput, self).__init__()

    def forward(self, x):
        return x[0]


def preprocess_mask_outputs(inputs, model, threshold, img_id):
    """
    Function to preprocess the outputs of MaskRCNN to get the relevant images required for the next step in the pipeline
    :param inputs:
    :param model:
    :return:
    """
    transform_size = transforms.Resize((224, 224))
    # input format: NxCxHxW
    model.eval()
    outputs, _, _ = model(inputs)

    segmentation_masks = []
    stage2_inputs_crop_box = []
    stage2_inputs_crop_segment = []
    stage2_segcrop_segmented = []
    stage2_bb = []
    stage2_seg_bb = []
    stage2_bbcrop_segmented = []
    stage2_img_id_to_save = []

    # save certainties for bird predictions
    top1_bird_score, avg_bird_score = 0, 0

    # deal with "no birds found" (especially/only for adversarial attacks)
    cnt_nobirds = 0
    img_no_birds, attr_labels_no_birds, labels_no_birds, no_birds_img_id_to_save = [], [], [], []
    for i in range(len(outputs)):  # process a full batch
        """threshold scores to get certainty gt 'threshold', get number of remaining scores 
        (outputs are sorted by descending scores)"""
        threshold_idx = len(
            [index(outputs[i]['scores'], x) for x in outputs[i]['scores'] if x > threshold])

        # apply threshold on outputs
        scores = outputs[i]["scores"][:threshold_idx]
        mask_labels = outputs[i]['labels'][:threshold_idx]
        bboxes = outputs[i]['boxes'][:threshold_idx]
        seg_masks = outputs[i]['masks'][:threshold_idx]

        # scores remain sorted in descending order
        bird_scores = [s for l, s in zip(mask_labels, scores) if l == BIRD_IDX]

        if len(bird_scores) > 0:
            top1_bird_score += bird_scores[0].item()
            avg_bird_score += sum(bird_scores) / float(len(bird_scores))

        if BIRD_IDX in mask_labels:
            # extract bird predictions for bounding_boxes
            bird_boxes = torch.stack([bb for label, bb in zip(mask_labels, bboxes) if label == BIRD_IDX])
            # extract most outer coordinates of bounding boxes - required for calculating the IoU
            bounding_box = torch.tensor(
                [bird_boxes[:, 0].min(), bird_boxes[:, 1].min(), bird_boxes[:, 2].max(), bird_boxes[:, 3].max()]
            ).type(torch.IntTensor)

            # cut out bounding box area of image and segmentation mask
            stage2_inputs_crop_box.append(inputs[i, :, bounding_box[1].item():bounding_box[3].item() + 1,
                                          bounding_box[0].item():bounding_box[2].item() + 1])

            # calculate bird mask of img
            segmentation_masks.append(
                torch.stack([m > 0.5 for label, m in zip(mask_labels, seg_masks) if label == BIRD_IDX]).sum(axis=0))

            """use bb crop after applying segmentation mask 
                (this part cannot be done after saving the img, since the coordinate systems differ after cropping!)"""
            stage2_bbcrop_segmented.append(
                (inputs[i] * (segmentation_masks[-1] > 0.5))[:, bounding_box[1].item():bounding_box[3].item() + 1,
                bounding_box[0].item():bounding_box[2].item() + 1])

            # crop image around edges of segmentation: outer coordinates of segmentation
            seg_idxs = torch.where(segmentation_masks[-1][0] > 0)
            seg_box = torch.tensor([seg_idxs[0].min(), seg_idxs[1].min(), seg_idxs[0].max(), seg_idxs[1].max()])

            stage2_inputs_crop_segment.append(inputs[i, :, seg_box[0].item():seg_box[2].item() + 1,
                                              seg_box[1].item():seg_box[3].item() + 1])

            stage2_segcrop_segmented.append(
                (inputs[i] * (segmentation_masks[-1] > 0.5))[:, seg_box[0].item():seg_box[2].item() + 1,
                seg_box[1].item():seg_box[3].item() + 1])

            # also save bounding boxes and current image id
            stage2_bb.append(bounding_box)
            stage2_seg_bb.append(seg_box)
            stage2_img_id_to_save.append(img_id[i])
        else:
            cnt_nobirds += 1
            img_no_birds.append(inputs[i])
            no_birds_img_id_to_save.append(img_id[i])

    stage2_inputs_crop_box = [transform_size(x) for x in stage2_inputs_crop_box]
    stage2_inputs_crop_segment = [transform_size(x) for x in stage2_inputs_crop_segment]
    segmentation_masks = \
        [transform_size(torch.where(x > 0, 1., 0.)) for x in segmentation_masks]
    stage2_bbcrop_segmented = [transform_size(x) for x in stage2_bbcrop_segmented]

    img_no_birds = [transform_size(x) for x in img_no_birds]

    # segmentation_masks contains the segmentation masks for the original images
    # stage2_segmentation_mask is the mask for the crop_segment image
    processed_outputs = {}
    if len(stage2_inputs_crop_box) > 0:
        processed_outputs.update(
            {"stage2_inputs_crop_box": torch.stack(stage2_inputs_crop_box)})  # crop around bounding box
        processed_outputs.update({"stage2_bbcrop_segmented": torch.stack(
            stage2_bbcrop_segmented)})  # apply segmentation mask, crop around BB
        processed_outputs.update({"stage2_inputs_crop_segment": torch.stack(
            stage2_inputs_crop_segment)})  # crop at borders of segmentation mask
        processed_outputs.update({
            "stage2_inputs_crop_segment_segmented": stage2_segcrop_segmented})  # crop around borders of seg mask, apply segmentation
        processed_outputs.update(
            {"stage2_segmentation_bb": torch.stack(stage2_seg_bb)})  # bounding box spanned at borders of segmentation
        processed_outputs.update(
            {"segmentation_mask": torch.stack(segmentation_masks)})  # segmentation mask from mask_model
        processed_outputs.update({"stage2_bb": torch.stack(stage2_bb)})  # bounding boxes
        processed_outputs.update({"stage2_img_id_to_save": stage2_img_id_to_save})  # ids of images with 'bird' as class
    processed_outputs.update({"cnt_nobirds": cnt_nobirds})  # number of images without 'bird' as class
    if cnt_nobirds > 0:
        processed_outputs.update({"img_no_birds": torch.stack(img_no_birds)})  # images without 'bird' as class
        processed_outputs.update(
            {"no_birds_img_id_to_save": no_birds_img_id_to_save})  # img_ids of images without 'bird' as class

    processed_outputs.update({"bird_top1_certainty": top1_bird_score})
    processed_outputs.update({"bird_avg_certainty": avg_bird_score})
    return processed_outputs


class MLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim):
        super(MLP, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
        self.expand_dim = expand_dim
        if self.expand_dim > 0:
            self.linear = torch.nn.Linear(input_dim, expand_dim)
            self.activation = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(expand_dim, num_classes)  # softmax is automatically handled by loss function

    def forward(self, x):
        x = self.linear(x)
        if hasattr(self, 'expand_dim') and self.expand_dim > 0:
            x = self.activation(x)
            x = self.linear2(x)
        return x


class Normalization(torch.nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(Normalization, self).__init__()
        self.mean = mean
        self.std = std
        self.transform = transforms.Normalize(mean=mean, std=std)

    def forward(self, x):
        y = x.clone()
        for i in range(x.shape[0]):
            y[i] = self.transform(x[i])
        return y


class InverseNormalization(torch.nn.Module):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[2, 2, 2]):
        super(InverseNormalization, self).__init__()
        self.inverse_mean = - torch.tensor(mean) / torch.tensor(std)
        self.inverse_std = 1 / torch.tensor(std)
        self.inv_trans = transforms.Normalize(mean=self.inverse_mean, std=self.inverse_std)

    def forward(self, x):
        return self.inv_trans(x[0]).unsqueeze(0)
