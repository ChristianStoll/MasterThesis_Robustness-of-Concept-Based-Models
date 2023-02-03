import numpy as np

import torch

from ..attack import Attack
from .deepfool_mask import DeepFool
from model_templates.utils_models import InverseNormalization, Normalization, index
from config import BIRD_IDX


class SparseFool(Attack):
    r"""
    Attack in the paper 'SparseFool: a few pixels make a big difference'
    [https://arxiv.org/abs/1811.02248]
    Modified from "https://github.com/LTS4/SparseFool/"
    Distance Measure : L0
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 20)
        lam (float): parameter for scaling DeepFool noise. (Default: 3)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        attack = torchattacks.SparseFool(model, steps=20, lam=3, overshoot=0.02)
        adv_images = attack(images, labels)
    """

    def __init__(self, model, steps=2, steps_deepfool=2, lam=3, overshoot=0.02, descision_threshold=0.3):
        super().__init__("SparseFool", model)
        self.steps = steps
        self.lam = lam
        self.overshoot = overshoot
        self.deepfool = DeepFool(model, steps=steps_deepfool)
        self._supported_mode = ['default']
        self.descision_threshold = descision_threshold
        self.model = model
        self.inverse_normalization = InverseNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalization = Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size = len(images)
        correct = torch.tensor([True] * batch_size)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx + 1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                image = images[idx:idx + 1]
                new_targets = labels[idx:idx + 1]

                self.model.eval()
                outputs, transformed_images, _ = self.model(images)  # eval mode
                transformed_images = transformed_images.tensors  # required to skip the normalization in the forward pass

                adv_image = self.inverse_normalization(transformed_images)

                threshold_idx = len([x for x in outputs[idx]["scores"] if x > self.descision_threshold])
                thresholded_pred = outputs[idx]["labels"][:threshold_idx]

                if BIRD_IDX not in thresholded_pred:
                    correct[idx] = False
                    continue

                """self.model.train()
                _, transformed_images, _ = self.model(images)  # eval mode
                transformed_images = transformed_images.tensors  # required to skip the normalization in the forward pass"""

                # TODO adv_image is not normalized and it is detached from the graph!
                adv_image, target_label = self.deepfool(adv_image, new_targets, return_target_labels=True)
                adv_image = transformed_images + self.lam * (self.normalization(adv_image).detach() - transformed_images)

                outputs, _, _ = self.model(images)  # eval mode
                threshold_idx = len([x for x in outputs[idx]["scores"] if x > self.descision_threshold])
                thresholded_pred = outputs[idx]["labels"][:threshold_idx]

                if BIRD_IDX in thresholded_pred:
                    pre = target_label

                # second part of substraction is the highest bird score
                if pre in outputs[idx]["scores"]:
                    score_prediction = outputs[idx]["scores"][pre]
                else:
                    score_prediction = [i for i, l in zip(outputs[idx]["scores"][:threshold_idx], thresholded_pred)
                                        if l != BIRD_IDX]
                    if len(score_prediction) > 0:
                        score_prediction = score_prediction[0]
                    else:
                        score_prediction = 1

                cost = score_prediction - outputs[idx]["scores"][index(thresholded_pred, BIRD_IDX)]

                print(cost)
                print(adv_image.unique(return_counts=True))

                # TODO adv_image needs to be in the graph
                grad = torch.autograd.grad(cost, adv_image, retain_graph=False, create_graph=False)[0]
                grad = grad / grad.norm()

                adv_image = self._linear_solver(image, grad, self.inverse_normalization(adv_image))
                adv_image = image + (1 + self.overshoot) * (adv_image - image)
                adv_images[idx] = torch.clamp(adv_image, min=0, max=1).detach()

            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()

        return adv_images

    def _linear_solver(self, x_0, coord_vec, boundary_point):
        input_shape = x_0.size()

        plane_normal = coord_vec.clone().detach().view(-1)
        plane_point = plane_normal.clone().detach().view(-1)

        x_i = x_0.clone().detach()

        f_k = torch.dot(plane_normal, x_0.view(-1) - plane_point)
        sign_true = f_k.sign().item()

        beta = 0.001 * sign_true
        current_sign = sign_true

        while current_sign == sign_true and coord_vec.nonzero().size()[0] > 0:
            f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point) + beta

            pert = f_k.abs() / coord_vec.abs().max()

            mask = torch.zeros_like(coord_vec)
            mask[np.unravel_index(torch.argmax(coord_vec.abs()).cpu(), input_shape)] = 1.

            r_i = torch.clamp(pert, min=1e-4) * mask * coord_vec.sign()

            x_i = x_i + r_i
            x_i = torch.clamp(x_i, min=0, max=1)

            f_k = torch.dot(plane_normal, x_i.view(-1) - plane_point)
            current_sign = f_k.sign().item()

            coord_vec[r_i != 0] = 0

        return x_i
