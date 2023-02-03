import torch
import torch.nn as nn

from ..attack import Attack
from model_templates.utils_models import InverseNormalization


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.007)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        attack = torchattacks.FGSM(model, eps=0.007)
        adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=0.007):
        super().__init__("FGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.inverse_normalization = InverseNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)
        else:
            target_labels = labels

        # output: losses, transformed and normalized images, new_targets -> transformed labels fitting to new img size
        self.model.train()
        loss_dict, transformed_images, new_targets = self.model(images, target_labels)  # train mode
        transformed_images = transformed_images.tensors # required to skip the normalization in the forward pass

        loss = sum(loss_dict.values())
        # Calculate loss
        if self._targeted:
            cost = -loss
        else:
            cost = loss
        # Update adversarial images
        grad = \
            torch.autograd.grad(cost, transformed_images,
                                retain_graph=False, create_graph=False)[0]

        # remove image normalization, since mask_rcnn does it by itself and we get the normalized images here
        adv_images = self.inverse_normalization(transformed_images) + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images