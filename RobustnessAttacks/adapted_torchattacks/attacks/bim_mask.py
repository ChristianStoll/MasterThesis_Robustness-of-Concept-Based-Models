import torch
import torch.nn as nn

from ..attack import Attack
from model_templates.utils_models import InverseNormalization


class BIM(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 4/255)
        alpha (float): step size. (Default: 1/255)
        steps (int): number of steps. (Default: 0)
    .. note:: If steps set to 0, steps will be automatically decided following the paper.
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        attack = torchattacks.BIM(model, eps=4/255, alpha=1/255, steps=0)
        adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=4/255, alpha=1/255, steps=0):
        super().__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        self._supported_mode = ['default', 'targeted']
        self.inverse_normalization = InverseNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        self.model.train()

        if self._targeted:
            target_labels = self._get_target_label(images, labels)
        else:
            target_labels = labels

        ori_images = images.clone().detach()
        for i in range(self.steps):
            # images.requires_grad_() is set inside the forward pass of the model
            # output: losses, transformed images, targets_labels fitting the new image size
            # input are modified target labels for each step of the loop!
            loss_dict, transformed_images, target_labels = self.model(images, target_labels)  # train mode
            transformed_images = transformed_images.tensors

            if i == 0:
                images = self.inverse_normalization(transformed_images.clone().detach())
                images.requires_grad = True
                ori_images = images.clone().detach()

            loss = sum(loss_dict.values())

            # Calculate loss
            if self._targeted:
                cost = -loss
            else:
                cost = loss

            # Update adversarial images
            grad = torch.autograd.grad(cost, transformed_images, retain_graph=False, create_graph=False)[0]

            adv_images = images + self.alpha * grad.sign()
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float()*adv_images \
                + (adv_images < a).float()*a
            c = (b > ori_images+self.eps).float()*(ori_images+self.eps) \
                + (b <= ori_images + self.eps).float()*b
            images = torch.clamp(c, max=1).detach()

        return images
