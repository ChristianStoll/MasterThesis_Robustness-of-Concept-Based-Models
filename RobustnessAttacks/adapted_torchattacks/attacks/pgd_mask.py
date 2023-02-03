import torch
import torch.nn as nn

from ..attack import Attack
from model_templates.utils_models import InverseNormalization


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        adv_images = attack(images, labels)
    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
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

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            # adv_images.requires_grad = True -> will be done in the forward pass of the mask model
            # output: losses, transformed and normalized images
            loss_dict, transformed_images, target_labels = self.model(adv_images, target_labels)  # train mode
            transformed_images = transformed_images.tensors

            # this is required because the preprocessing & transformation is done in the forward step
            if i == 0:
                adv_images = self.inverse_normalization(transformed_images).clone().detach()
                adv_images.requires_grad = True
                images = adv_images.clone().detach()

            loss = sum(loss_dict.values())
            # Calculate loss
            if self._targeted:
                cost = -loss
            else:
                cost = loss

            # Update adversarial images
            grad = torch.autograd.grad(cost, transformed_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
