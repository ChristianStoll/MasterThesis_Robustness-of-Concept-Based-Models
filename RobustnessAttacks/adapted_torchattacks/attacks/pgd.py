import torch
import torch.nn as nn
from utils import hungarian_loss

from ..attack import Attack


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
                 alpha=2/255, steps=40, random_start=True, use_segmentation=False, XtoC=''):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.use_segmentation = use_segmentation
        self.XtoC = XtoC

    def forward(self, images, labels, segmentations=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        self.model.train()

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # -------------------- adaptations from here --------------------
            if 'cb' in self.XtoC.lower():
                attr_criterion = []  # separate criterion (loss function) for each attribute
                for i in range(112):
                    attr_criterion.append(torch.nn.BCEWithLogitsLoss())

                losses = []
                for i in range(len(attr_criterion)):
                    if self._targeted:
                        losses.append(
                            -attr_criterion[i](outputs[i].squeeze().type(torch.cuda.FloatTensor), target_labels[:, i]))
                    else:
                        losses.append(
                            attr_criterion[i](outputs[i].squeeze().type(torch.cuda.FloatTensor), labels[:, i]))

                cost = sum(losses) / float(112)
            elif 'mask' in self.XtoC.lower():
                bce_loss = torch.nn.BCEWithLogitsLoss()
                if self._targeted:
                    cost = -bce_loss(outputs, target_labels)
                else:
                    cost = bce_loss(outputs, labels)

            else:
                # Calculate loss
                if self._targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
            # -------------------- until here --------------------
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            if self.use_segmentation:
                segmentations = segmentations.clone().detach().to(self.device)
                adv_images = adv_images.detach() + segmentations * self.alpha * grad.sign()
            else:
                adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
