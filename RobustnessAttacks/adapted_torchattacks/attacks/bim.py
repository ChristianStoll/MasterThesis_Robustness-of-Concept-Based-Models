import torch
import torch.nn as nn
from utils import hungarian_loss

from ..attack import Attack


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
    def __init__(self, model, eps=4/255, alpha=1/255, steps=0, use_segmentation=False, XtoC=''):
        super().__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        self._supported_mode = ['default', 'targeted']
        self.use_segmentation = use_segmentation
        self.XtoC = XtoC

    def forward(self, images, labels, segmentations=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()

        for _ in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)

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
            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False,
                                       create_graph=False)[0]

            if self.use_segmentation:
                segmentations = segmentations.clone().detach().to(self.device)
                adv_images = images + segmentations * self.alpha * grad.sign()
            else:
                adv_images = images + self.alpha * grad.sign()

            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float()*adv_images \
                + (adv_images < a).float()*a
            c = (b > ori_images+self.eps).float()*(ori_images+self.eps) \
                + (b <= ori_images + self.eps).float()*b
            images = torch.clamp(c, max=1).detach()

        return images
