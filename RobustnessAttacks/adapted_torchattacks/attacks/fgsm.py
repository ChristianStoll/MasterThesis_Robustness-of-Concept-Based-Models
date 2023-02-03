import torch
import torch.nn as nn

from utils import hungarian_loss
from ..attack import Attack


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
    def __init__(self, model, eps=0.007, use_segmentation=False, XtoC=''):
        super().__init__("FGSM", model)
        self.eps = eps
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
        elif 'mask' in self.XtoC.lower(): # TODO
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
                                   retain_graph=False, create_graph=False)[0]

        if self.use_segmentation:
            segmentations = segmentations.clone().detach().to(self.device)
            adv_images = images + segmentations * self.eps * grad.sign()
        else:
            adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
